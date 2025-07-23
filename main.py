import logging
import time
import os
import yaml
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import shutil
import tempfile
from src.utils.embeddings import generate_embedding, EmbeddingResponse
from src.utils.summary import extract_invoice_data, SummaryResponse
from src.utils.invoice_analyzer import analyze_invoice
from src.utils.clustering import process_and_cluster_images
from src.utils.zip_handler import extract_zip, create_zip, save_uploaded_zip, cleanup_temp_files
from src.llm.llm import ImagePathRequest
from src.db.postgres import (
    track_api_call,
    store_embedding_in_db,
    get_embedding_from_db,
    store_summary_in_db,
    get_summary_from_db,
    store_cluster_job,
    update_cluster_job,
    get_cluster_job,
    SYSTEM_USERNAME
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "credentials/secrets.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

config = load_config()
openai_config = config.get('openai', {})
MODEL = openai_config.get('model', 'gpt-4o')  # Updated default model to match test.py

# Initialize FastAPI app
app = FastAPI(title="Invoice Processing API")
logger.info("Initializing Invoice Processing API")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directories for uploads, outputs, and static files
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request model for invoice RAG
class InvoiceRAGRequest(BaseModel):
    file_path: str
    question: str

# Define response model for invoice RAG
class InvoiceRAGResponse(BaseModel):
    image_path: str
    request_id: str
    question: str
    elapsed_time: float
    retrieval_results: Dict[str, Any]
    responses: List[Dict[str, Any]]
    most_relevant_answer: Dict[str, Any]

# Define response model for clustering
class ClusteringResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None
    output_zip_url: Optional[str] = None
    num_clusters: Optional[int] = None
    num_images_processed: Optional[int] = None
    processing_time: Optional[float] = None
    token_usage: Optional[int] = None

# Define request model for clustering status
class ClusteringStatusRequest(BaseModel):
    job_id: str

# Store background jobs
clustering_jobs = {}

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Invoice Processing API is running", "status": "active"}

@app.post("/generate_embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: ImagePathRequest):
    """
    Generate embedding for an image at the specified file path.
    The embedding is stored in the database but not returned in the response.
    If an embedding already exists for the image, it will be updated.
    
    Args:
        request: The request containing the image file path
        
    Returns:
        Processing time, status, and token count information
    """
    start_time = time.time()
    username = SYSTEM_USERNAME
    use_case = "generate_embeddings"  # Use case is the API endpoint name
    logger.info(f"Processing file for embeddings: {request.file_path} for user: {username}")
    
    try:
        file_path = request.file_path
        
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Generate embedding - now returns both embedding and token count
        embedding, token_count = generate_embedding(file_path)
        
        # Check if we got a valid embedding
        is_mock = all(v == 0.0 for v in embedding)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Track API call in database with actual token count
        track_api_call(username, file_path, MODEL, token_count=token_count, use_case=use_case)
        
        # Store embedding in database (if not mock)
        # This will update the embedding if it already exists
        if not is_mock:
            store_embedding_in_db(username, file_path, embedding)
            logger.info(f"Stored/updated embedding for {file_path}")
        
        if is_mock:
            logger.warning(f"Generated mock embedding for {file_path}")
            return EmbeddingResponse(
                processing_time=processing_time,
                status="warning",
                is_mock=True,
                message="API connection failed, using mock embedding",
                token_count=token_count,
                username=username,
                dimension=len(embedding)
            )
        else:
            logger.info(f"Successfully generated embedding for {file_path}, used {token_count} tokens")
            return EmbeddingResponse(
                processing_time=processing_time,
                status="success",
                is_mock=False,
                message="Embedding generated successfully",
                token_count=token_count,
                username=username,
                dimension=len(embedding)
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full exception with traceback
        logger.exception(f"Error generating embedding: {str(e)}")
        processing_time = time.time() - start_time
        
        # Try to generate a mock embedding as fallback
        try:
            from src.utils.embeddings import generate_mock_embedding
            mock_embedding = generate_mock_embedding()
            
            # No tokens used for mock embeddings
            token_count = 0
            
            # Track API call in database (failed)
            track_api_call(username, file_path, MODEL, token_count=token_count, use_case=use_case)
            
            logger.warning(f"Using mock embedding after error for {request.file_path}")
            return EmbeddingResponse(
                processing_time=processing_time,
                status="error",
                is_mock=True,
                message=f"Error: {str(e)}. Using mock embedding as fallback.",
                token_count=token_count,
                username=username,
                dimension=len(mock_embedding)
            )
        except Exception as mock_error:
            logger.exception(f"Error generating mock embedding: {str(mock_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating embedding: {str(e)}"
            )

@app.post("/get_summary", response_model=SummaryResponse)
async def get_summary(request: ImagePathRequest):
    """
    Extract invoice data from an image at the specified file path.
    Uses the same approach as in test.py.
    Always generates and updates embeddings for the image.
    
    Args:
        request: The request containing the image file path
        
    Returns:
        The extracted invoice data, processing time, and status
    """
    start_time = time.time()
    username = SYSTEM_USERNAME
    use_case = "get_summary"  # Use case is the API endpoint name
    logger.info(f"Processing file for invoice data extraction: {request.file_path} for user: {username}")
    
    try:
        file_path = request.file_path
        
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Check if summary already exists in database
        existing_summary = get_summary_from_db(file_path)
        if existing_summary:
            logger.info(f"Using existing invoice data from database for {file_path}")
            
            # Generate new embedding for the image and update it in the database
            embedding, embedding_token_count = generate_embedding(file_path)
            if not all(v == 0.0 for v in embedding):
                store_embedding_in_db(username, file_path, embedding)
                logger.info(f"Updated embedding for {file_path} even though using existing summary")
            
            processing_time = time.time() - start_time
            
            # Track API call in database (using cached summary)
            track_api_call(username, file_path, MODEL, token_count=embedding_token_count, use_case=use_case)
            
            return SummaryResponse(
                summary=existing_summary,
                processing_time=processing_time,
                status="success",
                is_mock=False,
                message="Invoice data retrieved from database, embedding updated",
                token_count=embedding_token_count,
                username=username
            )
        
        # Generate embedding for the image (for future similarity searches)
        embedding, embedding_token_count = generate_embedding(file_path)
        
        # Extract invoice data using the approach from test.py
        invoice_data, extraction_token_count = extract_invoice_data(file_path)
        
        # Calculate total token count
        token_count = embedding_token_count + extraction_token_count
        
        # Check if we got valid invoice data
        is_mock = invoice_data.get("vendor_name") == "Mock Vendor Inc."
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Track API call in database with actual token count
        track_api_call(username, file_path, MODEL, token_count=token_count, use_case=use_case)
        
        # Store data in database (if not mock)
        if not is_mock:
            # Store invoice data
            store_summary_in_db(username, file_path, invoice_data)
            
            # Always store/update the embedding
            if not all(v == 0.0 for v in embedding):
                store_embedding_in_db(username, file_path, embedding)
                logger.info(f"Stored/updated embedding for {file_path}")
        
        if is_mock:
            logger.warning(f"Generated mock invoice data for {file_path}")
            return SummaryResponse(
                summary=invoice_data,
                processing_time=processing_time,
                status="warning",
                is_mock=True,
                message="API connection failed, using mock invoice data",
                token_count=token_count,
                username=username
            )
        else:
            logger.info(f"Successfully extracted invoice data for {file_path}, used {token_count} tokens")
            return SummaryResponse(
                summary=invoice_data,
                processing_time=processing_time,
                status="success",
                is_mock=False,
                message="Invoice data extracted successfully",
                token_count=token_count,
                username=username
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full exception with traceback
        logger.exception(f"Error extracting invoice data: {str(e)}")
        processing_time = time.time() - start_time
        
        # Try to generate mock invoice data as fallback
        try:
            from src.utils.summary import generate_mock_summary
            mock_invoice_data = generate_mock_summary()
            
            # No tokens used for mock data
            token_count = 0
            
            # Track API call in database (failed)
            track_api_call(username, file_path, MODEL, token_count=token_count, use_case=use_case)
            
            logger.warning(f"Using mock invoice data after error for {request.file_path}")
            return SummaryResponse(
                summary=mock_invoice_data,
                processing_time=processing_time,
                status="error",
                is_mock=True,
                message=f"Error: {str(e)}. Using mock invoice data as fallback.",
                token_count=token_count,
                username=username
            )
        except Exception as mock_error:
            logger.exception(f"Error generating mock invoice data: {str(mock_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error extracting invoice data: {str(e)}"
            )

@app.post("/invoice_rag_response")
async def invoice_rag_response(request: Request):
    """
    Handle the invoice RAG response request.
    Takes an invoice image path and a question about the invoice,
    and returns an answer based on the invoice data.
    
    Args:
        request: The request containing the image path and question
        
    Returns:
        JSON response with the answer to the question
    """
    try:
        # Parse request data
        request_data = await request.json()
        file_path = request_data.get('file_path')
        question = request_data.get('question')
        
        # Validate request data
        if not file_path:
            raise HTTPException(status_code=400, detail="Missing file_path in request")
        if not question:
            raise HTTPException(status_code=400, detail="Missing question in request")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Generate a unique request ID
        username = SYSTEM_USERNAME
        request_id = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Start timing
        start_time = time.time()
        
        # Check if summary exists in database, if not, generate it
        existing_summary = get_summary_from_db(file_path)
        if not existing_summary:
            logger.info(f"No summary found for {file_path}, generating one")
            # Generate embedding
            embedding, embedding_token_count = generate_embedding(file_path)
            
            # Extract invoice data
            invoice_data, extraction_token_count = extract_invoice_data(file_path)
            
            # Store data in database
            if invoice_data.get("vendor_name") != "Mock Vendor Inc.":
                store_summary_in_db(username, file_path, invoice_data)
                store_embedding_in_db(username, file_path, embedding)
        else:
            logger.info(f"Using existing summary for {file_path}")
            # Update embedding
            embedding, embedding_token_count = generate_embedding(file_path)
            store_embedding_in_db(username, file_path, embedding)
        
        # Analyze the invoice based on the question
        analysis_results, token_usage_details = analyze_invoice(
            image_path=file_path,
            question=question,
            request_id=request_id,
            username=username
        )
        
        # Track token usage in database
        for token_usage in token_usage_details:
            track_api_call(
                username=token_usage.get("username", username),
                image=file_path,
                model=token_usage.get("model", MODEL),
                token_count=token_usage.get("token_count", 0),
                use_case=token_usage.get("request_reason", "invoice_rag_response")
            )
        
        # Calculate total elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Update elapsed time in results
        analysis_results["elapsed_time"] = elapsed_time
        
        # Parse the most_relevant_answer from JSON string to dict
        try:
            analysis_results["most_relevant_answer"] = json.loads(analysis_results["most_relevant_answer"])
            # Ensure the answer doesn't contain nested JSON or code blocks
            if isinstance(analysis_results["most_relevant_answer"]["answer"], str):
                import re
                # Remove any markdown code block formatting
                analysis_results["most_relevant_answer"]["answer"] = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', analysis_results["most_relevant_answer"]["answer"])
                # Remove any JSON formatting within the answer
                analysis_results["most_relevant_answer"]["answer"] = re.sub(r'^\s*{\s*"answer"\s*:\s*"(.+?)"\s*,.*$', r'\1', analysis_results["most_relevant_answer"]["answer"])
        except Exception as e:
            logger.warning(f"Error parsing most_relevant_answer as JSON: {str(e)}")
        
        return JSONResponse(content=analysis_results, status_code=200)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full exception with traceback
        logger.exception(f"Error in invoice_rag_response: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server error: {str(e)}"
        )

def process_clustering_job(job_id: str, zip_file_path: str, n_clusters: Optional[int], method: str, visualize: bool):
    """
    Background task to process clustering job.
    
    Args:
        job_id: Unique identifier for the job
        zip_file_path: Path to the uploaded zip file
        n_clusters: Number of clusters (if None, will be determined automatically)
        method: Clustering method ('kmeans' or 'dbscan')
        visualize: Whether to create a visualization of the clusters
    """
    try:
        # Update job status in memory
        clustering_jobs[job_id]["status"] = "processing"
        
        # Update job status in database
        update_cluster_job(job_id, "processing")
        
        # Extract the zip file
        extract_dir, extracted_images = extract_zip(zip_file_path)
        
        if not extracted_images:
            # Update job status in memory
            clustering_jobs[job_id].update({
                "status": "failed",
                "message": "No image files found in the zip file"
            })
            
            # Update job status in database
            update_cluster_job(job_id, "failed")
            return
        
        # Create output directory
        output_dir = os.path.join("output", f"clusters_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process and cluster the images
        start_time = time.time()
        results = process_and_cluster_images(
            image_paths=extracted_images,
            output_dir=output_dir,
            n_clusters=n_clusters,
            method=method,
            visualize=visualize
        )
        processing_time = time.time() - start_time
        
        # Create a zip file of the clustered images
        output_zip_path = create_zip(output_dir)
        
        # Copy the zip file to the static directory for download
        static_zip_path = os.path.join("static", f"clusters_{job_id}.zip")
        shutil.copy2(output_zip_path, static_zip_path)
        
        # Update job status in memory
        clustering_jobs[job_id].update({
            "status": "completed",
            "message": "Clustering completed successfully",
            "output_zip_url": f"/static/clusters_{job_id}.zip",
            "num_clusters": results.get("num_clusters", 0),
            "num_images_processed": results.get("num_images_processed", 0),
            "processing_time": processing_time,
            "token_usage": results.get("token_usage", 0)
        })
        
        # Update job status in database
        update_cluster_job(
            job_id=job_id,
            status="completed",
            num_clusters=results.get("num_clusters", 0),
            processing_time=processing_time
        )
        
        # Clean up temporary files
        cleanup_temp_files([zip_file_path], [extract_dir])
        
    except Exception as e:
        logger.exception(f"Error processing clustering job {job_id}: {str(e)}")
        
        # Update job status in memory
        clustering_jobs[job_id].update({
            "status": "failed",
            "message": f"Error processing clustering job: {str(e)}"
        })
        
        # Update job status in database
        update_cluster_job(job_id, "failed")

@app.post("/cluster_invoice", response_model=ClusteringResponse)
async def cluster_invoice(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    n_clusters: Optional[int] = Form(None),
    method: str = Form("kmeans"),
    visualize: bool = Form(True)
):
    """
    Cluster invoice images based on their embeddings.
    
    Args:
        background_tasks: FastAPI background tasks
        zip_file: Uploaded zip file containing invoice images
        n_clusters: Number of clusters (if None, will be determined automatically)
        method: Clustering method ('kmeans' or 'dbscan')
        visualize: Whether to create a visualization of the clusters
        
    Returns:
        Job ID and status information
    """
    try:
        # Generate a unique job ID
        job_id = str(int(time.time()))
        
        # Get the file name from the uploaded file
        file_name = zip_file.filename or f"upload_{job_id}.zip"
        
        # Read the zip file content
        zip_content = await zip_file.read()
        
        # Save the uploaded zip file
        zip_file_path = os.path.join("uploads", f"invoice_upload_{job_id}.zip")
        with open(zip_file_path, "wb") as f:
            f.write(zip_content)
        
        # Create job entry in memory
        clustering_jobs[job_id] = {
            "status": "queued",
            "message": "Job queued for processing",
            "created_at": datetime.now().isoformat(),
            "zip_file_path": zip_file_path
        }
        
        # Count the number of images in the zip file
        # Use a temporary file to avoid file not found errors
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(zip_content)
            temp_file_path = temp_file.name
        
        try:
            # Extract the zip file to count images
            extract_dir, extracted_images = extract_zip(temp_file_path, extract_files=False)
            num_images = len(extracted_images)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Error counting images in zip file: {str(e)}")
            # Default to 0 images if counting fails
            num_images = 0
            # Clean up the temporary file if it exists
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Store job information in the database
        store_cluster_job(
            job_id=job_id,
            file_name=file_name,
            images=num_images,
            embedding_model=MODEL,
            cluster_model=method
        )
        
        # Start background task
        background_tasks.add_task(
            process_clustering_job,
            job_id=job_id,
            zip_file_path=zip_file_path,
            n_clusters=n_clusters,
            method=method,
            visualize=visualize
        )
        
        return ClusteringResponse(
            status="queued",
            message="Clustering job queued for processing",
            job_id=job_id
        )
    
    except Exception as e:
        logger.exception(f"Error starting clustering job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting clustering job: {str(e)}"
        )

@app.get("/cluster_status/{job_id}", response_model=ClusteringResponse)
async def cluster_status(job_id: str):
    """
    Get the status of a clustering job.
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Job status information
    """
    if job_id not in clustering_jobs:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    job_info = clustering_jobs[job_id]
    
    return ClusteringResponse(
        status=job_info.get("status", "unknown"),
        message=job_info.get("message", ""),
        job_id=job_id,
        output_zip_url=job_info.get("output_zip_url"),
        num_clusters=job_info.get("num_clusters"),
        num_images_processed=job_info.get("num_images_processed"),
        processing_time=job_info.get("processing_time"),
        token_usage=job_info.get("token_usage")
    )

@app.get("/download_clusters/{job_id}")
async def download_clusters(job_id: str):
    """
    Download the clustered images as a zip file.
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Zip file for download
    """
    if job_id not in clustering_jobs:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    job_info = clustering_jobs[job_id]
    
    if job_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")
    
    zip_path = os.path.join("static", f"clusters_{job_id}.zip")
    
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"Zip file for job {job_id} not found")
    
    return FileResponse(
        path=zip_path,
        filename=f"invoice_clusters_{job_id}.zip",
        media_type="application/zip"
    )

@app.on_event("startup")
async def startup_event():
    """Log when the application starts up."""
    logger.info("Application startup complete")
    logger.info(f"System username: {SYSTEM_USERNAME}")
    
    # Test the database connection
    try:
        from src.db.postgres import get_db_instance
        db = get_db_instance()
        logger.info("Database connection test successful")
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        logger.warning("Database features may not be fully functional")
    
    # Test the OpenAI connection at startup
    try:
        logger.info("Testing OpenAI connection...")
        test_file = "data/1.jpg"
        
        if os.path.exists(test_file):
            _, token_count = generate_embedding(test_file)
            logger.info(f"OpenAI connection test successful, used {token_count} tokens")
        else:
            logger.warning(f"Test file not found: {test_file}")
    except Exception as e:
        logger.error(f"OpenAI connection test failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Log when the application shuts down."""
    logger.info("Application shutdown")
    
    # Close database connection
    try:
        from src.db.postgres import get_db_instance
        db = get_db_instance()
        db.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Invoice Processing API server on http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
    