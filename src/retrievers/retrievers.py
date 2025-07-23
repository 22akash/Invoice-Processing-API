"""
This module handles the retrieval of relevant information from the vector database.
"""

import logging
import time
from typing import Dict, Any, List, Tuple
from src.db.postgres import get_embedding_from_db, find_similar_images, get_summary_from_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def retrieve_invoice_data(
    image_path: str,
    question: str,
    request_id: str,
    top_k: int = 3
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Retrieve relevant invoice data based on the question.
    
    Args:
        image_path: Path to the invoice image
        question: Question about the invoice
        request_id: Unique identifier for the request
        top_k: Number of similar images to retrieve
        
    Returns:
        Tuple containing:
        - Dictionary with retrieval results
        - List of token usage details
    """
    start_time = time.time()
    token_usage_details_list = []
    
    # First, try to get the summary for the specific image directly
    # This ensures we prioritize the exact image the user is asking about
    direct_summary = get_summary_from_db(image_path)
    
    if direct_summary:
        logger.info(f"Found direct summary for the requested image: {image_path}")
        document_data = [{
            "image": image_path,
            "similarity": 1.0,  # Perfect match
            "matching_records": [
                {"content": str(direct_summary)}
            ]
        }]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "response_from_vector_db": "Success",
            "matching_documents_found": True,
            "diagnostic_message": {
                "diagnostic_message": f"Found direct summary for the requested image",
                "processing_time": processing_time
            },
            "document_data": document_data
        }, token_usage_details_list
    
    # If no direct summary, fall back to embedding-based retrieval
    logger.info(f"No direct summary found for {image_path}, using embedding-based retrieval")
    
    # Get embedding for the image
    embedding = get_embedding_from_db(image_path)
    if not embedding:
        logger.warning(f"No embedding found for image: {image_path}")
        return {
            "response_from_vector_db": None,
            "matching_documents_found": False,
            "diagnostic_message": {
                "diagnostic_message": f"No embedding found for image: {image_path}"
            },
            "document_data": None
        }, token_usage_details_list
    
    # Find similar images
    similar_images = find_similar_images(embedding, limit=top_k)
    if not similar_images:
        logger.warning(f"No similar images found for image: {image_path}")
        return {
            "response_from_vector_db": None,
            "matching_documents_found": False,
            "diagnostic_message": {
                "diagnostic_message": f"No similar images found for image: {image_path}"
            },
            "document_data": None
        }, token_usage_details_list
    
    # Get summaries for similar images
    document_data = []
    
    # First, check if the original image is in the similar images list
    original_image_in_results = False
    for similar_image in similar_images:
        if similar_image["image"] == image_path:
            original_image_in_results = True
            break
    
    # If the original image is not in the results, add it manually
    if not original_image_in_results and direct_summary:
        document_data.append({
            "image": image_path,
            "similarity": 1.0,  # Perfect match
            "matching_records": [
                {"content": str(direct_summary)}
            ]
        })
    
    # Process the similar images
    for similar_image in similar_images:
        image = similar_image["image"]
        similarity = similar_image["similarity"]
        
        # Skip if this is the original image and we've already added it
        if image == image_path and original_image_in_results and direct_summary:
            continue
        
        summary = get_summary_from_db(image)
        if summary:
            document_data.append({
                "image": image,
                "similarity": similarity,
                "matching_records": [
                    {"content": str(summary)}
                ]
            })
    
    if not document_data:
        logger.warning(f"No summaries found for similar images of: {image_path}")
        return {
            "response_from_vector_db": None,
            "matching_documents_found": False,
            "diagnostic_message": {
                "diagnostic_message": f"No summaries found for similar images of: {image_path}"
            },
            "document_data": None
        }, token_usage_details_list
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "response_from_vector_db": "Success",
        "matching_documents_found": True,
        "diagnostic_message": {
            "diagnostic_message": f"Found {len(document_data)} matching documents",
            "processing_time": processing_time
        },
        "document_data": document_data
    }, token_usage_details_list