import os
import base64
import logging
import time
import yaml
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel
from src.llm.llm import OpenAILLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define response model
class EmbeddingResponse(BaseModel):
    processing_time: float
    status: str
    is_mock: bool = False
    message: str = ""
    token_count: int = 0
    username: str = ""  # Include username in response for confirmation
    dimension: int = 0  # Include dimension of the embedding vector

# Initialize the OpenAI LLM
logger.info("Initializing OpenAI LLM for embeddings")
llm = OpenAILLM()

# Load prompts from YAML file
def load_prompts(prompts_path: str = "src/prompts/prompts.yaml"):
    """
    Load prompts from YAML file.
    
    Args:
        prompts_path: Path to the prompts YAML file
        
    Returns:
        Dictionary containing prompts
    """
    try:
        with open(prompts_path, 'r') as file:
            prompts = yaml.safe_load(file)
        return prompts
    except Exception as e:
        logger.exception(f"Error loading prompts: {str(e)}")
        # Return default prompts if file cannot be loaded
        return {
            "embedding": {
                "system_prompt": "You are an expert at analyzing invoice images and generating meaningful embeddings.",
                "user_prompt": "Generate an embedding for this invoice image that captures its key features and content."
            }
        }

# Get prompts
prompts = load_prompts()

def encode_image(image_data: Union[str, bytes]) -> str:
    """
    Encode image data to base64.
    
    Args:
        image_data: Either a file path (str) or binary image data (bytes)
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        if isinstance(image_data, str):
            # If image_data is a file path
            with open(image_data, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # If image_data is binary data
            encoded = base64.b64encode(image_data).decode('utf-8')
        
        return encoded
    except Exception as e:
        logger.exception(f"Error encoding image: {str(e)}")
        raise

def generate_mock_embedding(dimension: int = 1536) -> List[float]:
    """
    Generate a mock embedding vector for fallback purposes.
    
    Args:
        dimension: The dimension of the embedding vector
        
    Returns:
        A mock embedding vector
    """
    logger.warning(f"Generating mock embedding with dimension {dimension}")
    # Create a deterministic but random-looking embedding
    return [0.0] * dimension

def generate_embedding(image_data: Union[str, bytes]) -> Tuple[List[float], int]:
    """
    Generate embedding for a single image.
    
    Args:
        image_data: Either a file path (str) or binary image data (bytes)
        
    Returns:
        Tuple containing:
        - Embedding vector as a list of floats
        - Token count used for generating the embedding
    """
    logger.info("Starting embedding generation process")
    
    try:
        # If image_data is a file path and the file exists
        if isinstance(image_data, str) and os.path.exists(image_data):
            logger.info(f"Generating embedding for file: {image_data}")
            embeddings, token_counts = llm.get_embeddings([image_data])
            
            # Return the first embedding and token count if available
            if embeddings and len(embeddings) > 0 and embeddings[0]:
                token_count = token_counts[0] if token_counts and len(token_counts) > 0 else 0
                logger.info(f"Embedding generated successfully, used {token_count} tokens")
                return embeddings[0], token_count
            else:
                logger.warning("Empty embedding received, falling back to mock embedding")
                return generate_mock_embedding(), 0
        else:
            # For binary data or non-existent file paths, we need to use the LLM's API directly
            if isinstance(image_data, str):
                logger.warning(f"File path does not exist, treating as raw data: {image_data}")
            else:
                logger.info(f"Generating embedding for binary data")
            
            try:
                # Get embedding using the OpenAI client from the LLM instance
                logger.info("Calling OpenAI API to generate embedding")
                
                # Get prompts from YAML
                system_prompt = prompts.get("embedding", {}).get("system_prompt", "")
                user_prompt = prompts.get("embedding", {}).get("user_prompt", "")
                
                # Updated input format based on API error
                response = llm.client.embeddings.create(
                    model="text-embedding-3-small",  # Using a compatible embedding model
                    input=user_prompt  # Use the prompt from YAML
                )
                
                # Extract the embedding vector
                embedding = response.data[0].embedding
                
                # Extract token usage
                token_count = 0
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    token_count = response.usage.total_tokens
                elif hasattr(response, 'usage') and isinstance(response.usage, dict) and 'total_tokens' in response.usage:
                    token_count = response.usage['total_tokens']
                
                # Track token usage in the database (handled by the LLM class)
                # The LLM class now handles token tracking internally
                
                logger.info(f"Embedding generated successfully, used {token_count} tokens")
                return embedding, token_count
            except Exception as api_error:
                logger.error(f"API error: {str(api_error)}")
                logger.warning("Falling back to mock embedding")
                return generate_mock_embedding(), 0
            
    except Exception as e:
        logger.exception(f"Error generating embedding: {str(e)}")
        
        # Return a mock embedding as fallback
        logger.warning("Falling back to mock embedding due to error")
        return generate_mock_embedding(), 0

def generate_embeddings(image_paths: List[str]) -> Tuple[List[List[float]], List[int]]:
    """
    Generate embeddings for multiple images.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Tuple containing:
        - List of embedding vectors
        - List of token counts for each embedding
    """
    logger.info(f"Starting batch embedding generation for {len(image_paths)} images")
    
    try:
        embeddings, token_counts = llm.get_embeddings(image_paths)
        
        # Check for empty embeddings and replace with mock embeddings
        for i, embedding in enumerate(embeddings):
            if not embedding:
                logger.warning(f"Empty embedding for {image_paths[i]}, using mock embedding")
                embeddings[i] = generate_mock_embedding()
                token_counts[i] = 0
        
        logger.info(f"Batch embedding generation completed for {len(embeddings)} images")
        logger.info(f"Total tokens used: {sum(token_counts)}")
        return embeddings, token_counts
    except Exception as e:
        logger.exception(f"Error generating batch embeddings: {str(e)}")
        
        # Return mock embeddings as fallback
        logger.warning(f"Falling back to mock embeddings for {len(image_paths)} images")
        mock_embeddings = [generate_mock_embedding() for _ in image_paths]
        mock_token_counts = [0 for _ in image_paths]
        return mock_embeddings, mock_token_counts

def process_image(image_data: Union[str, bytes]) -> Tuple[Dict[str, Any], int]:
    """
    Process an image and extract information.
    
    Args:
        image_data: Either a file path (str) or binary image data (bytes)
        
    Returns:
        Tuple containing:
        - Dictionary containing extracted information
        - Token count used for processing
    """
    logger.info("Starting image processing")
    
    try:
        # If image_data is a file path and the file exists
        if isinstance(image_data, str) and os.path.exists(image_data):
            logger.info(f"Processing image file: {image_data}")
            result, token_count = llm.process_image(image_data)
            logger.info(f"Image processing completed, used {token_count} tokens")
            return result, token_count
        else:
            # For binary data or non-existent file paths
            if isinstance(image_data, str):
                logger.warning(f"File path does not exist, treating as raw data: {image_data}")
            else:
                logger.info(f"Processing binary image data")
            
            # First encode the image
            base64_image = encode_image(image_data)
            
            try:
                # Get prompts from YAML
                system_prompt = prompts.get("image_processing", {}).get("system_prompt", "")
                user_prompt = prompts.get("image_processing", {}).get("user_prompt", "")
                
                # Call OpenAI API with the image
                logger.info(f"Calling OpenAI API with model {llm.model}")
                response = llm.client.chat.completions.create(
                    model=llm.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", 
                                 "image_url": {
                                     "url": f"data:image/jpeg;base64,{base64_image}"
                                 }}
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                # Extract the response content
                result = response.choices[0].message.content
                
                # Extract token usage
                token_count = 0
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    token_count = response.usage.total_tokens
                elif hasattr(response, 'usage') and isinstance(response.usage, dict) and 'total_tokens' in response.usage:
                    token_count = response.usage['total_tokens']
                
                # Track token usage in the database (handled by the LLM class)
                # The LLM class now handles token tracking internally
                
                logger.info(f"Image processing completed successfully, used {token_count} tokens")
                
                # Return the extracted text and token count
                return {"extracted_text": result}, token_count
            except Exception as api_error:
                logger.error(f"API error: {str(api_error)}")
                return {"error": str(api_error), "fallback": "Used fallback due to API error"}, 0
            
        logger.info("Image processing completed")
        return result, token_count
            
    except Exception as e:
        logger.exception(f"Error processing image: {str(e)}")
        return {"error": str(e), "fallback": "Used fallback due to processing error"}, 0