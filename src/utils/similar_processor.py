import os
import logging
import yaml
import json
from typing import List, Dict, Any, Optional, Tuple
from src.utils.embeddings import generate_embedding
from src.db.postgres import get_embedding_from_db, find_similar_images
from src.llm.llm import OpenAILLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI LLM
logger.info("Initializing OpenAI LLM for similar image processing")
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
            "similar_image": {
                "system_prompt": "You are an expert at analyzing invoices. Use information about similar invoices to help extract details from the current invoice more accurately.",
                "user_prompt": "Based on the similar invoices I've provided, extract all key information from this new invoice image."
            }
        }

# Get prompts
prompts = load_prompts()

def process_with_similar_images(image_path: str, limit: int = 3) -> Tuple[Dict[str, Any], int]:
    """
    Process an invoice image using information from similar images.
    
    Args:
        image_path: Path to the image file
        limit: Maximum number of similar images to consider
        
    Returns:
        Tuple containing:
        - Summary dictionary with invoice details
        - Token count used for processing
    """
    logger.info(f"Processing image with similar images: {image_path}")
    
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return {}, 0
        
        # Get or generate embedding for the image
        embedding = get_embedding_from_db(image_path)
        if not embedding:
            embedding, token_count = generate_embedding(image_path)
        else:
            token_count = 0
            
        # Find similar images
        similar_results = find_similar_images(embedding, limit)
        
        if not similar_results:
            logger.info(f"No similar images found for {image_path}")
            return {}, token_count
            
        # Get summaries for similar images
        similar_summaries = []
        for result in similar_results:
            similar_image = result["image"]
            similarity = result["similarity"]
            
            # Get summary from database
            from src.db.postgres import get_summary_from_db
            summary = get_summary_from_db(similar_image)
            
            if summary:
                similar_summaries.append({
                    "image": similar_image,
                    "similarity": similarity,
                    "summary": summary
                })
        
        if not similar_summaries:
            logger.info(f"No summaries found for similar images of {image_path}")
            return {}, token_count
            
        # Process the image with context from similar images
        result, additional_tokens = process_image_with_context(image_path, similar_summaries)
        
        # Update token count
        total_tokens = token_count + additional_tokens
        
        return result, total_tokens
        
    except Exception as e:
        logger.exception(f"Error processing with similar images: {str(e)}")
        return {}, 0

def process_image_with_context(image_path: str, similar_summaries: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
    """
    Process an image with context from similar images.
    
    Args:
        image_path: Path to the image file
        similar_summaries: List of summaries from similar images
        
    Returns:
        Tuple containing:
        - Extracted information as a dictionary
        - Token count used
    """
    try:
        # Encode the image to base64
        base64_image = llm._encode_image(image_path)
        
        # Prepare context from similar summaries
        context = "Here are summaries from similar invoices that might help you extract information more accurately:\n\n"
        for i, item in enumerate(similar_summaries):
            context += f"Similar Invoice {i+1} (Similarity: {item['similarity']:.2f}):\n"
            context += json.dumps(item["summary"], indent=2)
            context += "\n\n"
            
        # Get prompts
        system_prompt = prompts.get("similar_image", {}).get("system_prompt", "")
        user_prompt = prompts.get("similar_image", {}).get("user_prompt", "")
        
        # Call OpenAI API with the image and context
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
                        {"type": "text", "text": f"{user_prompt}\n\nContext from similar invoices:\n{context}"},
                        {"type": "image_url", 
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{base64_image}"
                         }}
                    ]
                }
            ],
            max_tokens=2000
        )
        
        # Extract the response content
        result = response.choices[0].message.content
        
        # Extract token usage
        token_count = 0
        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
            token_count = response.usage.total_tokens
        elif hasattr(response, 'usage') and isinstance(response.usage, dict) and 'total_tokens' in response.usage:
            token_count = response.usage['total_tokens']
        
        # Track token usage
        llm._track_token_usage(image_path, token_count, "llm_similar_image_processing")
        
        # Parse the result to extract JSON
        from src.utils.summary import extract_json_from_result
        summary = extract_json_from_result(result)
        
        logger.info(f"Processed image with context from similar images, used {token_count} tokens")
        
        return summary, token_count
        
    except Exception as e:
        logger.exception(f"Error processing image with context: {str(e)}")
        return {}, 0