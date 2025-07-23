import os
import json
import logging
import time
import yaml
from typing import Dict, Any, Optional, Union, Tuple, List
from pydantic import BaseModel
from src.llm.llm import OpenAILLM
from src.utils.embeddings import generate_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define response model
class SummaryResponse(BaseModel):
    summary: Dict[str, Any]
    processing_time: float
    status: str
    is_mock: bool = False
    message: str = ""
    token_count: int = 0
    username: str = ""  # Include username in response for confirmation

# Initialize the OpenAI LLM
logger.info("Initializing OpenAI LLM for summary generation")
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
            "summary": {
                "system_prompt": "You are an intelligent document processing assistant.",
                "user_prompt": "Extract all key information from this invoice image."
            }
        }

# Get prompts
prompts = load_prompts()

def generate_mock_summary() -> Dict[str, Any]:
    """
    Generate a mock summary for fallback purposes.
    
    Returns:
        A mock summary dictionary
    """
    logger.warning("Generating mock summary")
    return {
        "vendor_name": "Mock Vendor Inc.",
        "invoice_number": "INV-12345",
        "invoice_date": "2025-01-01",
        "due_date": "2025-01-31",
        "billing_address": "123 Mock Street, Mock City, MC 12345",
        "shipping_address": None,
        "line_items": [
            {
                "description": "Mock Product",
                "quantity": "1",
                "unit_price": "100.00",
                "total_price": "100.00"
            }
        ],
        "subtotal": "100.00",
        "taxes": "10.00",
        "total_amount": "110.00",
        "currency": "USD"
    }

def extract_invoice_data(image_path: str) -> Tuple[Dict[str, Any], int]:
    """
    Extract invoice data using the same approach as in test.py.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing:
        - Summary dictionary with invoice details
        - Token count used for processing
    """
    logger.info(f"Extracting invoice data from {image_path}")
    
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return generate_mock_summary(), 0
        
        # Encode the image to base64
        base64_image = llm._encode_image(image_path)
        
        # Get prompt from YAML
        prompt = prompts.get("summary", {}).get("system_prompt", "")
        
        # Call OpenAI API with the image - exactly like in test.py
        logger.info(f"Calling OpenAI API with model {llm.model} for invoice data extraction")
        
        response = llm.client.chat.completions.create(
            model="gpt-4o",  # Use the exact model name from test.py
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        # Extract the response content
        result = response.choices[0].message.content
        
        # Extract token usage
        token_count = 0
        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
            token_count = response.usage.total_tokens
        elif hasattr(response, 'usage') and isinstance(response.usage, dict) and 'total_tokens' in response.usage:
            token_count = response.usage['total_tokens']
        
        # Track token usage in the database
        llm._track_token_usage(image_path, token_count, "llm_invoice_extraction")
        
        logger.info(f"Invoice data extraction completed successfully, used {token_count} tokens")
        
        # Parse the result to extract the JSON
        extracted_data = extract_json_from_result(result)
        
        return extracted_data, token_count
        
    except Exception as e:
        logger.exception(f"Error extracting invoice data: {str(e)}")
        return generate_mock_summary(), 0

def generate_summary_with_embedding(image_path: str) -> Tuple[Dict[str, Any], int, List[float]]:
    """
    Generate a summary for an invoice image using both direct LLM processing and embeddings.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing:
        - Summary dictionary with invoice details
        - Token count used for generating the summary
        - Embedding vector for the image
    """
    logger.info(f"Starting enhanced summary generation for {image_path}")
    
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return generate_mock_summary(), 0, []
        
        # First, generate embeddings for the image
        logger.info(f"Generating embeddings for {image_path} to enhance summary")
        embedding, embedding_token_count = generate_embedding(image_path)
        
        # Extract invoice data using the approach from test.py
        summary, extraction_token_count = extract_invoice_data(image_path)
        
        # Calculate total token count
        total_token_count = embedding_token_count + extraction_token_count
        
        logger.info(f"Enhanced summary generated successfully for {image_path}, used {total_token_count} tokens")
        return summary, total_token_count, embedding
        
    except Exception as e:
        logger.exception(f"Error generating enhanced summary: {str(e)}")
        
        # Return a mock summary as fallback
        logger.warning("Falling back to mock summary due to error")
        return generate_mock_summary(), 0, []

def generate_summary(image_path: str) -> Tuple[Dict[str, Any], int]:
    """
    Generate a summary for an invoice image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing:
        - Summary dictionary with invoice details
        - Token count used for generating the summary
    """
    # Call the enhanced version and discard the embedding
    summary, token_count, _ = generate_summary_with_embedding(image_path)
    return summary, token_count

def process_image_for_summary(image_path: str, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    """
    Process an image to generate a summary using the LLM.
    
    Args:
        image_path: Path to the image file
        system_prompt: The system prompt for the LLM
        user_prompt: The user prompt for the LLM
        
    Returns:
        Tuple containing:
        - The LLM's response text
        - Token count used
    """
    try:
        # Encode the image to base64
        with open(image_path, "rb") as image_file:
            base64_image = llm._encode_image(image_path)
        
        # Call OpenAI API with the image
        logger.info(f"Calling OpenAI API with model {llm.model} for summary generation")
        
        response = llm.client.chat.completions.create(
            model="gpt-4o",  # Use the exact model name from test.py
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
            temperature=0.2,
            max_tokens=1500
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
        llm._track_token_usage(image_path, token_count, "llm_summary_generation")
        
        logger.info(f"Image processing for summary completed successfully, used {token_count} tokens")
        
        return result, token_count
        
    except Exception as e:
        logger.exception(f"Error processing image for summary: {str(e)}")
        return "", 0

def extract_json_from_result(result: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the LLM's response.
    
    Args:
        result: The LLM's response text
        
    Returns:
        A dictionary containing the extracted JSON
    """
    try:
        # Try to find JSON in the response
        # First, try to parse the entire response as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If that fails, try to extract anything that looks like JSON
        json_match = re.search(r'({[\s\S]*})', result)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all parsing attempts fail, extract structured data manually
        logger.warning("Could not parse JSON from LLM response, extracting manually")
        
        # Create a basic structure matching the expected format
        summary = {
            "vendor_name": None,
            "invoice_number": None,
            "invoice_date": None,
            "due_date": None,
            "billing_address": None,
            "shipping_address": None,
            "line_items": [],
            "subtotal": None,
            "taxes": None,
            "total_amount": None,
            "currency": None
        }
        
        # Try to extract each field
        for line in result.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if value.lower() in ["null", "none", "n/a", ""]:
                    value = None
                
                if key in summary:
                    summary[key] = value
        
        return summary
        
    except Exception as e:
        logger.exception(f"Error extracting JSON from result: {str(e)}")
        return generate_mock_summary()