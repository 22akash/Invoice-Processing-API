from abc import ABC, abstractmethod
import os
import yaml
import base64
import time
import ssl
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from openai import OpenAI
import logging
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define request model
class ImagePathRequest(BaseModel):
    file_path: str

# Disable SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

class LLM(ABC):
    """Abstract base class for Language Learning Models."""
    
    @abstractmethod
    def get_embeddings(self, images: List[str]) -> Tuple[List[List[float]], List[int]]:
        """
        Generate embeddings for a list of image paths.
        
        Args:
            images: List of image file paths
            
        Returns:
            Tuple containing:
            - List of embedding vectors
            - List of token counts for each embedding
        """
        pass
    
    @abstractmethod
    def process_image(self, image_path: str) -> Tuple[Dict[str, Any], int]:
        """
        Process a single image and extract information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
            - Dictionary containing extracted information
            - Token count used for processing
        """
        pass


class OpenAILLM(LLM):
    """Implementation of LLM using OpenAI's models."""
    
    def __init__(self, config_path: str = "credentials/secrets.yaml"):
        """
        Initialize the OpenAI LLM with configuration from secrets.yaml.
        
        Args:
            config_path: Path to the configuration file
        """
        logger.info(f"Initializing OpenAI LLM with config from: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            openai_config = config.get('openai', {})
            self.api_key = openai_config.get('api_key', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFrYXNoLmd1bmR1QGdyYW1lbmVyLmNvbSJ9.gY4pT7cZoafaS2EC3lgLokeRUyvdD99yW6O-Ggydi-A')
            self.base_url = openai_config.get('base_url', 'https://llmfoundry.straive.com/openai/v1/')
            self.model = openai_config.get('model', 'gpt-4o')  # Updated default model to match test.py
            
            if not self.api_key:
                logger.error("API key is missing in configuration")
                raise ValueError("API key is required")
                
            if self.api_key == "your-openai-api-key-here":
                logger.warning("API key not properly configured in secrets.yaml")
            
            # Create a custom httpx client with SSL verification disabled
            logger.warning("Creating OpenAI client with SSL verification disabled - NOT RECOMMENDED FOR PRODUCTION")
            http_client = httpx.Client(verify=False)
            
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                http_client=http_client
            )
            
            # Import here to avoid circular imports
            from src.db.postgres import SYSTEM_USERNAME
            self.username = SYSTEM_USERNAME
            
            logger.info(f"OpenAI LLM initialized with model: {self.model}")
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Error initializing OpenAI LLM: {str(e)}")
            raise
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.exception(f"Error encoding image {image_path}: {str(e)}")
            raise
    
    def _track_token_usage(self, image_path: str, token_count: int, use_case: str):
        """
        Track token usage in the database.
        
        Args:
            image_path: Path to the image file
            token_count: Number of tokens used
            use_case: The use case for the token usage
        """
        try:
            # Import here to avoid circular imports
            from src.db.postgres import track_api_call
            
            # Track token usage
            track_api_call(
                username=self.username,
                image=image_path,
                model=self.model,
                token_count=token_count,
                use_case=use_case
            )
            logger.info(f"Tracked token usage: {token_count} tokens for {use_case}")
        except Exception as e:
            logger.exception(f"Error tracking token usage: {str(e)}")
    
    def _extract_text_from_image(self, image_path: str) -> Tuple[str, int]:
        """
        Extract text content from an image using vision capabilities.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
            - Extracted text content
            - Token count used for extraction
        """
        logger.info(f"Extracting text content from image: {image_path}")
        
        try:
            # Encode the image to base64
            base64_image = self._encode_image(image_path)
            
            # Load prompts from YAML
            try:
                with open("src/prompts/prompts.yaml", 'r') as file:
                    prompts = yaml.safe_load(file)
                    
                embedding_prompts = prompts.get("embedding", {})
                system_prompt = embedding_prompts.get("system_prompt", "You are an expert at analyzing invoice images and extracting detailed information.")
                user_prompt = embedding_prompts.get("user_prompt", "Extract all key information from this invoice including vendor name, invoice number, date, line items, amounts, and any other unique identifying details. Be comprehensive and specific.")
            except Exception as e:
                logger.warning(f"Error loading prompts: {str(e)}, using default prompts")
                system_prompt = "You are an expert at analyzing invoice images and extracting detailed information."
                user_prompt = "Extract all key information from this invoice including vendor name, invoice number, date, line items, amounts, and any other unique identifying details. Be comprehensive and specific."
            
            # Call the vision model to extract text
            response = self.client.chat.completions.create(
                model=self.model,  # Using the vision-capable model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1000
            )
            
            # Extract the response content
            extracted_text = response.choices[0].message.content
            
            # Extract token usage
            token_count = 0
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                token_count = response.usage.total_tokens
            elif hasattr(response, 'usage') and isinstance(response.usage, dict) and 'total_tokens' in response.usage:
                token_count = response.usage['total_tokens']
            
            # Track token usage
            self._track_token_usage(image_path, token_count, "image_text_extraction")
            
            logger.info(f"Successfully extracted text from image, used {token_count} tokens")
            return extracted_text, token_count
            
        except Exception as e:
            logger.exception(f"Error extracting text from image: {str(e)}")
            return f"Invoice image: {os.path.basename(image_path)}", 0
    
    def get_embeddings(self, images: List[str]) -> Tuple[List[List[float]], List[int]]:
        """
        Generate embeddings for a list of image paths using OpenAI's API.
        First extracts text content from each image, then generates embeddings for that text.
        
        Args:
            images: List of image file paths
            
        Returns:
            Tuple containing:
            - List of embedding vectors
            - List of token counts for each embedding
        """
        logger.info(f"Generating embeddings for {len(images)} images")
        embeddings = []
        token_counts = []
        
        for i, image_path in enumerate(images):
            try:
                # Check if the image exists
                if not os.path.exists(image_path):
                    logger.error(f"Image file not found: {image_path}")
                    embeddings.append([])
                    token_counts.append(0)
                    continue
                
                # First, extract text content from the image
                extracted_text, extraction_token_count = self._extract_text_from_image(image_path)
                
                # Now, generate embeddings for the extracted text
                logger.info(f"Generating embedding for extracted text from image {i+1}/{len(images)}")
                
                try:
                    # Generate embedding for the extracted text
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=extracted_text
                    )
                    
                    # Extract the embedding vector
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                    
                    # Extract token usage
                    embedding_token_count = 0
                    if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                        embedding_token_count = response.usage.total_tokens
                    elif hasattr(response, 'usage') and isinstance(response.usage, dict) and 'total_tokens' in response.usage:
                        embedding_token_count = response.usage['total_tokens']
                    
                    # Total token count is the sum of extraction and embedding token counts
                    total_token_count = extraction_token_count + embedding_token_count
                    token_counts.append(total_token_count)
                    
                    logger.info(f"Used {total_token_count} tokens for embedding {image_path}")
                    
                    # Track token usage in the database
                    self._track_token_usage(image_path, embedding_token_count, "llm_embedding_generation")
                    
                except Exception as api_error:
                    logger.error(f"API error for {image_path}: {str(api_error)}")
                    logger.warning("Using mock embedding as fallback")
                    
                    # Create a mock embedding (all zeros) as a fallback
                    mock_embedding = [0.0] * 1536  # Standard size for OpenAI embeddings
                    embeddings.append(mock_embedding)
                    token_counts.append(extraction_token_count)  # Only count the extraction tokens
                
            except Exception as e:
                logger.exception(f"Error generating embedding for {image_path}: {str(e)}")
                # Add empty vector to maintain order
                embeddings.append([])
                token_counts.append(0)
                
        logger.info(f"Completed generating {len(embeddings)} embeddings")
        return embeddings, token_counts
    
    def process_image(self, image_path: str) -> Tuple[Dict[str, Any], int]:
        """
        Process an invoice image and extract information using OpenAI's vision capabilities.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing:
            - Dictionary containing extracted information
            - Token count used for processing
        """
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Check if the image exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return {"error": f"File not found: {image_path}"}, 0
            
            # Encode the image to base64
            base64_image = self._encode_image(image_path)
            
            # Call OpenAI API with the image
            logger.info(f"Calling OpenAI API with model {self.model}")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting information from invoice images. Extract all relevant details such as invoice number, date, vendor, line items, amounts, and totals."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all information from this invoice image."},
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
                
                # Track token usage in the database
                self._track_token_usage(image_path, token_count, "llm_image_processing")
                
                logger.info(f"Image processing completed successfully, used {token_count} tokens")
                
                return {"extracted_text": result}, token_count
            except Exception as api_error:
                logger.error(f"API error: {str(api_error)}")
                return {"error": f"API error: {str(api_error)}"}, 0
            
        except Exception as e:
            logger.exception(f"Error processing image {image_path}: {str(e)}")
            return {"error": str(e)}, 0


# Example usage
if __name__ == "__main__":
    logger.info("Running LLM module as main")
    try:
        llm = OpenAILLM()
        
        # Example to process a single image
        logger.info("Testing image processing")
        result, token_count = llm.process_image("data/1.jpg")
        logger.info(f"Processing result: {result}, token count: {token_count}")
        
        # Example to get embeddings for multiple images
        logger.info("Testing embedding generation")
        image_paths = ["data/1.jpg", "data/2.jpg"]
        embeddings, token_counts = llm.get_embeddings(image_paths)
        logger.info(f"Generated {len(embeddings)} embeddings, token counts: {token_counts}")
    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")