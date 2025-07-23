"""
This module handles the relevance checking of responses from different sources.
"""

import logging
import time
import json
from typing import Dict, Any, List, Tuple
from src.llm.llm import OpenAILLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI LLM
llm = OpenAILLM()

def check_relevance(
    request_id: str,
    username: str,
    question: str,
    responses: List[Dict[str, Any]],
    original_image_path: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Check the relevance of multiple responses and select the most relevant one.
    
    Args:
        request_id: Unique identifier for the request
        username: Name of the user making the request
        question: Original question asked by the user
        responses: List of responses from different sources
        original_image_path: Path to the original image being queried (optional)
        
    Returns:
        Tuple containing:
        - Most relevant response as a JSON string
        - Token usage details
    """
    if not responses:
        logger.warning("No responses to check for relevance")
        return json.dumps({
            "answer": "I couldn't find any relevant information to answer your question.",
            "source": "none",
            "confidence": 0.0
        }), {"request_id": request_id, "username": username, "token_count": 0, "model": llm.model, "request_reason": "relevance_check"}
    
    if len(responses) == 1:
        logger.info("Only one response available, returning it directly")
        return json.dumps({
            "answer": responses[0]["response"],
            "source": responses[0]["source"],
            "confidence": 1.0
        }), {"request_id": request_id, "username": username, "token_count": 0, "model": llm.model, "request_reason": "relevance_check"}
    
    # If we have the original image path, prioritize responses from that image
    if original_image_path:
        for response in responses:
            # Check if this response is from the original image
            if original_image_path in response["source"]:
                logger.info(f"Prioritizing response from original image: {original_image_path}")
                return json.dumps({
                    "answer": response["response"],
                    "source": response["source"],
                    "confidence": 0.95  # High confidence since it's from the original image
                }), {"request_id": request_id, "username": username, "token_count": 0, "model": llm.model, "request_reason": "relevance_check"}
    
    # Prepare the prompt for relevance checking
    system_prompt = """
    You are an expert at evaluating responses to questions about invoices.
    You will be given a question and multiple responses.
    Your task is to determine which response best answers the question.
    
    IMPORTANT: Your answer must be DIRECTLY RELATED to the question asked. 
    Do not include any information that is not directly answering the question.
    Be precise and concise in your answer.
    
    Return your answer in the following JSON format:
    {
        "answer": "The most relevant and accurate response that directly answers the question",
        "source": "The source of the most relevant response",
        "confidence": "A number between 0 and 1 indicating your confidence in this selection"
    }
    """
    
    user_prompt = f"""
    Question: {question}
    
    Responses:
    """
    
    for i, response in enumerate(responses):
        user_prompt += f"\nResponse {i+1} (Source: {response['source']}):\n{response['response']}\n"
    
    user_prompt += "\nWhich response best answers the question? Return your answer in the specified JSON format."
    
    # Call the LLM to determine the most relevant response
    try:
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
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
        
        # Track token usage
        llm._track_token_usage(
            "relevance_check", 
            token_count, 
            "relevance_check"
        )
        
        # Parse the result to ensure it's valid JSON
        try:
            # First, check if the result contains a JSON block
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
            if json_match:
                try:
                    json_result = json.loads(json_match.group(1))
                    logger.info("Successfully parsed JSON from code block")
                except json.JSONDecodeError:
                    # If parsing the code block fails, try the whole response
                    json_result = json.loads(result)
            else:
                # Try to parse the whole response as JSON
                json_result = json.loads(result)
            
            # Ensure the required fields are present
            if not all(key in json_result for key in ["answer", "source", "confidence"]):
                logger.warning("Response missing required fields, using default format")
                json_result = {
                    "answer": result,
                    "source": "combined",
                    "confidence": 0.5
                }
            
            # Clean up the answer to remove any JSON formatting or code blocks
            if isinstance(json_result["answer"], str):
                # Remove any markdown code block formatting
                json_result["answer"] = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', json_result["answer"])
                # Remove any JSON formatting within the answer
                json_result["answer"] = re.sub(r'^\s*{\s*"answer"\s*:\s*"(.+?)"\s*,.*$', r'\1', json_result["answer"])
            
            return json.dumps(json_result), {"request_id": request_id, "username": username, "token_count": token_count, "model": llm.model, "request_reason": "relevance_check"}
        except json.JSONDecodeError:
            logger.warning("Could not parse response as JSON, using raw response")
            return json.dumps({
                "answer": result,
                "source": "combined",
                "confidence": 0.5
            }), {"request_id": request_id, "username": username, "token_count": token_count, "model": llm.model, "request_reason": "relevance_check"}
    
    except Exception as e:
        logger.exception(f"Error checking relevance: {str(e)}")
        return json.dumps({
            "answer": "I encountered an error while determining the most relevant response.",
            "source": "error",
            "confidence": 0.0
        }), {"request_id": request_id, "username": username, "token_count": 0, "model": llm.model, "request_reason": "relevance_check"}