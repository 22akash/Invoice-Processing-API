"""
This module handles the analysis of invoices based on user questions.
"""

import logging
import time
import json
import yaml
from typing import Dict, Any, List, Tuple
from src.llm.llm import OpenAILLM
from src.retrievers.retrievers import retrieve_invoice_data
from src.retrievers.relevance_checker import check_relevance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI LLM
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
            "analyze_invoice": {
                "system_prompt": "You are an expert at analyzing invoices and answering questions about them.",
                "user_prompt": "Here is the invoice data: {invoice_data}\n\nQuestion: {question}\n\nPlease provide a detailed answer based on the invoice data."
            }
        }

# Get prompts
prompts = load_prompts()

def analyze_invoice(
    image_path: str,
    question: str,
    request_id: str,
    username: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Analyze an invoice based on a user question.
    
    Args:
        image_path: Path to the invoice image
        question: Question about the invoice
        request_id: Unique identifier for the request
        username: Name of the user making the request
        
    Returns:
        Tuple containing:
        - Dictionary with analysis results
        - List of token usage details
    """
    start_time = time.time()
    token_usage_details_list = []
    
    # Log the original request details
    logger.info(f"Analyzing invoice {image_path} with question: {question}")
    
    # Retrieve relevant invoice data
    retrieval_results, token_usage_details = retrieve_invoice_data(
        image_path=image_path,
        question=question,
        request_id=request_id
    )
    token_usage_details_list.extend(token_usage_details)
    
    # If no relevant data found, return early
    if not retrieval_results.get("matching_documents_found"):
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "image_path": image_path,
            "request_id": request_id,
            "question": question,
            "elapsed_time": elapsed_time,
            "retrieval_results": retrieval_results,
            "responses": [],
            "most_relevant_answer": json.dumps({
                "answer": "I couldn't find any relevant information to answer your question.",
                "source": "none",
                "confidence": 0.0
            })
        }, token_usage_details_list
    
    # Process the retrieved data and generate responses
    responses = []
    document_data = retrieval_results.get("document_data", [])
    
    for doc in document_data:
        invoice_data = doc.get("matching_records", [{}])[0].get("content", "{}")
        doc_image = doc.get("image", "unknown")
        
        # Get prompts for invoice analysis
        analyze_invoice_prompts = prompts.get("analyze_invoice", {})
        system_prompt = analyze_invoice_prompts.get("system_prompt", "")
        user_prompt_template = analyze_invoice_prompts.get("user_prompt", "")
        
        # Format the user prompt with invoice data and question
        user_prompt = user_prompt_template.format(
            invoice_data=invoice_data,
            question=question
        )
        
        # Call the LLM to analyze the invoice
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
                image_path, 
                token_count, 
                "invoice_analysis"
            )
            
            token_usage_details_list.append({
                "request_id": request_id,
                "username": username,
                "token_count": token_count,
                "model": llm.model,
                "request_reason": "invoice_analysis"
            })
            
            # Add the response to the list
            responses.append({
                "source": f"invoice_{doc_image}",
                "response": result
            })
            
        except Exception as e:
            logger.exception(f"Error analyzing invoice: {str(e)}")
    
    # If no responses were generated, return early
    if not responses:
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "image_path": image_path,
            "request_id": request_id,
            "question": question,
            "elapsed_time": elapsed_time,
            "retrieval_results": retrieval_results,
            "responses": [],
            "most_relevant_answer": json.dumps({
                "answer": "I couldn't generate any responses to your question.",
                "source": "none",
                "confidence": 0.0
            })
        }, token_usage_details_list
    
    # Check relevance of responses and get the most relevant one
    # Pass the original image path to prioritize responses from that image
    most_relevant_answer, token_usage = check_relevance(
        request_id=request_id,
        username=username,
        question=question,
        responses=responses,
        original_image_path=image_path
    )
    token_usage_details_list.append(token_usage)
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Return the final result
    return {
        "image_path": image_path,
        "request_id": request_id,
        "question": question,
        "elapsed_time": elapsed_time,
        "retrieval_results": retrieval_results,
        "responses": responses,
        "most_relevant_answer": most_relevant_answer
    }, token_usage_details_list