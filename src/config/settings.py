"""
Configuration utilities for the Invoice Processing API.
"""

import logging
import yaml
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "credentials/secrets.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

# Load configuration
config = load_config()
openai_config = config.get('openai', {})
MODEL = openai_config.get('model', 'gpt-4o')

# Create necessary directories
def create_required_directories():
    """Create directories required by the application."""
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("static", exist_ok=True)