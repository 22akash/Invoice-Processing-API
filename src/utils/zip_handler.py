"""
This module handles zip file operations for the invoice clustering API.
"""

import os
import logging
import zipfile
import tempfile
import shutil
import uuid
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_zip(zip_file_path: str, extract_dir: str = None, extract_files: bool = True) -> Tuple[str, List[str]]:
    """
    Extract a zip file to a directory or just list the image files without extracting.
    
    Args:
        zip_file_path: Path to the zip file
        extract_dir: Directory to extract to (if None, a temporary directory will be created)
        extract_files: Whether to actually extract the files or just list them
        
    Returns:
        Tuple containing:
        - Path to the extraction directory
        - List of paths to extracted image files (or paths they would have if extracted)
    """
    # Create extraction directory if not provided
    if extract_dir is None:
        unique_id = str(uuid.uuid4())[:8]
        extract_dir = os.path.join(tempfile.gettempdir(), f"invoice_extract_{unique_id}")
    
    if extract_files:
        os.makedirs(extract_dir, exist_ok=True)
        logger.info(f"Extracting zip file {zip_file_path} to {extract_dir}")
    else:
        logger.info(f"Listing image files in zip file {zip_file_path} without extracting")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']
    extracted_images = []
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Get list of files in the zip
            file_list = zip_ref.namelist()
            
            # Extract all files if requested
            if extract_files:
                zip_ref.extractall(extract_dir)
            
            # Find all image files
            for file_path in file_list:
                # Skip directories
                if file_path.endswith('/'):
                    continue
                
                # Check if it's an image file
                _, ext = os.path.splitext(file_path.lower())
                if ext in image_extensions:
                    extracted_path = os.path.join(extract_dir, file_path)
                    
                    if extract_files:
                        # Ensure the path exists (in case of nested directories)
                        os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                    
                    extracted_images.append(extracted_path)
        
        if extract_files:
            logger.info(f"Extracted {len(extracted_images)} image files from zip")
        else:
            logger.info(f"Found {len(extracted_images)} image files in zip")
        
        return extract_dir, extracted_images
    
    except Exception as e:
        logger.exception(f"Error processing zip file: {str(e)}")
        return extract_dir, []

def create_zip(directory_path: str, output_zip_path: str = None) -> str:
    """
    Create a zip file from a directory.
    
    Args:
        directory_path: Path to the directory to zip
        output_zip_path: Path to the output zip file (if None, a name will be generated)
        
    Returns:
        Path to the created zip file
    """
    # Create output zip path if not provided
    if output_zip_path is None:
        dir_name = os.path.basename(directory_path)
        output_zip_path = os.path.join(os.path.dirname(directory_path), f"{dir_name}.zip")
    
    logger.info(f"Creating zip file {output_zip_path} from directory {directory_path}")
    
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate the relative path for the file in the zip
                    rel_path = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, rel_path)
        
        logger.info(f"Created zip file {output_zip_path}")
        return output_zip_path
    
    except Exception as e:
        logger.exception(f"Error creating zip file: {str(e)}")
        return None

def save_uploaded_zip(uploaded_file, upload_dir: str = "uploads") -> str:
    """
    Save an uploaded zip file to disk.
    
    Args:
        uploaded_file: The uploaded file object from FastAPI
        upload_dir: Directory to save the file to
        
    Returns:
        Path to the saved zip file
    """
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate a unique filename
    unique_id = str(uuid.uuid4())[:8]
    filename = f"invoice_upload_{unique_id}.zip"
    file_path = os.path.join(upload_dir, filename)
    
    logger.info(f"Saving uploaded zip file to {file_path}")
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())
        
        logger.info(f"Saved uploaded zip file to {file_path}")
        return file_path
    
    except Exception as e:
        logger.exception(f"Error saving uploaded zip file: {str(e)}")
        return None

def cleanup_temp_files(file_paths: List[str], directories: List[str] = None):
    """
    Clean up temporary files and directories.
    
    Args:
        file_paths: List of file paths to delete
        directories: List of directories to delete
    """
    # Delete files
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted temporary file {file_path}")
        except Exception as e:
            logger.warning(f"Error deleting temporary file {file_path}: {str(e)}")
    
    # Delete directories
    if directories:
        for directory in directories:
            try:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                    logger.info(f"Deleted temporary directory {directory}")
            except Exception as e:
                logger.warning(f"Error deleting temporary directory {directory}: {str(e)}")

def maintain_folder_structure(source_dir: str, target_dir: str, cluster_mapping: Dict[str, int]):
    """
    Maintain the original folder structure when organizing files into clusters.
    
    Args:
        source_dir: Source directory containing the extracted files
        target_dir: Target directory for the clustered files
        cluster_mapping: Dictionary mapping file paths to cluster IDs
    """
    logger.info(f"Maintaining folder structure from {source_dir} to {target_dir}")
    
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Process each file in the cluster mapping
        for file_path, cluster_id in cluster_mapping.items():
            # Calculate the relative path from the source directory
            rel_path = os.path.relpath(file_path, source_dir)
            
            # Create the target path with cluster subdirectory
            cluster_dir = os.path.join(target_dir, f"cluster_{cluster_id}")
            target_path = os.path.join(cluster_dir, rel_path)
            
            # Create the directory structure
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(file_path, target_path)
            logger.debug(f"Copied {file_path} to {target_path}")
        
        logger.info(f"Maintained folder structure for {len(cluster_mapping)} files")
    
    except Exception as e:
        logger.exception(f"Error maintaining folder structure: {str(e)}")