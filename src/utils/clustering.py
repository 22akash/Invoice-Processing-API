"""
This module handles clustering of invoice images based on their embeddings.
"""

import os
import logging
import numpy as np
import shutil
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import uuid
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def determine_optimal_clusters(embeddings: List[List[float]], min_clusters: int = 2, max_clusters: int = 10) -> int:
    """
    Determine the optimal number of clusters using silhouette score.
    
    Args:
        embeddings: List of embedding vectors
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    if len(embeddings) < min_clusters:
        logger.warning(f"Not enough samples ({len(embeddings)}) for clustering. Using minimum clusters: {min_clusters}")
        return min(len(embeddings), min_clusters)
    
    # Convert embeddings to numpy array
    X = np.array(embeddings)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of clusters and compute silhouette score
    silhouette_scores = []
    max_clusters_to_try = min(max_clusters, len(embeddings) - 1)
    
    for n_clusters in range(min_clusters, max_clusters_to_try + 1):
        # Skip if we have too few samples for the number of clusters
        if len(embeddings) <= n_clusters:
            silhouette_scores.append(-1)  # Invalid score
            continue
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            logger.info(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")
        except Exception as e:
            logger.warning(f"Error calculating silhouette score for n_clusters={n_clusters}: {str(e)}")
            silhouette_scores.append(-1)  # Invalid score
    
    # Find the best number of clusters
    if all(score == -1 for score in silhouette_scores):
        logger.warning("Could not determine optimal clusters. Using default: 2")
        return min(2, len(embeddings))
    
    best_n_clusters = min_clusters + silhouette_scores.index(max([s for s in silhouette_scores if s != -1]))
    logger.info(f"Optimal number of clusters: {best_n_clusters}")
    
    return best_n_clusters

def cluster_embeddings(embeddings: List[List[float]], image_paths: List[str], 
                       n_clusters: int = None, method: str = 'kmeans') -> Dict[int, List[str]]:
    """
    Cluster embeddings and return image paths grouped by cluster.
    
    Args:
        embeddings: List of embedding vectors
        image_paths: List of image file paths corresponding to the embeddings
        n_clusters: Number of clusters (if None, will be determined automatically)
        method: Clustering method ('kmeans' or 'dbscan')
        
    Returns:
        Dictionary mapping cluster IDs to lists of image paths
    """
    if len(embeddings) == 0:
        logger.warning("No embeddings provided for clustering")
        return {}
    
    if len(embeddings) != len(image_paths):
        logger.error(f"Number of embeddings ({len(embeddings)}) does not match number of image paths ({len(image_paths)})")
        return {}
    
    # Convert embeddings to numpy array
    X = np.array(embeddings)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters if not provided
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(embeddings)
    
    # Perform clustering
    if method == 'kmeans':
        logger.info(f"Performing KMeans clustering with {n_clusters} clusters")
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(X_scaled)
    elif method == 'dbscan':
        logger.info("Performing DBSCAN clustering")
        clustering = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clustering.fit_predict(X_scaled)
    else:
        logger.error(f"Unknown clustering method: {method}")
        return {}
    
    # Group image paths by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(image_paths[i])
    
    logger.info(f"Created {len(clusters)} clusters")
    for cluster_id, paths in clusters.items():
        logger.info(f"Cluster {cluster_id}: {len(paths)} images")
    
    return clusters

def visualize_clusters(embeddings: List[List[float]], cluster_labels: List[int], output_path: str):
    """
    Visualize clusters using PCA for dimensionality reduction.
    
    Args:
        embeddings: List of embedding vectors
        cluster_labels: List of cluster labels
        output_path: Path to save the visualization
    """
    try:
        # Convert embeddings to numpy array
        X = np.array(embeddings)
        
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Invoice Clusters (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Cluster visualization saved to {output_path}")
    except Exception as e:
        logger.exception(f"Error visualizing clusters: {str(e)}")

def organize_clusters(clusters: Dict[int, List[str]], output_dir: str) -> Dict[int, str]:
    """
    Organize images into cluster directories.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of image paths
        output_dir: Base directory to create cluster subdirectories
        
    Returns:
        Dictionary mapping cluster IDs to cluster directory paths
    """
    cluster_dirs = {}
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a directory for each cluster and copy images
        for cluster_id, image_paths in clusters.items():
            cluster_name = f"cluster_{cluster_id}"
            cluster_dir = os.path.join(output_dir, cluster_name)
            os.makedirs(cluster_dir, exist_ok=True)
            
            for image_path in image_paths:
                # Copy the image to the cluster directory
                image_filename = os.path.basename(image_path)
                dest_path = os.path.join(cluster_dir, image_filename)
                shutil.copy2(image_path, dest_path)
            
            cluster_dirs[cluster_id] = cluster_dir
            logger.info(f"Created cluster directory {cluster_dir} with {len(image_paths)} images")
        
        return cluster_dirs
    
    except Exception as e:
        logger.exception(f"Error organizing clusters: {str(e)}")
        return {}

def process_and_cluster_images(image_paths: List[str], output_dir: str = None, n_clusters: int = None, 
                               method: str = 'kmeans', visualize: bool = True) -> Dict[str, Any]:
    """
    Process images, generate embeddings, cluster them, and organize into directories.
    
    Args:
        image_paths: List of image file paths
        output_dir: Base directory to create cluster subdirectories (if None, a temporary directory will be created)
        n_clusters: Number of clusters (if None, will be determined automatically)
        method: Clustering method ('kmeans' or 'dbscan')
        visualize: Whether to create a visualization of the clusters
        
    Returns:
        Dictionary with clustering results
    """
    from src.utils.embeddings import generate_embeddings
    
    start_time = time.time()
    
    # Create a unique output directory if not provided
    if output_dir is None:
        unique_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join("output", f"clusters_{unique_id}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate embeddings for all images
        logger.info(f"Generating embeddings for {len(image_paths)} images")
        embeddings, token_counts = generate_embeddings(image_paths)
        
        # Filter out any failed embeddings
        valid_embeddings = []
        valid_image_paths = []
        for i, embedding in enumerate(embeddings):
            if embedding and len(embedding) > 0:
                valid_embeddings.append(embedding)
                valid_image_paths.append(image_paths[i])
        
        logger.info(f"Generated {len(valid_embeddings)} valid embeddings out of {len(image_paths)} images")
        
        # Cluster the embeddings
        clusters = cluster_embeddings(valid_embeddings, valid_image_paths, n_clusters, method)
        
        # Visualize the clusters if requested
        if visualize and len(valid_embeddings) > 1:
            # Convert clusters dictionary to labels list for visualization
            labels = [-1] * len(valid_embeddings)  # Default to -1 (no cluster)
            for cluster_id, paths in clusters.items():
                for path in paths:
                    idx = valid_image_paths.index(path)
                    labels[idx] = cluster_id
            
            visualization_path = os.path.join(output_dir, "cluster_visualization.png")
            visualize_clusters(valid_embeddings, labels, visualization_path)
        
        # Organize images into cluster directories
        cluster_dirs = organize_clusters(clusters, output_dir)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        return {
            "output_directory": output_dir,
            "cluster_directories": cluster_dirs,
            "num_clusters": len(clusters),
            "num_images_processed": len(valid_embeddings),
            "processing_time": processing_time,
            "token_usage": sum(token_counts)
        }
    
    except Exception as e:
        logger.exception(f"Error in process_and_cluster_images: {str(e)}")
        return {
            "error": str(e),
            "output_directory": output_dir,
            "processing_time": time.time() - start_time
        }