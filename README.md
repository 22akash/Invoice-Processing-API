# Invoice Processing API

This API provides endpoints for processing invoice images, including embedding generation, data extraction, question answering, and clustering.

## Table of Contents

- [Invoice Processing API](#invoice-processing-api)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Architecture Overview](#architecture-overview)
  - [Invoice Clustering](#invoice-clustering)
    - [Clustering Process](#clustering-process)
    - [Clustering Methods](#clustering-methods)
    - [Visualization](#visualization)
    - [Technical Implementation](#technical-implementation)
  - [API Endpoints](#api-endpoints)
    - [Generate Embeddings](#generate-embeddings)
    - [Get Summary](#get-summary)
    - [Invoice RAG Response](#invoice-rag-response)
    - [Cluster Invoices](#cluster-invoices)
    - [Check Clustering Status](#check-clustering-status)
    - [Download Clustered Images](#download-clustered-images)
  - [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Web Interface](#web-interface)
  - [Folder Structure](#folder-structure)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Generate Embeddings**: Create vector embeddings for invoice images
- **Extract Invoice Data**: Extract structured data from invoice images
- **Invoice RAG**: Ask questions about invoices and get answers
- **Invoice Clustering**: Group similar invoices together based on their content

## Architecture Overview

The system uses a Retrieval-Augmented Generation (RAG) approach to process and analyze invoice images:

1. **Image Processing**: Invoice images are processed to extract text and visual information
2. **Embedding Generation**: OpenAI's API is used to create vector embeddings that represent the content
3. **Data Extraction**: Structured data is extracted from invoices using LLM-based analysis
4. **Question Answering**: RAG system combines retrieved information with LLM capabilities
5. **Clustering**: Similar invoices are grouped together based on their embeddings

## Invoice Clustering

The clustering functionality allows you to organize invoice images into groups based on their content similarity. This is particularly useful for:

- **Vendor Identification**: Automatically group invoices from the same vendor
- **Template Recognition**: Identify invoices that follow the same template
- **Anomaly Detection**: Find invoices that don't fit into any established pattern
- **Batch Processing**: Process similar invoices together for efficiency

### Clustering Process

1. **Embedding Generation**: Each invoice image is processed to generate a vector embedding that represents its content
2. **Optimal Cluster Determination**: The system automatically determines the optimal number of clusters using silhouette analysis
3. **Clustering Algorithm**: Invoices are grouped using either K-means or DBSCAN algorithms
4. **Visualization**: Optional 2D visualization of clusters using Principal Component Analysis (PCA)
5. **Output Organization**: Clustered invoices are organized into folders for easy access

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Invoice Images │────▶│  Extract Text   │────▶│   Generate      │
│  (ZIP File)     │     │  & Features     │     │   Embeddings    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Download       │◀────│  Create Cluster │◀────│  Apply          │
│  Results        │     │  ZIP File       │     │  Clustering     │
│                 │     │                 │     │  Algorithm      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Clustering Methods

- **K-means**: Partitions the invoices into K clusters where each invoice belongs to the cluster with the nearest mean
- **DBSCAN**: Density-based clustering that groups together invoices that are closely packed together

### Visualization

The clustering API can generate a 2D visualization of the clusters using PCA to reduce the high-dimensional embeddings to 2 dimensions. This visualization helps to understand the distribution of invoices and the effectiveness of the clustering.

### Technical Implementation

The clustering functionality is implemented in the following files:

- `src/utils/clustering.py`: Contains the core clustering algorithms and visualization functions
- `src/utils/zip_handler.py`: Handles zip file operations for input and output
- `main.py`: Implements the API endpoints for clustering

The clustering process works as follows:

1. **Job Creation**: When a clustering request is received, a unique job ID is generated and the job is queued for processing
2. **Background Processing**: The clustering is performed in a background task to avoid blocking the API
3. **Embedding Generation**: Each invoice image is processed to extract text and generate embeddings using OpenAI's API
4. **Optimal Cluster Determination**:
   - For K-means, the system tries different numbers of clusters and selects the one with the best silhouette score
   - For DBSCAN, the system automatically determines the appropriate epsilon value based on the data distribution
5. **Clustering**: The embeddings are clustered using the selected algorithm
6. **Visualization**: If requested, a 2D visualization is created using PCA and saved as an image
7. **Output Organization**: The clustered invoices are organized into folders based on their cluster assignments
8. **ZIP Creation**: A zip file is created with the clustered invoices and visualization (if generated)

The system maintains the original folder structure from the input zip file, ensuring that related files stay together within their assigned clusters.

## API Endpoints

### Generate Embeddings

```
POST /generate_embeddings
```

Generate embeddings for an invoice image.

**Request Body**:
```json
{
  "file_path": "path/to/invoice.jpg"
}
```

**Response**:
```json
{
  "embedding": [0.123, 0.456, ...],
  "embedding_id": "emb_123456",
  "file_path": "path/to/invoice.jpg"
}
```

### Get Summary

```
POST /get_summary
```

Extract structured data from an invoice image.

**Request Body**:
```json
{
  "file_path": "path/to/invoice.jpg"
}
```

**Response**:
```json
{
  "vendor": "ABC Company",
  "invoice_number": "INV-12345",
  "date": "2023-01-15",
  "total_amount": 1250.00,
  "line_items": [
    {
      "description": "Product A",
      "quantity": 5,
      "unit_price": 100.00,
      "amount": 500.00
    },
    ...
  ]
}
```

### Invoice RAG Response

```
POST /invoice_rag_response
```

Ask a question about an invoice and get an answer.

**Request Body**:
```json
{
  "file_path": "path/to/invoice.jpg",
  "question": "What is the total amount of this invoice?"
}
```

**Response**:
```json
{
  "answer": "The total amount of this invoice is $1,250.00.",
  "confidence": 0.95,
  "sources": [
    {
      "text": "Total: $1,250.00",
      "relevance": 0.98
    }
  ]
}
```

### Cluster Invoices

```
POST /cluster_invoice
```

Cluster invoice images based on their content.

**Request Body**:
- Form data with:
  - `zip_file`: A zip file containing invoice images
  - `n_clusters` (optional): Number of clusters (if not provided, will be determined automatically)
  - `method` (optional): Clustering method ('kmeans' or 'dbscan')
  - `visualize` (optional): Whether to create a visualization of the clusters (true/false)

**Response**:
```json
{
  "status": "queued",
  "message": "Clustering job queued for processing",
  "job_id": "1623456789"
}
```

### Check Clustering Status

```
GET /cluster_status/{job_id}
```

Check the status of a clustering job.

**Response**:
```json
{
  "status": "completed",
  "message": "Clustering completed successfully",
  "job_id": "1623456789",
  "output_zip_url": "/static/clusters_1623456789.zip",
  "num_clusters": 3,
  "num_images_processed": 10,
  "processing_time": 15.5,
  "token_usage": 5000
}
```

### Download Clustered Images

```
GET /download_clusters/{job_id}
```

Download the clustered images as a zip file.

## Setup and Installation

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/invoice-processing-api.git
   cd invoice-processing-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database:
   ```bash
   # Create a PostgreSQL database
   createdb invoice_db
   
   # Apply the schema
   psql -d invoice_db -f src/db/schema.sql
   ```

4. Configure API keys:
   ```bash
   # Create credentials directory
   mkdir -p credentials
   
   # Create secrets.yaml file with your OpenAI API key
   echo "openai_api_key: your_api_key_here" > credentials/secrets.yaml
   ```

5. Run the API:
   ```bash
   python main.py
   ```

## Web Interface

A simple web interface for clustering invoices is available at:

```
http://localhost:8080/static/cluster_form.html
```

This interface allows you to:
- Upload a zip file containing invoice images
- Specify the number of clusters (optional)
- Choose the clustering method (K-means or DBSCAN)
- Enable/disable visualization
- Submit the clustering job
- Download the clustered images

## Folder Structure

```
.
├── credentials/          # API keys and credentials
├── data/                 # Sample invoice images
├── output/               # Output files (clustered images)
├── src/                  # Source code
│   ├── db/               # Database operations
│   │   ├── postgres.py   # PostgreSQL connection and queries
│   │   └── schema.sql    # Database schema
│   ├── llm/              # LLM integration
│   │   └── llm.py        # OpenAI API integration
│   ├── prompts/          # Prompts for LLM
│   │   └── prompts.yaml  # YAML file with prompts
│   ├── retrievers/       # Retrieval utilities
│   │   ├── retrievers.py # Vector retrieval functions
│   │   └── relevance_checker.py # Check relevance of retrieved content
│   └── utils/            # Utility functions
│       ├── embeddings.py # Generate and store embeddings
│       ├── invoice_analyzer.py # Extract data from invoices
│       ├── summary.py    # Generate summaries
│       ├── clustering.py # Clustering algorithms
│       └── zip_handler.py # Handle zip file operations
├── static/               # Static files for web interface and stores zipped clusters
│   └── cluster_form.html # HTML form for clustering
├── uploads/              # Temporary storage for uploaded files
├── main.py               # Main API entry point
└── requirements.txt      # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<!-- ## Example Usage

### Python Client Example

Here's a simple Python script that demonstrates how to use the clustering API programmatically:

```python
"""
Example script demonstrating how to use the Invoice Clustering API programmatically.
"""

import requests
import time
import os
import zipfile
from pathlib import Path

# API base URL - change this to match your deployment
API_BASE_URL = "http://localhost:8080"

def cluster_invoices(zip_file_path, n_clusters=None, method="kmeans", visualize=True):
    """
    Submit a clustering job to the API.
    
    Args:
        zip_file_path (str): Path to the zip file containing invoice images
        n_clusters (int, optional): Number of clusters. If None, will be determined automatically.
        method (str, optional): Clustering method ('kmeans' or 'dbscan'). Defaults to 'kmeans'.
        visualize (bool, optional): Whether to create a visualization. Defaults to True.
        
    Returns:
        dict: Response from the API containing the job_id
    """
    # Prepare the form data
    files = {'zip_file': open(zip_file_path, 'rb')}
    data = {}
    
    if n_clusters is not None:
        data['n_clusters'] = str(n_clusters)
    
    data['method'] = method
    data['visualize'] = str(visualize).lower()
    
    # Submit the clustering job
    response = requests.post(
        f"{API_BASE_URL}/cluster_invoice",
        files=files,
        data=data
    )
    
    # Close the file
    files['zip_file'].close()
    
    # Return the response
    return response.json()

def check_clustering_status(job_id):
    """
    Check the status of a clustering job.
    
    Args:
        job_id (str): The job ID returned by the cluster_invoices function
        
    Returns:
        dict: Response from the API containing the job status
    """
    response = requests.get(f"{API_BASE_URL}/cluster_status/{job_id}")
    return response.json()

def download_clustered_invoices(job_id, output_path):
    """
    Download the clustered invoices.
    
    Args:
        job_id (str): The job ID returned by the cluster_invoices function
        output_path (str): Path where the zip file will be saved
        
    Returns:
        str: Path to the downloaded zip file
    """
    response = requests.get(f"{API_BASE_URL}/download_clusters/{job_id}", stream=True)
    
    # Save the zip file
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return output_path

def extract_and_explore_clusters(zip_path, extract_dir):
    """
    Extract the clustered invoices and print information about the clusters.
    
    Args:
        zip_path (str): Path to the downloaded zip file
        extract_dir (str): Directory where the zip file will be extracted
        
    Returns:
        dict: Information about the clusters
    """
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Get information about the clusters
    clusters = {}
    cluster_dirs = [d for d in os.listdir(extract_dir) if d.startswith('cluster_')]
    
    for cluster_dir in cluster_dirs:
        cluster_path = os.path.join(extract_dir, cluster_dir)
        if os.path.isdir(cluster_path):
            # Count the number of files in the cluster
            file_count = sum(1 for _ in Path(cluster_path).rglob('*') if _.is_file())
            clusters[cluster_dir] = {
                'file_count': file_count,
                'path': cluster_path
            }
    
    return clusters

def main():
    """
    Main function demonstrating the complete workflow.
    """
    # Path to the zip file containing invoice images
    zip_file_path = "data/invoices.zip"
    
    # Submit the clustering job
    print("Submitting clustering job...")
    response = cluster_invoices(zip_file_path)
    job_id = response['job_id']
    print(f"Job submitted with ID: {job_id}")
    
    # Check the status of the job until it's completed
    print("Checking job status...")
    status = check_clustering_status(job_id)
    
    while status['status'] not in ['completed', 'failed']:
        print(f"Job status: {status['status']} - {status['message']}")
        time.sleep(5)  # Wait for 5 seconds before checking again
        status = check_clustering_status(job_id)
    
    print(f"Final job status: {status['status']} - {status['message']}")
    
    if status['status'] == 'completed':
        # Download the clustered invoices
        output_path = "output/clustered_invoices.zip"
        print(f"Downloading clustered invoices to {output_path}...")
        download_clustered_invoices(job_id, output_path)
        
        # Extract and explore the clusters
        extract_dir = "output/clustered_invoices"
        print(f"Extracting and exploring clusters in {extract_dir}...")
        clusters = extract_and_explore_clusters(output_path, extract_dir)
        
        # Print information about the clusters
        print("\nCluster Information:")
        for cluster_name, cluster_info in clusters.items():
            print(f"  {cluster_name}: {cluster_info['file_count']} files")
        
        print(f"\nClustering completed successfully with {len(clusters)} clusters.")
        print(f"Clustered invoices are available in: {extract_dir}")
    else:
        print("Clustering failed.")

if __name__ == "__main__":
    main()
```

### Using the API with cURL

You can also use the API with cURL:

```bash
# Submit a clustering job
curl -X POST -F "zip_file=@invoices.zip" -F "method=kmeans" -F "visualize=true" http://localhost:8080/cluster_invoice

# Check the status of a job
curl http://localhost:8080/cluster_status/1623456789

# Download the clustered invoices
curl -o clustered_invoices.zip http://localhost:8080/download_clusters/1623456789
```

## Future Improvements

The current implementation of the invoice clustering API provides a solid foundation, but there are several potential improvements that could enhance its functionality:

### Clustering Enhancements

- **Hierarchical Clustering**: Implement hierarchical clustering to create a tree-like structure of clusters
- **Interactive Visualization**: Create an interactive web-based visualization of clusters that allows users to explore the data
- **Cluster Labeling**: Automatically generate descriptive labels for each cluster based on the common characteristics of invoices within it
- **Anomaly Detection**: Identify outliers that don't fit well into any cluster, which could indicate unusual or fraudulent invoices
- **Incremental Clustering**: Allow adding new invoices to existing clusters without reprocessing the entire dataset

### Performance Improvements

- **Distributed Processing**: Implement distributed processing for handling large datasets more efficiently
- **Caching**: Cache embeddings and intermediate results to speed up repeated operations
- **Batch Processing**: Process invoices in batches to optimize memory usage and processing time
- **GPU Acceleration**: Utilize GPU acceleration for embedding generation and clustering algorithms

### User Experience

- **Progress Tracking**: Provide more detailed progress tracking during the clustering process
- **Cluster Comparison**: Allow users to compare different clustering results (e.g., K-means vs. DBSCAN)
- **Custom Metadata**: Allow users to add custom metadata to invoices and use it in the clustering process
- **Saved Configurations**: Enable users to save and reuse clustering configurations

### Integration

- **API Authentication**: Add authentication to the API endpoints for security
- **Webhook Notifications**: Implement webhook notifications for job completion
- **Export Formats**: Support additional export formats beyond ZIP files (e.g., JSON reports, CSV summaries)
- **Integration with Document Management Systems**: Provide connectors for popular document management systems

These improvements could be implemented in future versions of the API to enhance its capabilities and user experience.

## Conclusion

The Invoice Processing API provides a powerful set of tools for working with invoice images, from basic data extraction to sophisticated clustering and question answering. The clustering functionality in particular offers a valuable way to organize and analyze large collections of invoices, making it easier to identify patterns, detect anomalies, and process similar documents together.

By leveraging modern machine learning techniques like vector embeddings and clustering algorithms, the API can automatically group invoices based on their content similarity, without requiring manual tagging or classification. This can save significant time and effort when dealing with large volumes of invoices from different vendors and in different formats.

The API is designed to be flexible and extensible, with a modular architecture that makes it easy to add new features and capabilities. The background processing system ensures that even large clustering jobs can be handled efficiently without blocking the API, and the job status tracking allows users to monitor the progress of their clustering tasks.

Whether you're building a document management system, an invoice processing pipeline, or a financial analysis tool, the Invoice Processing API provides the building blocks you need to work effectively with invoice data. -->