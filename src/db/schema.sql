-- Schema for Invoice Embedding API Database

-- Enable the pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to track LLM token usage
CREATE TABLE IF NOT EXISTS llm_token_tracking (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    time TIME NOT NULL DEFAULT CURRENT_TIME,
    username VARCHAR(100) NOT NULL,
    image VARCHAR(255) NOT NULL,
    model VARCHAR(100) NOT NULL,
    token_count INTEGER NOT NULL,
    use_case VARCHAR(100) DEFAULT NULL
);

-- Table to store embeddings for images using pgvector
CREATE TABLE IF NOT EXISTS created_embeddings (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    time TIME NOT NULL DEFAULT CURRENT_TIME,
    username VARCHAR(100) NOT NULL,
    image VARCHAR(255) NOT NULL,
    embedding vector(1536) NOT NULL -- Using pgvector's vector type with dimension 1536
);

-- Table to store invoice summaries with updated schema
CREATE TABLE IF NOT EXISTS invoice_summaries (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    time TIME NOT NULL DEFAULT CURRENT_TIME,
    username VARCHAR(100) NOT NULL,
    image VARCHAR(255) NOT NULL,
    vendor_name VARCHAR(255),
    invoice_number VARCHAR(100),
    invoice_date VARCHAR(100),
    due_date VARCHAR(100),
    billing_address TEXT,
    shipping_address TEXT,
    line_items JSONB,
    subtotal VARCHAR(100),
    taxes VARCHAR(100),
    total_amount VARCHAR(100),
    currency VARCHAR(50),
    raw_data JSONB NOT NULL -- Store the complete raw JSON data
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_token_tracking_date ON llm_token_tracking(date);
CREATE INDEX IF NOT EXISTS idx_token_tracking_username ON llm_token_tracking(username);
CREATE INDEX IF NOT EXISTS idx_token_tracking_model ON llm_token_tracking(model);
CREATE INDEX IF NOT EXISTS idx_token_tracking_use_case ON llm_token_tracking(use_case);

CREATE INDEX IF NOT EXISTS idx_embeddings_date ON created_embeddings(date);
CREATE INDEX IF NOT EXISTS idx_embeddings_username ON created_embeddings(username);
CREATE INDEX IF NOT EXISTS idx_embeddings_image ON created_embeddings(image);

-- Create a vector index for faster similarity searches
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON created_embeddings USING ivfflat (embedding vector_l2_ops);

CREATE INDEX IF NOT EXISTS idx_summaries_date ON invoice_summaries(date);
CREATE INDEX IF NOT EXISTS idx_summaries_username ON invoice_summaries(username);
CREATE INDEX IF NOT EXISTS idx_summaries_image ON invoice_summaries(image);
CREATE INDEX IF NOT EXISTS idx_summaries_vendor_name ON invoice_summaries(vendor_name);
CREATE INDEX IF NOT EXISTS idx_summaries_invoice_number ON invoice_summaries(invoice_number);
CREATE INDEX IF NOT EXISTS idx_summaries_invoice_date ON invoice_summaries(invoice_date);

-- Comments for documentation
COMMENT ON TABLE llm_token_tracking IS 'Tracks token usage for LLM API calls';
COMMENT ON TABLE created_embeddings IS 'Stores embeddings generated for invoice images using pgvector';
COMMENT ON TABLE invoice_summaries IS 'Stores structured invoice data extracted from images';
COMMENT ON COLUMN llm_token_tracking.use_case IS 'API endpoint used for the LLM call (e.g., generate_embeddings, get_summary)';
COMMENT ON COLUMN created_embeddings.embedding IS 'Vector embedding of the image using pgvector with dimension 1536';
COMMENT ON COLUMN invoice_summaries.raw_data IS 'Complete JSON data extracted from the invoice';

-- Table to track clustering jobs
CREATE TABLE IF NOT EXISTS cluster_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(100) NOT NULL,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    time TIME NOT NULL DEFAULT CURRENT_TIME,
    file_name VARCHAR(255) NOT NULL,
    images INTEGER NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    cluster_model VARCHAR(100) NOT NULL,
    num_clusters INTEGER,
    processing_time FLOAT,
    status VARCHAR(50) NOT NULL DEFAULT 'queued'
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_cluster_jobs_job_id ON cluster_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_cluster_jobs_date ON cluster_jobs(date);
CREATE INDEX IF NOT EXISTS idx_cluster_jobs_status ON cluster_jobs(status);

-- Comments for documentation
COMMENT ON TABLE cluster_jobs IS 'Tracks clustering jobs and their results';
COMMENT ON COLUMN cluster_jobs.job_id IS 'Unique identifier for the clustering job';
COMMENT ON COLUMN cluster_jobs.file_name IS 'Name of the uploaded zip file';
COMMENT ON COLUMN cluster_jobs.images IS 'Number of images in the uploaded file';
COMMENT ON COLUMN cluster_jobs.embedding_model IS 'LLM model used for generating embeddings';
COMMENT ON COLUMN cluster_jobs.cluster_model IS 'Clustering algorithm used (kmeans or dbscan)';
COMMENT ON COLUMN cluster_jobs.num_clusters IS 'Number of clusters formed';
COMMENT ON COLUMN cluster_jobs.processing_time IS 'Time taken for the clustering process in seconds';
COMMENT ON COLUMN cluster_jobs.status IS 'Current status of the job (queued, processing, completed, failed)';