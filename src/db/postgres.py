import os
import logging
import yaml
import psycopg2
import getpass
import json
from psycopg2.extras import execute_values, Json
from typing import List, Dict, Any, Optional, Union
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_system_username():
    """
    Get the current system username.
    
    Returns:
        The username of the current user
    """
    try:
        # Try different methods to get the username
        username = os.environ.get('USERNAME') or os.environ.get('USER')
        if not username:
            username = getpass.getuser()
        if not username:
            username = os.path.expanduser('~').split(os.sep)[-1]
        
        logger.info(f"Using system username: {username}")
        return username
    except Exception as e:
        logger.warning(f"Could not determine system username: {str(e)}")
        return "unknown_user"

# Get the system username once at module load time
SYSTEM_USERNAME = get_system_username()

class PostgresDB:
    """Class to handle PostgreSQL database operations."""
    
    def __init__(self, config_path: str = "credentials/secrets.yaml"):
        """
        Initialize the PostgreSQL database connection.
        
        Args:
            config_path: Path to the configuration file
        """
        logger.info(f"Initializing PostgreSQL connection from config: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            pg_config = config.get('postgres', {})
            self.host = pg_config.get('host', 'localhost')
            self.port = pg_config.get('port', 5432)
            self.database = pg_config.get('database', 'rag_invoice')
            self.user = pg_config.get('user', 'postgres')
            self.password = pg_config.get('password', 'postgres')
            
            # Initialize connection to None
            self.conn = None
            self.cursor = None
            
            # Connect to the database
            self._connect()
            
            # Initialize the database schema if needed
            self._init_schema()
            
            # Ensure pgvector extension is enabled
            self._ensure_pgvector()
            
            logger.info(f"PostgreSQL connection initialized to {self.host}:{self.port}/{self.database}")
            
        except Exception as e:
            logger.exception(f"Error initializing PostgreSQL connection: {str(e)}")
            raise
    
    def _connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
                self.conn.autocommit = True
                self.cursor = self.conn.cursor()
                logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.exception(f"Error connecting to PostgreSQL: {str(e)}")
            # Create a mock connection for fallback
            self._create_mock_connection()
    
    def _create_mock_connection(self):
        """Create a mock connection for fallback when database is unavailable."""
        logger.warning("Creating mock database connection for fallback")
        self.is_mock = True
        
        # Define mock methods to prevent errors when database is unavailable
        class MockCursor:
            def execute(self, *args, **kwargs):
                logger.warning(f"Mock execute: {args}")
                return None
                
            def fetchall(self, *args, **kwargs):
                return []
                
            def fetchone(self, *args, **kwargs):
                return None
                
            def close(self):
                pass
        
        class MockConnection:
            def __init__(self):
                self.closed = False
                
            def cursor(self):
                return MockCursor()
                
            def commit(self):
                pass
                
            def close(self):
                pass
        
        self.conn = MockConnection()
        self.cursor = self.conn.cursor()
    
    def _init_schema(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            # Read the schema SQL file
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            with open(schema_path, 'r') as file:
                schema_sql = file.read()
            
            # Execute the schema SQL
            self.cursor.execute(schema_sql)
            logger.info("Database schema initialized")
        except Exception as e:
            logger.exception(f"Error initializing database schema: {str(e)}")
    
    def _ensure_pgvector(self):
        """Ensure the pgvector extension is enabled."""
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.exception(f"Error enabling pgvector extension: {str(e)}")
            logger.warning("Vector operations may not be available")
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning("Closing mock database connection")
            return
            
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn and not self.conn.closed:
                self.conn.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.exception(f"Error closing database connection: {str(e)}")
    
    def track_token_usage(self, username: str, image: str, model: str, token_count: int, use_case: str = None):
        """
        Track token usage for an LLM API call.
        
        Args:
            username: Name or ID of the user
            image: Path to the image file
            model: Name of the model used
            token_count: Number of tokens used
            use_case: API endpoint used for the LLM call (e.g., generate_embeddings, get_summary)
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock token tracking: {username}, {image}, {model}, {token_count}, {use_case}")
            return
            
        try:
            self._connect()  # Ensure connection is active
            
            query = """
                INSERT INTO llm_token_tracking (date, time, username, image, model, token_count, use_case)
                VALUES (CURRENT_DATE, CURRENT_TIME, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (username, image, model, token_count, use_case))
            logger.info(f"Tracked token usage: {token_count} tokens for {model} ({use_case})")
        except Exception as e:
            logger.exception(f"Error tracking token usage: {str(e)}")
    
    def store_embedding(self, username: str, image: str, embedding: List[float]):
        """
        Store an embedding for an image using pgvector.
        If an embedding already exists for the image, it will be updated.
        
        Args:
            username: Name or ID of the user
            image: Path to the image file
            embedding: The embedding vector
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock embedding storage: {username}, {image}, {len(embedding)} dimensions")
            return
            
        try:
            self._connect()  # Ensure connection is active
            
            # Convert the embedding list to a pgvector compatible format
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            # Check if an embedding already exists for this image
            check_query = """
                SELECT id FROM created_embeddings
                WHERE image = %s
                LIMIT 1
            """
            
            self.cursor.execute(check_query, (image,))
            existing_embedding = self.cursor.fetchone()
            
            if existing_embedding:
                # Update the existing embedding
                update_query = """
                    DELETE FROM created_embeddings
                    WHERE image = %s
                """
                self.cursor.execute(update_query, (image,))
                logger.info(f"Deleted existing embedding for {image}")
                
                # Insert the new embedding
                insert_query = """
                    INSERT INTO created_embeddings (date, time, username, image, embedding)
                    VALUES (CURRENT_DATE, CURRENT_TIME, %s, %s, %s::vector)
                """
                self.cursor.execute(insert_query, (username, image, embedding_str))
                logger.info(f"Updated embedding for {image}: {len(embedding)} dimensions")
            else:
                # Insert a new embedding
                insert_query = """
                    INSERT INTO created_embeddings (date, time, username, image, embedding)
                    VALUES (CURRENT_DATE, CURRENT_TIME, %s, %s, %s::vector)
                """
                self.cursor.execute(insert_query, (username, image, embedding_str))
                logger.info(f"Stored new embedding for {image}: {len(embedding)} dimensions")
        except Exception as e:
            logger.exception(f"Error storing embedding: {str(e)}")
    
    def store_invoice_summary(self, username: str, image: str, summary: Dict[str, Any]):
        """
        Store an invoice summary with the updated schema.
        
        Args:
            username: Name or ID of the user
            image: Path to the image file
            summary: The invoice summary dictionary
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock invoice summary storage: {username}, {image}")
            return
            
        try:
            self._connect()  # Ensure connection is active
            
            # Extract fields from the summary
            vendor_name = summary.get('vendor_name')
            invoice_number = summary.get('invoice_number')
            invoice_date = summary.get('invoice_date')
            due_date = summary.get('due_date')
            billing_address = summary.get('billing_address')
            shipping_address = summary.get('shipping_address')
            line_items = json.dumps(summary.get('line_items', []))
            subtotal = summary.get('subtotal')
            taxes = summary.get('taxes')
            total_amount = summary.get('total_amount')
            currency = summary.get('currency')
            raw_data = json.dumps(summary)
            
            query = """
                INSERT INTO invoice_summaries (
                    date, time, username, image, 
                    vendor_name, invoice_number, invoice_date, due_date,
                    billing_address, shipping_address, line_items,
                    subtotal, taxes, total_amount, currency, raw_data
                )
                VALUES (
                    CURRENT_DATE, CURRENT_TIME, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s::jsonb,
                    %s, %s, %s, %s, %s::jsonb
                )
            """
            
            self.cursor.execute(query, (
                username, image, 
                vendor_name, invoice_number, invoice_date, due_date,
                billing_address, shipping_address, line_items,
                subtotal, taxes, total_amount, currency, raw_data
            ))
            
            logger.info(f"Stored invoice summary for {image}")
        except Exception as e:
            logger.exception(f"Error storing invoice summary: {str(e)}")
    
    def get_embedding(self, image: str) -> Optional[List[float]]:
        """
        Retrieve an embedding for an image.
        
        Args:
            image: Path to the image file
            
        Returns:
            The embedding vector if found, None otherwise
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock embedding retrieval: {image}")
            return None
            
        try:
            self._connect()  # Ensure connection is active
            
            query = """
                SELECT embedding
                FROM created_embeddings
                WHERE image = %s
                ORDER BY date DESC, time DESC
                LIMIT 1
            """
            
            self.cursor.execute(query, (image,))
            result = self.cursor.fetchone()
            
            if result:
                logger.info(f"Retrieved embedding for {image}")
                # Convert pgvector result to a list of floats
                embedding_vector = result[0]
                if isinstance(embedding_vector, str):
                    # If it's a string, parse it
                    embedding_vector = json.loads(embedding_vector.replace('{', '[').replace('}', ']'))
                return list(embedding_vector)
            else:
                logger.info(f"No embedding found for {image}")
                return None
        except Exception as e:
            logger.exception(f"Error retrieving embedding: {str(e)}")
            return None
    
    def find_similar_images(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find images with similar embeddings using pgvector.
        
        Args:
            embedding: The query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with image paths and similarity scores
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock similarity search")
            return []
            
        try:
            self._connect()  # Ensure connection is active
            
            # Convert the embedding list to a pgvector compatible format
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            query = """
                SELECT image, 1 - (embedding <-> %s::vector) as similarity
                FROM created_embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """
            
            self.cursor.execute(query, (embedding_str, embedding_str, limit))
            results = self.cursor.fetchall()
            
            # Convert to list of dictionaries
            similar_images = [{"image": row[0], "similarity": row[1]} for row in results]
            
            logger.info(f"Found {len(similar_images)} similar images")
            return similar_images
        except Exception as e:
            logger.exception(f"Error finding similar images: {str(e)}")
            return []
    
    def get_invoice_summary(self, image: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an invoice summary for an image.
        
        Args:
            image: Path to the image file
            
        Returns:
            The invoice summary dictionary if found, None otherwise
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock invoice summary retrieval: {image}")
            return None
            
        try:
            self._connect()  # Ensure connection is active
            
            query = """
                SELECT raw_data
                FROM invoice_summaries
                WHERE image = %s
                ORDER BY date DESC, time DESC
                LIMIT 1
            """
            
            self.cursor.execute(query, (image,))
            result = self.cursor.fetchone()
            
            if result:
                logger.info(f"Retrieved invoice summary for {image}")
                # Parse JSON string to dictionary
                if isinstance(result[0], str):
                    return json.loads(result[0])
                return result[0]
            else:
                logger.info(f"No invoice summary found for {image}")
                return None
        except Exception as e:
            logger.exception(f"Error retrieving invoice summary: {str(e)}")
            return None
    
    def get_token_usage(self, username: Optional[str] = None, 
                        start_date: Optional[datetime.date] = None,
                        end_date: Optional[datetime.date] = None,
                        use_case: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get token usage statistics.
        
        Args:
            username: Filter by username (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            use_case: Filter by use case (optional)
            
        Returns:
            List of token usage records
        """
        if hasattr(self, 'is_mock') and self.is_mock:
            logger.warning(f"Mock token usage retrieval")
            return []
            
        try:
            self._connect()  # Ensure connection is active
            
            query = """
                SELECT date, username, model, use_case, SUM(token_count) as total_tokens
                FROM llm_token_tracking
                WHERE 1=1
            """
            params = []
            
            if username:
                query += " AND username = %s"
                params.append(username)
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
                
            if use_case:
                query += " AND use_case = %s"
                params.append(use_case)
            
            query += " GROUP BY date, username, model, use_case ORDER BY date DESC"
            
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = ['date', 'username', 'model', 'use_case', 'total_tokens']
            records = [dict(zip(columns, row)) for row in results]
            
            logger.info(f"Retrieved {len(records)} token usage records")
            return records
        except Exception as e:
            logger.exception(f"Error retrieving token usage: {str(e)}")
            return []

# Singleton instance
_db_instance = None

def get_db_instance():
    """Get the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = PostgresDB()
    return _db_instance

# Functions to be used by the API

def track_api_call(username: str, image: str, model: str, token_count: int = 0, use_case: str = None):
    """
    Track an API call in the database.
    
    Args:
        username: Name or ID of the user
        image: Path to the image file
        model: Name of the model used
        token_count: Number of tokens used
        use_case: API endpoint used for the LLM call (e.g., generate_embeddings, get_summary)
    """
    db = get_db_instance()
    db.track_token_usage(username, image, model, token_count, use_case)

def store_embedding_in_db(username: str, image: str, embedding: List[float]):
    """
    Store an embedding in the database.
    If an embedding already exists for the image, it will be updated.
    
    Args:
        username: Name or ID of the user
        image: Path to the image file
        embedding: The embedding vector
    """
    db = get_db_instance()
    db.store_embedding(username, image, embedding)

def store_summary_in_db(username: str, image: str, summary: Dict[str, Any]):
    """
    Store a summary in the database.
    
    Args:
        username: Name or ID of the user
        image: Path to the image file
        summary: The summary dictionary
    """
    db = get_db_instance()
    db.store_invoice_summary(username, image, summary)

def get_embedding_from_db(image: str) -> Optional[List[float]]:
    """
    Retrieve an embedding from the database.
    
    Args:
        image: Path to the image file
        
    Returns:
        The embedding vector if found, None otherwise
    """
    db = get_db_instance()
    return db.get_embedding(image)

def find_similar_images(embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find images with similar embeddings.
    
    Args:
        embedding: The query embedding vector
        limit: Maximum number of results to return
        
    Returns:
        List of dictionaries with image paths and similarity scores
    """
    db = get_db_instance()
    return db.find_similar_images(embedding, limit)

def get_summary_from_db(image: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a summary from the database.
    
    Args:
        image: Path to the image file
        
    Returns:
        The summary dictionary if found, None otherwise
    """
    db = get_db_instance()
    return db.get_invoice_summary(image)

def get_token_usage_stats(username: Optional[str] = None, 
                         start_date: Optional[datetime.date] = None,
                         end_date: Optional[datetime.date] = None,
                         use_case: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get token usage statistics from the database.
    
    Args:
        username: Filter by username (optional)
        start_date: Filter by start date (optional)
        end_date: Filter by end date (optional)
        use_case: Filter by use case (optional)
        
    Returns:
        List of token usage records
    """
    db = get_db_instance()
    return db.get_token_usage(username, start_date, end_date, use_case)

def store_cluster_job(job_id: str, file_name: str, images: int, embedding_model: str, cluster_model: str):
    """
    Store a new clustering job in the database.
    
    Args:
        job_id: Unique identifier for the job
        file_name: Name of the uploaded zip file
        images: Number of images in the uploaded file
        embedding_model: LLM model used for generating embeddings
        cluster_model: Clustering algorithm used (kmeans or dbscan)
    """
    db = get_db_instance()
    
    if hasattr(db, 'is_mock') and db.is_mock:
        logger.warning(f"Mock cluster job storage: {job_id}, {file_name}")
        return
        
    try:
        db._connect()  # Ensure connection is active
        
        query = """
            INSERT INTO cluster_jobs (job_id, file_name, images, embedding_model, cluster_model, status)
            VALUES (%s, %s, %s, %s, %s, 'queued')
        """
        
        db.cursor.execute(query, (job_id, file_name, images, embedding_model, cluster_model))
        logger.info(f"Stored new clustering job: {job_id}")
    except Exception as e:
        logger.exception(f"Error storing clustering job: {str(e)}")

def update_cluster_job(job_id: str, status: str, num_clusters: Optional[int] = None, processing_time: Optional[float] = None):
    """
    Update a clustering job in the database with results.
    
    Args:
        job_id: Unique identifier for the job
        status: Current status of the job (processing, completed, failed)
        num_clusters: Number of clusters formed (if completed)
        processing_time: Time taken for the clustering process in seconds (if completed)
    """
    db = get_db_instance()
    
    if hasattr(db, 'is_mock') and db.is_mock:
        logger.warning(f"Mock cluster job update: {job_id}, {status}")
        return
        
    try:
        db._connect()  # Ensure connection is active
        
        query = """
            UPDATE cluster_jobs
            SET status = %s
        """
        params = [status]
        
        if num_clusters is not None:
            query += ", num_clusters = %s"
            params.append(num_clusters)
            
        if processing_time is not None:
            query += ", processing_time = %s"
            params.append(processing_time)
            
        query += " WHERE job_id = %s"
        params.append(job_id)
        
        db.cursor.execute(query, params)
        logger.info(f"Updated clustering job {job_id} with status: {status}")
    except Exception as e:
        logger.exception(f"Error updating clustering job: {str(e)}")

def get_cluster_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a clustering job from the database.
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Dictionary with job information if found, None otherwise
    """
    db = get_db_instance()
    
    if hasattr(db, 'is_mock') and db.is_mock:
        logger.warning(f"Mock cluster job retrieval: {job_id}")
        return None
        
    try:
        db._connect()  # Ensure connection is active
        
        query = """
            SELECT job_id, date, time, file_name, images, embedding_model,
                   cluster_model, num_clusters, processing_time, status
            FROM cluster_jobs
            WHERE job_id = %s
        """
        
        db.cursor.execute(query, (job_id,))
        result = db.cursor.fetchone()
        
        if result:
            columns = ['job_id', 'date', 'time', 'file_name', 'images', 'embedding_model',
                      'cluster_model', 'num_clusters', 'processing_time', 'status']
            return dict(zip(columns, result))
        else:
            return None
    except Exception as e:
        logger.exception(f"Error retrieving clustering job: {str(e)}")
        return None

