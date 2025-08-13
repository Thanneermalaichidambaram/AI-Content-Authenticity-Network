import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'authenticity_network')
    VERTEX_AI_LOCATION = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Optional - only for comparison testing
    
    # BigQuery table names
    TEXT_CONTENT_TABLE = 'text_content'
    IMAGE_CONTENT_TABLE = 'image_content'
    AUTHENTICITY_SCORES_TABLE = 'authenticity_scores'
    CAMPAIGNS_TABLE = 'campaigns'
    EMBEDDINGS_TABLE = 'embeddings'
    
    # Model parameters
    EMBEDDING_MODEL = 'textembedding-gecko@001'
    AUTHENTICITY_THRESHOLD = 0.7
    SIMILARITY_THRESHOLD = 0.85
    CAMPAIGN_MIN_SIZE = 5