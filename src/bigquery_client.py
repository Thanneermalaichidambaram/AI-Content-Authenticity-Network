from google.cloud import bigquery
from google.cloud import aiplatform
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import uuid

from .config import Config

class BigQueryClient:
    def __init__(self):
        self.client = bigquery.Client(project=Config.GOOGLE_CLOUD_PROJECT)
        self.dataset_id = f"{Config.GOOGLE_CLOUD_PROJECT}.{Config.BIGQUERY_DATASET}"
        
    def create_dataset(self):
        dataset = bigquery.Dataset(self.dataset_id)
        dataset.location = "US"
        dataset.description = "AI Content Authenticity Network Dataset"
        
        try:
            dataset = self.client.create_dataset(dataset, timeout=30)
            print(f"Created dataset {dataset.dataset_id}")
        except Exception as e:
            print(f"Dataset already exists or error: {e}")
    
    def execute_sql_file(self, file_path: str, **kwargs):
        with open(file_path, 'r') as f:
            sql_content = f.read()
        
        formatted_sql = sql_content.format(project_id=Config.GOOGLE_CLOUD_PROJECT, **kwargs)
        
        for statement in formatted_sql.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    query_job = self.client.query(statement)
                    query_job.result()
                    print(f"Executed SQL statement successfully")
                except Exception as e:
                    print(f"Error executing SQL: {e}")
    
    def insert_text_content(self, content_data: List[Dict[str, Any]]):
        table_ref = self.client.dataset(Config.BIGQUERY_DATASET).table(Config.TEXT_CONTENT_TABLE)
        table = self.client.get_table(table_ref)
        
        rows_to_insert = []
        for data in content_data:
            timestamp = data.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            row = {
                'id': data.get('id', str(uuid.uuid4())),
                'content': data['content'],
                'source': data['source'],
                'source_platform': data.get('source_platform'),
                'timestamp': timestamp,
                'language': data.get('language'),
                'word_count': len(data['content'].split()) if data['content'] else 0,
                'char_count': len(data['content']) if data['content'] else 0,
                'metadata': json.dumps(data.get('metadata', {}), default=str)
            }
            rows_to_insert.append(row)
        
        # Use load job instead of streaming insert for free tier compatibility
        df = pd.DataFrame(rows_to_insert)
        
        # Convert timestamp column to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_NEVER"
        )
        
        try:
            job = self.client.load_table_from_dataframe(df, table, job_config=job_config)
            job.result()  # Wait for the job to complete
            print(f"Successfully inserted {len(rows_to_insert)} text content rows")
        except Exception as e:
            print(f"Error inserting rows: {e}")
    
    def insert_authenticity_scores(self, scores_data: List[Dict[str, Any]]):
        table_ref = self.client.dataset(Config.BIGQUERY_DATASET).table(Config.AUTHENTICITY_SCORES_TABLE)
        table = self.client.get_table(table_ref)
        
        rows_to_insert = []
        for data in scores_data:
            row = {
                'content_id': data['content_id'],
                'content_type': data['content_type'],
                'authenticity_score': float(data['authenticity_score']),
                'confidence_score': float(data['confidence_score']),
                'model_version': data.get('model_version', 'v1.0'),
                'features': json.dumps(data.get('features', {}), default=str),
                'explanation': data.get('explanation')
            }
            rows_to_insert.append(row)
        
        # Use load job instead of streaming insert for free tier compatibility
        df = pd.DataFrame(rows_to_insert)
        
        # Convert timestamp column to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_NEVER"
        )
        
        try:
            job = self.client.load_table_from_dataframe(df, table, job_config=job_config)
            job.result()  # Wait for the job to complete
            print(f"Successfully inserted {len(rows_to_insert)} authenticity scores")
        except Exception as e:
            print(f"Error inserting scores: {e}")
    
    def insert_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        table_ref = self.client.dataset(Config.BIGQUERY_DATASET).table(Config.EMBEDDINGS_TABLE)
        table = self.client.get_table(table_ref)
        
        rows_to_insert = []
        for data in embeddings_data:
            row = {
                'content_id': data['content_id'],
                'content_type': data['content_type'],
                'embedding': data['embedding'],
                'model_name': data.get('model_name', Config.EMBEDDING_MODEL)
            }
            rows_to_insert.append(row)
        
        # Use load job instead of streaming insert for free tier compatibility
        df = pd.DataFrame(rows_to_insert)
        
        # Convert timestamp column to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_NEVER"
        )
        
        try:
            job = self.client.load_table_from_dataframe(df, table, job_config=job_config)
            job.result()  # Wait for the job to complete
            print(f"Successfully inserted {len(rows_to_insert)} embeddings")
        except Exception as e:
            print(f"Error inserting embeddings: {e}")
    
    def get_text_content(self, limit: int = 1000, source: Optional[str] = None) -> pd.DataFrame:
        query = f"""
        SELECT * FROM `{self.dataset_id}.{Config.TEXT_CONTENT_TABLE}`
        {f"WHERE source = '{source}'" if source else ""}
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        return self.client.query(query).to_dataframe()
    
    def get_authenticity_scores(self, content_type: str = 'text', limit: int = 1000) -> pd.DataFrame:
        query = f"""
        SELECT * FROM `{self.dataset_id}.{Config.AUTHENTICITY_SCORES_TABLE}`
        WHERE content_type = '{content_type}'
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        return self.client.query(query).to_dataframe()
    
    def detect_campaigns(self):
        query = f"CALL `{self.dataset_id}.detect_campaigns`()"
        query_job = self.client.query(query)
        query_job.result()
        print("Campaign detection completed")
    
    def get_campaigns(self, limit: int = 100) -> pd.DataFrame:
        query = f"""
        SELECT * FROM `{self.dataset_id}.{Config.CAMPAIGNS_TABLE}`
        ORDER BY detected_at DESC
        LIMIT {limit}
        """
        return self.client.query(query).to_dataframe()