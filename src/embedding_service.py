from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import numpy as np
from typing import List, Dict, Any
import time
import asyncio
import concurrent.futures
from transformers import AutoTokenizer, AutoModel
import torch

from .config import Config

class EmbeddingService:
    def __init__(self):
        self.vertex_model = None
        try:
            aiplatform.init(project=Config.GOOGLE_CLOUD_PROJECT, location=Config.VERTEX_AI_LOCATION)
            self.vertex_model = TextEmbeddingModel.from_pretrained(Config.EMBEDDING_MODEL)
            print("✅ Vertex AI initialized successfully")
        except Exception as e:
            print(f"⚠️ Vertex AI not available (billing required): {str(e)[:100]}...")
            print("📝 Using local embeddings instead")
        
        # Fallback to local model if Vertex AI is not available
        self.local_tokenizer = None
        self.local_model = None
        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.local_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Could not load local embedding model: {e}")
    
    def get_text_embeddings(self, texts: List[str], use_vertex: bool = False) -> List[List[float]]:
        # Default to local embeddings since Vertex AI requires billing
        if use_vertex and self.vertex_model:
            try:
                return self._get_vertex_embeddings(texts)
            except Exception as e:
                print(f"Vertex AI embedding failed, falling back to local model: {e}")
                return self._get_local_embeddings(texts)
        else:
            return self._get_local_embeddings(texts)
    
    def _get_vertex_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 5  # Vertex AI rate limiting
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.vertex_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error getting embeddings for batch {i}: {e}")
                # Fallback to zeros for failed batch
                embeddings.extend([[0.0] * 768 for _ in batch])
        
        return embeddings
    
    def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.local_model or not self.local_tokenizer:
            print("Local model not available, returning random embeddings")
            return [[np.random.random() for _ in range(384)] for _ in texts]
        
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.local_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = self.local_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
                embeddings.append(embedding)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_content(self, query_embedding: List[float], 
                           candidate_embeddings: List[List[float]], 
                           threshold: float = Config.SIMILARITY_THRESHOLD) -> List[int]:
        similar_indices = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate_embedding)
            if similarity >= threshold:
                similar_indices.append(i)
        
        return similar_indices
    
    def batch_process_embeddings(self, texts: List[str], content_ids: List[str], 
                                content_type: str = 'text') -> List[Dict[str, Any]]:
        embeddings = self.get_text_embeddings(texts)
        
        result = []
        for content_id, embedding in zip(content_ids, embeddings):
            result.append({
                'content_id': content_id,
                'content_type': content_type,
                'embedding': embedding,
                'model_name': Config.EMBEDDING_MODEL
            })
        
        return result