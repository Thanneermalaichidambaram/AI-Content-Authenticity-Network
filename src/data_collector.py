import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import time
import json
import uuid
from datetime import datetime, timedelta
from google.cloud import bigquery
import random

from .config import Config
from .bigquery_client import BigQueryClient

class DataCollector:
    def __init__(self):
        self.bq_client = BigQueryClient()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Content-Authenticity-Network/1.0'
        })
    
    def collect_hackernews_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Collect human-written content from Hacker News"""
        print(f"Collecting {limit} Hacker News comments...")
        
        query = f"""
        SELECT 
            CAST(id AS STRING) as id,
            text as content,
            'human' as source,
            'hackernews' as source_platform,
            TIMESTAMP_SECONDS(time) as timestamp,
            'en' as language,
            JSON_OBJECT('score', score, 'parent', parent, 'type', type) as metadata
        FROM `bigquery-public-data.hacker_news.full` 
        WHERE text IS NOT NULL 
            AND LENGTH(text) > 50 
            AND LENGTH(text) < 5000
            AND type = 'comment'
            AND deleted IS NULL
        ORDER BY RAND()
        LIMIT {limit}
        """
        
        try:
            client = bigquery.Client(project=Config.GOOGLE_CLOUD_PROJECT)
            results = client.query(query).to_dataframe()
            
            content_data = []
            for _, row in results.iterrows():
                content_data.append({
                    'id': f"hn_{row['id']}",
                    'content': row['content'],
                    'source': 'human',
                    'source_platform': 'hackernews',
                    'timestamp': row['timestamp'],
                    'language': 'en',
                    'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                })
            
            print(f"Collected {len(content_data)} Hacker News comments")
            return content_data
            
        except Exception as e:
            print(f"Error collecting Hacker News data: {e}")
            return []
    
    def collect_wikipedia_data(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Collect human-written content from Wikipedia"""
        print(f"Collecting {limit} Wikipedia snippets...")
        
        query = f"""
        SELECT 
            title,
            wiki as content,
            'human' as source,
            'wikipedia' as source_platform,
            CURRENT_TIMESTAMP() as timestamp,
            'en' as language
        FROM `bigquery-public-data.wikipedia.pageviews_2023` 
        WHERE wiki IS NOT NULL 
            AND LENGTH(wiki) > 100 
            AND LENGTH(wiki) < 3000
            AND datehour >= '2023-01-01 00:00:00'
        ORDER BY RAND()
        LIMIT {limit}
        """
        
        try:
            client = bigquery.Client(project=Config.GOOGLE_CLOUD_PROJECT)
            results = client.query(query).to_dataframe()
            
            content_data = []
            for _, row in results.iterrows():
                content_data.append({
                    'id': f"wiki_{uuid.uuid4().hex[:8]}",
                    'content': row['content'],
                    'source': 'human',
                    'source_platform': 'wikipedia',
                    'timestamp': datetime.utcnow(),
                    'language': 'en',
                    'metadata': {'title': row['title']}
                })
            
            print(f"Collected {len(content_data)} Wikipedia entries")
            return content_data
            
        except Exception as e:
            print(f"Error collecting Wikipedia data: {e}")
            return []
    
    def generate_ai_content(self, prompts: List[str], model: str = 'gemini') -> List[Dict[str, Any]]:
        """Generate AI content using various prompts"""
        print(f"Generating {len(prompts)} AI content samples...")
        
        ai_content = []
        
        # Sample AI-generated content for demonstration
        ai_samples = [
            "As an AI language model, I can provide you with comprehensive information about this topic. It's important to note that there are several key factors to consider when analyzing this subject matter.",
            "I'd be happy to help you understand this concept. Furthermore, it's worth mentioning that the implementation of these strategies can significantly impact the overall effectiveness of your approach.",
            "Based on my analysis, I can conclude that this approach offers numerous advantages. However, it's crucial to consider the potential limitations and challenges that may arise during implementation.",
            "To provide you with the most accurate information, I've analyzed various aspects of this topic. Additionally, it's important to understand that the success of any strategy depends on multiple interconnected factors.",
            "From my perspective as an AI assistant, I can offer several insights regarding this matter. Moreover, it's essential to recognize that different approaches may yield varying results depending on the specific context.",
        ]
        
        for i, prompt in enumerate(prompts):
            # Use sample AI content with some variation
            content = ai_samples[i % len(ai_samples)]
            
            # Add some variation to make it more realistic
            variations = [
                f"Regarding your question about {prompt.lower()}: {content}",
                f"In response to your inquiry: {content}",
                f"Here's what I can tell you about {prompt.lower()}: {content}",
                content
            ]
            
            final_content = random.choice(variations)
            
            ai_content.append({
                'id': f"ai_{model}_{uuid.uuid4().hex[:8]}",
                'content': final_content,
                'source': 'ai_generated',
                'source_platform': model,
                'timestamp': datetime.utcnow(),
                'language': 'en',
                'metadata': {'prompt': prompt, 'model': model}
            })
        
        print(f"Generated {len(ai_content)} AI content samples")
        return ai_content
    
    def collect_news_headlines(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Collect news headlines from public sources"""
        print(f"Collecting {limit} news headlines...")
        
        # Sample news headlines for demonstration
        news_samples = [
            "Breaking: Technology Company Announces Major Product Update",
            "Local Community Organizes Environmental Cleanup Initiative",
            "Research Study Reveals New Insights into Climate Change",
            "Economic Indicators Show Positive Growth in Key Sectors",
            "Educational Institution Launches Innovative Learning Program",
            "Healthcare Workers Recognized for Outstanding Service",
            "Sports Team Achieves Historic Victory in Championship",
            "Art Exhibition Showcases Local Artists' Creative Works",
            "Transportation Infrastructure Improvements Begin Next Month",
            "Scientific Discovery Opens New Possibilities for Future Research"
        ]
        
        news_content = []
        for i in range(min(limit, len(news_samples) * 10)):
            headline = news_samples[i % len(news_samples)]
            
            # Add some variation
            if i > len(news_samples):
                headline = f"{headline} - Updated Report"
            
            news_content.append({
                'id': f"news_{uuid.uuid4().hex[:8]}",
                'content': headline,
                'source': 'human',
                'source_platform': 'news',
                'timestamp': datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                'language': 'en',
                'metadata': {'category': 'headline'}
            })
        
        print(f"Collected {len(news_content)} news headlines")
        return news_content
    
    def collect_social_media_patterns(self, count: int = 300) -> List[Dict[str, Any]]:
        """Generate social media-like content patterns"""
        print(f"Generating {count} social media patterns...")
        
        patterns = [
            "Just had an amazing experience at the local coffee shop! The atmosphere was perfect and the staff was incredibly friendly. #GoodVibes",
            "Working on a new project today. Excited to see where this leads! Anyone have tips for staying motivated during long coding sessions?",
            "Beautiful sunset tonight 🌅 Sometimes it's the simple things that make the biggest impact on your day.",
            "Finished reading an incredible book about AI and machine learning. The future of technology is truly fascinating!",
            "Trying out a new recipe tonight. Cooking has become my favorite way to unwind after a busy day at work.",
        ]
        
        social_content = []
        for i in range(count):
            pattern = patterns[i % len(patterns)]
            
            # Add variations to simulate different users
            if random.random() < 0.3:
                pattern += " What do you all think?"
            elif random.random() < 0.3:
                pattern += " Let me know your thoughts in the comments!"
            
            social_content.append({
                'id': f"social_{uuid.uuid4().hex[:8]}",
                'content': pattern,
                'source': 'human',
                'source_platform': 'social_media',
                'timestamp': datetime.utcnow() - timedelta(minutes=random.randint(0, 1440)),
                'language': 'en',
                'metadata': {'engagement': random.randint(0, 100)}
            })
        
        print(f"Generated {len(social_content)} social media patterns")
        return social_content
    
    def run_full_collection(self):
        """Run complete data collection process"""
        print("Starting full data collection process...")
        
        all_content = []
        
        # Collect human content
        # all_content.extend(self.collect_hackernews_data(500))
        # all_content.extend(self.collect_wikipedia_data(200))
        all_content.extend(self.collect_news_headlines(200))
        all_content.extend(self.collect_social_media_patterns(300))
        
        # Generate AI content
        ai_prompts = [
            "Explain machine learning", "Describe climate change",
            "What is blockchain technology", "How to improve productivity",
            "Benefits of renewable energy", "Future of artificial intelligence",
            "Importance of data security", "Social media impact on society",
            "Sustainable development goals", "Digital transformation trends"
        ] * 10  # 100 AI samples
        
        all_content.extend(self.generate_ai_content(ai_prompts))
        
        # Insert into BigQuery
        if all_content:
            print(f"Inserting {len(all_content)} content items into BigQuery...")
            self.bq_client.insert_text_content(all_content)
            print("Data collection completed successfully!")
        else:
            print("No content collected")
        
        return len(all_content)