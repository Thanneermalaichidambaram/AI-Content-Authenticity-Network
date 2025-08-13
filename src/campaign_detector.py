import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import uuid
import json
from collections import defaultdict

from .similarity_detector import SimilarityDetector
from .embedding_service import EmbeddingService
from .authenticity_detector import AuthenticityDetector
from .bigquery_client import BigQueryClient
from .config import Config

class CampaignDetector:
    def __init__(self):
        self.similarity_detector = SimilarityDetector()
        self.embedding_service = EmbeddingService()
        self.authenticity_detector = AuthenticityDetector()
        self.bq_client = BigQueryClient()
    
    def detect_all_campaigns(self, limit: int = 5000) -> Dict[str, Any]:
        """Detect all types of campaigns from the database"""
        print("Starting comprehensive campaign detection...")
        
        try:
            # Get content data from BigQuery
            content_data = self.bq_client.get_text_content(limit=limit)
        except Exception as e:
            print(f"⚠️ Could not access BigQuery data: {e}")
            print("📝 Using sample data for campaign detection...")
            # Create sample data for testing
            sample_data = self._create_sample_data()
            content_data = pd.DataFrame(sample_data)
        
        if content_data.empty:
            print("No content data found in database")
            return {'error': 'No content data available'}
        
        print(f"Analyzing {len(content_data)} content items...")
        
        # Run comprehensive campaign detection
        campaigns = self.similarity_detector.detect_coordinated_campaigns(content_data)
        
        # Add additional campaign types
        campaigns.update(self.detect_authenticity_based_campaigns(content_data))
        campaigns.update(self.detect_behavioral_campaigns(content_data))
        
        # Store detected campaigns in BigQuery (optional)
        try:
            self.store_campaigns(campaigns)
        except Exception as e:
            print(f"⚠️ Could not store campaigns in BigQuery: {e}")
            print("📝 Campaign results available in memory only")
        
        return campaigns
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample data for testing when BigQuery is not available"""
        from datetime import datetime, timedelta
        import uuid
        
        sample_data = []
        
        # Coordinated campaign simulation
        campaign_templates = [
            "This amazing product has changed my life! Everyone should try it now! #amazing",
            "This incredible product has transformed my life! Everyone should get it now! #incredible", 
            "This fantastic product has revolutionized my life! Everyone should buy it now! #fantastic"
        ]
        
        # Add coordinated content
        for i, template in enumerate(campaign_templates):
            for j in range(3):  # 3 instances of each
                sample_data.append({
                    'id': f'campaign_{i}_{j}',
                    'content': template,
                    'source': 'ai_generated',
                    'source_platform': 'social_media',
                    'timestamp': datetime.utcnow() - timedelta(minutes=j*5),
                    'created_at': datetime.utcnow()
                })
        
        # Add some normal content
        normal_content = [
            "Just had a great coffee at the local cafe. The weather is nice today.",
            "Working on an interesting project. Learning a lot about machine learning.",
            "Visited the museum yesterday. The art exhibition was fascinating.",
            "Reading a good book about artificial intelligence. Very insightful.",
            "Enjoyed a walk in the park. Nature is so beautiful in spring."
        ]
        
        for i, content in enumerate(normal_content):
            sample_data.append({
                'id': f'normal_{i}',
                'content': content,
                'source': 'human',
                'source_platform': 'social_media',
                'timestamp': datetime.utcnow() - timedelta(hours=i),
                'created_at': datetime.utcnow()
            })
        
        return sample_data
    
    def detect_authenticity_based_campaigns(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect campaigns based on authenticity patterns"""
        campaigns = {
            'authenticity_campaigns': [],
            'low_authenticity_clusters': []
        }
        
        if 'content' not in content_data.columns:
            return campaigns
        
        print("Detecting authenticity-based campaigns...")
        
        # Get authenticity scores for all content
        authenticity_results = []
        for _, row in content_data.iterrows():
            result = self.authenticity_detector.process_content(row['content'], row['id'])
            authenticity_results.append(result)
        
        # Find clusters of low-authenticity content
        low_auth_content = [r for r in authenticity_results if r['authenticity_score'] < 0.3]
        
        if len(low_auth_content) >= Config.CAMPAIGN_MIN_SIZE:
            # Group by similar authenticity scores and explanations
            auth_groups = defaultdict(list)
            
            for content in low_auth_content:
                # Group by explanation (similar AI patterns)
                key = content['explanation'][:50]  # First 50 chars of explanation
                auth_groups[key].append(content)
            
            for explanation_key, group in auth_groups.items():
                if len(group) >= Config.CAMPAIGN_MIN_SIZE:
                    campaigns['authenticity_campaigns'].append({
                        'campaign_id': f"auth_{uuid.uuid4().hex[:8]}",
                        'content_ids': [c['content_id'] for c in group],
                        'campaign_size': len(group),
                        'avg_authenticity_score': np.mean([c['authenticity_score'] for c in group]),
                        'common_explanation': explanation_key,
                        'campaign_type': 'authenticity_pattern',
                        'detected_at': datetime.utcnow()
                    })
        
        return campaigns
    
    def detect_behavioral_campaigns(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect campaigns based on behavioral patterns"""
        campaigns = {
            'behavioral_campaigns': [],
            'volume_spikes': [],
            'length_pattern_campaigns': []
        }
        
        print("Detecting behavioral campaigns...")
        
        # 1. Volume spike detection
        if 'timestamp' in content_data.columns:
            campaigns['volume_spikes'] = self.detect_volume_spikes(content_data)
        
        # 2. Content length pattern detection
        if 'content' in content_data.columns:
            campaigns['length_pattern_campaigns'] = self.detect_length_patterns(content_data)
        
        # 3. Platform behavior patterns
        if 'source_platform' in content_data.columns:
            behavioral_patterns = self.detect_platform_behavior_patterns(content_data)
            campaigns['behavioral_campaigns'].extend(behavioral_patterns)
        
        return campaigns
    
    def detect_volume_spikes(self, content_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual volume spikes in content posting"""
        volume_spikes = []
        
        # Convert timestamp and group by hour
        content_data = content_data.copy()
        content_data['timestamp'] = pd.to_datetime(content_data['timestamp'])
        content_data['hour'] = content_data['timestamp'].dt.floor('H')
        
        hourly_counts = content_data.groupby('hour').size()
        
        # Calculate baseline and identify spikes
        mean_volume = hourly_counts.mean()
        std_volume = hourly_counts.std()
        spike_threshold = mean_volume + (2 * std_volume)  # 2 standard deviations
        
        spikes = hourly_counts[hourly_counts > spike_threshold]
        
        for hour, count in spikes.items():
            spike_content = content_data[content_data['hour'] == hour]
            
            if len(spike_content) >= Config.CAMPAIGN_MIN_SIZE:
                volume_spikes.append({
                    'campaign_id': f"volume_spike_{hour.strftime('%Y%m%d_%H')}",
                    'content_ids': spike_content['id'].tolist(),
                    'campaign_size': int(count),
                    'time_window': hour,
                    'volume_multiplier': float(count / mean_volume),
                    'campaign_type': 'volume_spike',
                    'detected_at': datetime.utcnow()
                })
        
        return volume_spikes
    
    def detect_length_patterns(self, content_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect suspicious content length patterns"""
        length_campaigns = []
        
        # Calculate content lengths
        content_data = content_data.copy()
        content_data['content_length'] = content_data['content'].str.len()
        
        # Group by similar lengths (within 10% range)
        length_groups = defaultdict(list)
        
        for _, row in content_data.iterrows():
            length = row['content_length']
            length_bucket = int(length / (length * 0.1 + 10))  # Group similar lengths
            length_groups[length_bucket].append(row)
        
        # Find suspiciously uniform length groups
        for length_bucket, group in length_groups.items():
            if len(group) >= Config.CAMPAIGN_MIN_SIZE:
                lengths = [item['content_length'] for item in group]
                length_std = np.std(lengths)
                length_mean = np.mean(lengths)
                
                # Suspiciously uniform lengths (low coefficient of variation)
                cv = length_std / length_mean if length_mean > 0 else 0
                
                if cv < 0.1:  # Very consistent lengths
                    length_campaigns.append({
                        'campaign_id': f"length_pattern_{length_bucket}_{uuid.uuid4().hex[:8]}",
                        'content_ids': [item['id'] for item in group],
                        'campaign_size': len(group),
                        'avg_length': float(length_mean),
                        'length_consistency': float(1 - cv),
                        'campaign_type': 'length_pattern',
                        'detected_at': datetime.utcnow()
                    })
        
        return length_campaigns
    
    def detect_platform_behavior_patterns(self, content_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect suspicious platform-specific behavior patterns"""
        platform_campaigns = []
        
        platform_groups = content_data.groupby('source_platform')
        
        for platform, group in platform_groups:
            if len(group) >= Config.CAMPAIGN_MIN_SIZE:
                # Analyze posting frequency patterns
                if 'timestamp' in group.columns:
                    timestamps = pd.to_datetime(group['timestamp']).sort_values()
                    time_diffs = timestamps.diff().dropna()
                    
                    if len(time_diffs) > 1:
                        # Calculate interval statistics
                        avg_interval = time_diffs.mean()
                        interval_std = time_diffs.std()
                        
                        # Detect bot-like regular intervals
                        if interval_std < avg_interval * 0.2:  # Very regular posting
                            platform_campaigns.append({
                                'campaign_id': f"platform_behavior_{platform}_{uuid.uuid4().hex[:8]}",
                                'content_ids': group['id'].tolist(),
                                'campaign_size': len(group),
                                'platform': platform,
                                'avg_posting_interval_minutes': float(avg_interval.total_seconds() / 60),
                                'posting_regularity': float(1 - (interval_std / avg_interval)),
                                'campaign_type': 'platform_behavior',
                                'detected_at': datetime.utcnow()
                            })
        
        return platform_campaigns
    
    def detect_multimodal_campaigns(self, text_content: pd.DataFrame, 
                                  image_content: pd.DataFrame = None) -> Dict[str, Any]:
        """Detect campaigns that span both text and image content"""
        multimodal_campaigns = {'multimodal_campaigns': []}
        
        if image_content is None or image_content.empty:
            return multimodal_campaigns
        
        print("Detecting multimodal campaigns...")
        
        # Find temporal correlations between text and image content
        if 'timestamp' in text_content.columns and 'timestamp' in image_content.columns:
            text_content = text_content.copy()
            image_content = image_content.copy()
            
            text_content['timestamp'] = pd.to_datetime(text_content['timestamp'])
            image_content['timestamp'] = pd.to_datetime(image_content['timestamp'])
            
            # Group by time windows
            text_content['time_window'] = text_content['timestamp'].dt.floor('H')
            image_content['time_window'] = image_content['timestamp'].dt.floor('H')
            
            # Find time windows with both text and image content
            text_windows = set(text_content['time_window'])
            image_windows = set(image_content['time_window'])
            common_windows = text_windows.intersection(image_windows)
            
            for window in common_windows:
                text_in_window = text_content[text_content['time_window'] == window]
                images_in_window = image_content[image_content['time_window'] == window]
                
                if len(text_in_window) >= 2 and len(images_in_window) >= 2:
                    multimodal_campaigns['multimodal_campaigns'].append({
                        'campaign_id': f"multimodal_{window.strftime('%Y%m%d_%H')}",
                        'text_content_ids': text_in_window['id'].tolist(),
                        'image_content_ids': images_in_window['id'].tolist(),
                        'campaign_size': len(text_in_window) + len(images_in_window),
                        'time_window': window,
                        'campaign_type': 'multimodal_temporal',
                        'detected_at': datetime.utcnow()
                    })
        
        return multimodal_campaigns
    
    def store_campaigns(self, campaigns: Dict[str, Any]):
        """Store detected campaigns in BigQuery"""
        campaign_records = []
        
        # Process all campaign types
        for campaign_type, campaign_list in campaigns.items():
            if campaign_type == 'summary':
                continue
            
            if isinstance(campaign_list, list):
                for campaign in campaign_list:
                    if 'content_ids' in campaign:
                        campaign_record = {
                            'campaign_id': campaign.get('campaign_id', str(uuid.uuid4())),
                            'content_ids': campaign['content_ids'],
                            'similarity_score': campaign.get('avg_similarity', campaign.get('similarity_score', 0.0)),
                            'campaign_size': campaign.get('campaign_size', len(campaign['content_ids'])),
                            'campaign_type': campaign.get('campaign_type', campaign_type),
                            'metadata': {
                                'detection_method': campaign_type,
                                'details': {k: v for k, v in campaign.items() 
                                          if k not in ['content_ids', 'campaign_id', 'campaign_size']}
                            }
                        }
                        campaign_records.append(campaign_record)
        
        if campaign_records:
            try:
                self.bq_client.client.dataset(Config.BIGQUERY_DATASET).table(Config.CAMPAIGNS_TABLE)
                
                # Insert campaign records
                rows_to_insert = []
                for record in campaign_records:
                    row = {
                        'campaign_id': record['campaign_id'],
                        'content_ids': record['content_ids'],
                        'similarity_score': float(record['similarity_score']),
                        'campaign_size': int(record['campaign_size']),
                        'campaign_type': record['campaign_type'],
                        'metadata': json.dumps(record['metadata'])
                    }
                    rows_to_insert.append(row)
                
                table_ref = self.bq_client.client.dataset(Config.BIGQUERY_DATASET).table(Config.CAMPAIGNS_TABLE)
                table = self.bq_client.client.get_table(table_ref)
                
                # Use load job instead of streaming insert for free tier compatibility
                import pandas as pd
                from google.cloud import bigquery
                df = pd.DataFrame(rows_to_insert)
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND",
                    create_disposition="CREATE_NEVER"
                )
                
                try:
                    job = self.bq_client.client.load_table_from_dataframe(df, table, job_config=job_config)
                    job.result()  # Wait for the job to complete
                    print(f"Successfully stored {len(rows_to_insert)} campaigns")
                except Exception as e:
                    print(f"Error inserting campaigns: {e}")
                    
            except Exception as e:
                print(f"Error storing campaigns: {e}")
    
    def generate_comprehensive_report(self, campaigns: Dict[str, Any]) -> str:
        """Generate a comprehensive campaign detection report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AI CONTENT AUTHENTICITY - CAMPAIGN DETECTION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Executive Summary
        total_campaigns = sum(len(v) for k, v in campaigns.items() 
                            if isinstance(v, list) and k != 'summary')
        
        report_lines.append("EXECUTIVE SUMMARY:")
        report_lines.append(f"• Total campaigns detected: {total_campaigns}")
        
        # Count by type
        campaign_types = {}
        for campaign_type, campaign_list in campaigns.items():
            if isinstance(campaign_list, list) and campaign_list:
                campaign_types[campaign_type] = len(campaign_list)
        
        for campaign_type, count in campaign_types.items():
            report_lines.append(f"• {campaign_type.replace('_', ' ').title()}: {count}")
        
        report_lines.append("")
        
        # Detailed findings
        for campaign_type, campaign_list in campaigns.items():
            if isinstance(campaign_list, list) and campaign_list:
                report_lines.append(f"{campaign_type.replace('_', ' ').upper()}:")
                
                for i, campaign in enumerate(campaign_list[:5]):  # Show top 5
                    campaign_id = campaign.get('campaign_id', f'Unknown_{i}')
                    campaign_size = campaign.get('campaign_size', 0)
                    
                    report_lines.append(f"  {i+1}. {campaign_id}")
                    report_lines.append(f"     Size: {campaign_size} content items")
                    
                    if 'avg_similarity' in campaign:
                        report_lines.append(f"     Avg Similarity: {campaign['avg_similarity']:.3f}")
                    
                    if 'avg_authenticity_score' in campaign:
                        report_lines.append(f"     Avg Authenticity: {campaign['avg_authenticity_score']:.3f}")
                
                if len(campaign_list) > 5:
                    report_lines.append(f"  ... and {len(campaign_list) - 5} more")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if total_campaigns == 0:
            report_lines.append("• No suspicious campaigns detected. Continue monitoring.")
        else:
            report_lines.append("• Investigate high-similarity content clusters")
            report_lines.append("• Review temporal posting patterns for automation")
            report_lines.append("• Analyze cross-platform coordination")
            report_lines.append("• Implement real-time monitoring for new campaigns")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)