import numpy as np
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from .config import Config
from .embedding_service import EmbeddingService
from .bigquery_client import BigQueryClient

class SimilarityDetector:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.bq_client = BigQueryClient()
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
    
    def find_similar_content_pairs(self, embeddings: List[List[float]], 
                                 content_ids: List[str],
                                 threshold: float = None) -> List[Dict[str, Any]]:
        """Find pairs of similar content based on embeddings"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        similar_pairs = []
        embeddings_array = np.array(embeddings)
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings_array)
        
        for i in range(len(content_ids)):
            for j in range(i + 1, len(content_ids)):
                similarity_score = similarity_matrix[i][j]
                
                if similarity_score >= threshold:
                    similar_pairs.append({
                        'content_id_1': content_ids[i],
                        'content_id_2': content_ids[j],
                        'similarity_score': float(similarity_score),
                        'similarity_type': 'semantic'
                    })
        
        return similar_pairs
    
    def detect_content_clusters(self, embeddings: List[List[float]], 
                              content_ids: List[str],
                              min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Detect clusters of similar content using DBSCAN"""
        if len(embeddings) < min_cluster_size:
            return []
        
        embeddings_array = np.array(embeddings)
        
        # Use DBSCAN for clustering
        # eps parameter controls the maximum distance between points in a cluster
        eps = 1 - self.similarity_threshold  # Convert similarity to distance
        dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine')
        
        cluster_labels = dbscan.fit_predict(embeddings_array)
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_content_ids = [content_ids[i] for i in cluster_indices]
            
            # Calculate average similarity within cluster
            cluster_embeddings = embeddings_array[cluster_indices]
            similarity_matrix = cosine_similarity(cluster_embeddings)
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            
            clusters.append({
                'cluster_id': f"cluster_{label}",
                'content_ids': cluster_content_ids,
                'cluster_size': len(cluster_content_ids),
                'avg_similarity': float(avg_similarity),
                'cluster_type': 'semantic_similarity'
            })
        
        return clusters
    
    def analyze_temporal_patterns(self, content_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in content posting"""
        temporal_clusters = []
        
        if 'timestamp' not in content_data.columns:
            return temporal_clusters
        
        # Convert timestamp to datetime if it's not already
        content_data = content_data.copy()
        content_data['timestamp'] = pd.to_datetime(content_data['timestamp'])
        
        # Group by time windows (1-hour intervals)
        content_data['time_window'] = content_data['timestamp'].dt.floor('H')
        
        time_groups = content_data.groupby('time_window')
        
        for time_window, group in time_groups:
            if len(group) >= Config.CAMPAIGN_MIN_SIZE:
                # Check if content is suspiciously similar
                if 'content' in group.columns:
                    contents = group['content'].tolist()
                    content_ids = group['id'].tolist()
                    
                    embeddings = self.embedding_service.get_text_embeddings(contents, use_vertex=False)
                    similar_pairs = self.find_similar_content_pairs(embeddings, content_ids, threshold=0.8)
                    
                    if len(similar_pairs) > 0:
                        temporal_clusters.append({
                            'cluster_id': f"temporal_{time_window.strftime('%Y%m%d_%H')}",
                            'content_ids': content_ids,
                            'cluster_size': len(content_ids),
                            'time_window': time_window,
                            'cluster_type': 'temporal_pattern',
                            'similar_pairs_count': len(similar_pairs)
                        })
        
        return temporal_clusters
    
    def detect_coordinated_campaigns(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect coordinated campaigns combining multiple signals"""
        campaigns = {
            'semantic_campaigns': [],
            'temporal_campaigns': [],
            'network_campaigns': [],
            'summary': {}
        }
        
        if len(content_data) < Config.CAMPAIGN_MIN_SIZE:
            return campaigns
        
        # 1. Semantic similarity campaigns
        if 'content' in content_data.columns:
            contents = content_data['content'].tolist()
            content_ids = content_data['id'].tolist()
            
            embeddings = self.embedding_service.get_text_embeddings(contents, use_vertex=False)
            semantic_clusters = self.detect_content_clusters(embeddings, content_ids)
            campaigns['semantic_campaigns'] = semantic_clusters
        
        # 2. Temporal pattern campaigns
        temporal_clusters = self.analyze_temporal_patterns(content_data)
        campaigns['temporal_campaigns'] = temporal_clusters
        
        # 3. Network-based campaigns (if platform data available)
        if 'source_platform' in content_data.columns:
            network_clusters = self.detect_network_patterns(content_data)
            campaigns['network_campaigns'] = network_clusters
        
        # 4. Cross-platform campaigns
        cross_platform_campaigns = self.detect_cross_platform_campaigns(content_data)
        campaigns['cross_platform_campaigns'] = cross_platform_campaigns
        
        # Generate summary
        campaigns['summary'] = {
            'total_semantic_campaigns': len(campaigns['semantic_campaigns']),
            'total_temporal_campaigns': len(campaigns['temporal_campaigns']),
            'total_network_campaigns': len(campaigns['network_campaigns']),
            'total_cross_platform_campaigns': len(campaigns.get('cross_platform_campaigns', [])),
            'total_suspicious_content': self._count_suspicious_content(campaigns),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        return campaigns
    
    def detect_network_patterns(self, content_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect network-based suspicious patterns"""
        network_clusters = []
        
        # Group by platform and analyze posting patterns
        if 'source_platform' in content_data.columns:
            platform_groups = content_data.groupby('source_platform')
            
            for platform, group in platform_groups:
                if len(group) >= Config.CAMPAIGN_MIN_SIZE:
                    # Analyze posting frequency patterns
                    if 'timestamp' in group.columns:
                        timestamps = pd.to_datetime(group['timestamp'])
                        time_diffs = timestamps.diff().dropna()
                        
                        # Check for suspicious regular intervals
                        if len(time_diffs) > 1:
                            avg_interval = time_diffs.mean()
                            interval_std = time_diffs.std()
                            
                            # Suspiciously regular posting (low variance)
                            if interval_std < avg_interval * 0.1:  # Very regular posting
                                network_clusters.append({
                                    'cluster_id': f"network_{platform}_{datetime.utcnow().strftime('%Y%m%d')}",
                                    'content_ids': group['id'].tolist(),
                                    'cluster_size': len(group),
                                    'platform': platform,
                                    'avg_posting_interval': str(avg_interval),
                                    'interval_consistency': float(1 - (interval_std / avg_interval)),
                                    'cluster_type': 'network_pattern'
                                })
        
        return network_clusters
    
    def detect_cross_platform_campaigns(self, content_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect campaigns that span multiple platforms"""
        cross_platform_campaigns = []
        
        if 'source_platform' not in content_data.columns or 'content' not in content_data.columns:
            return cross_platform_campaigns
        
        platforms = content_data['source_platform'].unique()
        
        if len(platforms) < 2:
            return cross_platform_campaigns
        
        # Compare content across different platforms
        for i, platform1 in enumerate(platforms):
            for platform2 in platforms[i+1:]:
                platform1_data = content_data[content_data['source_platform'] == platform1]
                platform2_data = content_data[content_data['source_platform'] == platform2]
                
                if len(platform1_data) >= 2 and len(platform2_data) >= 2:
                    # Get embeddings for both platforms
                    contents1 = platform1_data['content'].tolist()
                    contents2 = platform2_data['content'].tolist()
                    
                    embeddings1 = self.embedding_service.get_text_embeddings(contents1, use_vertex=False)
                    embeddings2 = self.embedding_service.get_text_embeddings(contents2, use_vertex=False)
                    
                    # Find cross-platform similarities
                    cross_similarities = []
                    for idx1, emb1 in enumerate(embeddings1):
                        for idx2, emb2 in enumerate(embeddings2):
                            similarity = self.embedding_service.calculate_similarity(emb1, emb2)
                            if similarity >= 0.85:  # High threshold for cross-platform
                                cross_similarities.append({
                                    'content_id_1': platform1_data.iloc[idx1]['id'],
                                    'content_id_2': platform2_data.iloc[idx2]['id'],
                                    'similarity': similarity,
                                    'platform_1': platform1,
                                    'platform_2': platform2
                                })
                    
                    if len(cross_similarities) >= 2:  # At least 2 cross-platform matches
                        all_content_ids = list(set([s['content_id_1'] for s in cross_similarities] + 
                                                 [s['content_id_2'] for s in cross_similarities]))
                        
                        cross_platform_campaigns.append({
                            'cluster_id': f"cross_platform_{platform1}_{platform2}_{datetime.utcnow().strftime('%Y%m%d')}",
                            'content_ids': all_content_ids,
                            'cluster_size': len(all_content_ids),
                            'platforms': [platform1, platform2],
                            'cross_similarities': cross_similarities,
                            'cluster_type': 'cross_platform_campaign'
                        })
        
        return cross_platform_campaigns
    
    def build_similarity_graph(self, embeddings: List[List[float]], 
                             content_ids: List[str],
                             threshold: float = None) -> nx.Graph:
        """Build a graph of similar content for network analysis"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        G = nx.Graph()
        
        # Add nodes
        for content_id in content_ids:
            G.add_node(content_id)
        
        # Add edges for similar content
        similar_pairs = self.find_similar_content_pairs(embeddings, content_ids, threshold)
        
        for pair in similar_pairs:
            G.add_edge(pair['content_id_1'], pair['content_id_2'], 
                      weight=pair['similarity_score'])
        
        return G
    
    def analyze_similarity_network(self, embeddings: List[List[float]], 
                                 content_ids: List[str]) -> Dict[str, Any]:
        """Analyze the network structure of similar content"""
        G = self.build_similarity_graph(embeddings, content_ids)
        
        analysis = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'connected_components': [],
            'network_metrics': {}
        }
        
        # Find connected components (separate campaigns)
        for i, component in enumerate(nx.connected_components(G)):
            if len(component) >= Config.CAMPAIGN_MIN_SIZE:
                subgraph = G.subgraph(component)
                
                analysis['connected_components'].append({
                    'component_id': f"component_{i}",
                    'nodes': list(component),
                    'size': len(component),
                    'edges': subgraph.number_of_edges(),
                    'density': nx.density(subgraph),
                    'avg_clustering': nx.average_clustering(subgraph)
                })
        
        # Overall network metrics
        if G.number_of_nodes() > 0:
            analysis['network_metrics'] = {
                'density': nx.density(G),
                'number_of_components': nx.number_connected_components(G),
                'avg_clustering': nx.average_clustering(G)
            }
        
        return analysis
    
    def _count_suspicious_content(self, campaigns: Dict[str, Any]) -> int:
        """Count total unique content items involved in campaigns"""
        all_content_ids = set()
        
        for campaign_type in ['semantic_campaigns', 'temporal_campaigns', 'network_campaigns']:
            if campaign_type in campaigns:
                for campaign in campaigns[campaign_type]:
                    if 'content_ids' in campaign:
                        all_content_ids.update(campaign['content_ids'])
        
        return len(all_content_ids)
    
    def generate_campaign_report(self, campaigns: Dict[str, Any]) -> str:
        """Generate a human-readable report of detected campaigns"""
        report_lines = []
        report_lines.append("=== AI Content Authenticity Campaign Detection Report ===")
        report_lines.append(f"Analysis completed at: {campaigns['summary']['analysis_timestamp']}")
        report_lines.append("")
        
        # Summary section
        summary = campaigns['summary']
        report_lines.append("SUMMARY:")
        report_lines.append(f"- Semantic campaigns detected: {summary['total_semantic_campaigns']}")
        report_lines.append(f"- Temporal campaigns detected: {summary['total_temporal_campaigns']}")
        report_lines.append(f"- Network campaigns detected: {summary['total_network_campaigns']}")
        report_lines.append(f"- Total suspicious content items: {summary['total_suspicious_content']}")
        report_lines.append("")
        
        # Detailed findings
        if campaigns['semantic_campaigns']:
            report_lines.append("SEMANTIC SIMILARITY CAMPAIGNS:")
            for campaign in campaigns['semantic_campaigns']:
                report_lines.append(f"- Cluster {campaign['cluster_id']}: {campaign['cluster_size']} items, avg similarity: {campaign['avg_similarity']:.3f}")
            report_lines.append("")
        
        if campaigns['temporal_campaigns']:
            report_lines.append("TEMPORAL PATTERN CAMPAIGNS:")
            for campaign in campaigns['temporal_campaigns']:
                report_lines.append(f"- Cluster {campaign['cluster_id']}: {campaign['cluster_size']} items in time window")
            report_lines.append("")
        
        return "\n".join(report_lines)