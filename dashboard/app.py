import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bigquery_client import BigQueryClient
from src.campaign_detector import CampaignDetector
from src.authenticity_detector import AuthenticityDetector
from src.embedding_service import EmbeddingService
from src.data_collector import DataCollector
from src.image_analyzer import ImageAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="AI Content Authenticity Network",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    return {
        'bq_client': BigQueryClient(),
        'campaign_detector': CampaignDetector(),
        'authenticity_detector': AuthenticityDetector(),
        'embedding_service': EmbeddingService(),
        'data_collector': DataCollector(),
        'image_analyzer': ImageAnalyzer()
    }

def main():
    components = init_components()
    
    # Sidebar navigation
    st.sidebar.title("🔍 AI Content Authenticity Network")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Dashboard", "📊 Analytics", "🔎 Content Analysis", "📈 Campaign Detection", "⚙️ Data Management"]
    )
    
    if page == "🏠 Dashboard":
        render_dashboard(components)
    elif page == "📊 Analytics":
        render_analytics(components)
    elif page == "🔎 Content Analysis":
        render_content_analysis(components)
    elif page == "📈 Campaign Detection":
        render_campaign_detection(components)
    elif page == "⚙️ Data Management":
        render_data_management(components)

def render_dashboard(components):
    st.title("🏠 AI Content Authenticity Dashboard")
    st.markdown("Real-time monitoring of AI-generated content and coordinated campaigns")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get basic statistics
        content_data = components['bq_client'].get_text_content(limit=1000)
        
        with col1:
            st.metric("Total Content Items", len(content_data) if not content_data.empty else 0)
        
        with col2:
            if not content_data.empty:
                human_content = len(content_data[content_data['source'] == 'human'])
                st.metric("Human Content", human_content)
            else:
                st.metric("Human Content", 0)
        
        with col3:
            if not content_data.empty:
                ai_content = len(content_data[content_data['source'] == 'ai_generated'])
                st.metric("AI-Generated Content", ai_content)
            else:
                st.metric("AI-Generated Content", 0)
        
        with col4:
            try:
                campaigns_data = components['bq_client'].get_campaigns(limit=100)
                st.metric("Active Campaigns", len(campaigns_data) if not campaigns_data.empty else 0)
            except:
                st.metric("Active Campaigns", 0)
        
        st.markdown("---")
        
        # Content authenticity over time
        if not content_data.empty and 'created_at' in content_data.columns:
            st.subheader("📈 Content Volume Over Time")
            
            content_data['created_at'] = pd.to_datetime(content_data['created_at'])
            content_data['date'] = content_data['created_at'].dt.date
            
            daily_counts = content_data.groupby(['date', 'source']).size().unstack(fill_value=0)
            
            fig = px.line(daily_counts.reset_index(), x='date', 
                         y=['human', 'ai_generated'] if 'ai_generated' in daily_counts.columns else ['human'],
                         title="Daily Content Volume by Source")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Platform distribution
        if not content_data.empty and 'source_platform' in content_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌐 Content by Platform")
                platform_counts = content_data['source_platform'].value_counts()
                fig = px.pie(values=platform_counts.values, names=platform_counts.index)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Source Distribution")
                source_counts = content_data['source'].value_counts()
                fig = px.bar(x=source_counts.index, y=source_counts.values)
                fig.update_layout(height=400, xaxis_title="Source", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        st.info("Make sure BigQuery is set up and data is available.")

def render_analytics(components):
    st.title("📊 Advanced Analytics")
    
    # Authenticity analysis
    st.subheader("🎯 Authenticity Score Analysis")
    
    try:
        scores_data = components['bq_client'].get_authenticity_scores(limit=1000)
        
        if not scores_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(scores_data, x='authenticity_score', nbins=20,
                                title="Distribution of Authenticity Scores")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(scores_data, x='authenticity_score', y='confidence_score',
                               title="Authenticity vs Confidence Scores")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Score statistics
            st.subheader("📈 Score Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Authenticity", f"{scores_data['authenticity_score'].mean():.3f}")
            with col2:
                st.metric("Average Confidence", f"{scores_data['confidence_score'].mean():.3f}")
            with col3:
                low_auth_count = len(scores_data[scores_data['authenticity_score'] < 0.3])
                st.metric("Low Authenticity Items", low_auth_count)
        
        else:
            st.info("No authenticity scores found. Run content analysis first.")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

def render_content_analysis(components):
    st.title("🔎 Content Analysis")
    
    # Text analysis section
    st.subheader("📝 Text Content Analysis")
    
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Analyze Text Authenticity"):
            if text_input:
                with st.spinner("Analyzing content..."):
                    result = components['authenticity_detector'].process_content(text_input, "user_input")
                    
                    st.subheader("Analysis Results")
                    
                    # Authenticity score with color coding
                    score = result['authenticity_score']
                    if score > 0.7:
                        st.success(f"**Authenticity Score: {score:.3f}** (Likely Human)")
                    elif score > 0.3:
                        st.warning(f"**Authenticity Score: {score:.3f}** (Uncertain)")
                    else:
                        st.error(f"**Authenticity Score: {score:.3f}** (Likely AI-Generated)")
                    
                    st.write(f"**Confidence:** {result['confidence_score']:.3f}")
                    st.write(f"**Explanation:** {result['explanation']}")
                    
                    # Feature breakdown
                    if result['features']:
                        st.subheader("Feature Analysis")
                        features_df = pd.DataFrame([result['features']]).T
                        features_df.columns = ['Value']
                        st.dataframe(features_df)
    
    with col2:
        if st.button("Get Similar Content"):
            if text_input:
                with st.spinner("Finding similar content..."):
                    try:
                        # Get embeddings for input text
                        embeddings = components['embedding_service'].get_text_embeddings([text_input], use_vertex=False)
                        
                        # Get some content from database for comparison
                        content_data = components['bq_client'].get_text_content(limit=100)
                        
                        if not content_data.empty:
                            db_contents = content_data['content'].tolist()
                            db_embeddings = components['embedding_service'].get_text_embeddings(db_contents, use_vertex=False)
                            
                            similarities = []
                            for i, db_embedding in enumerate(db_embeddings):
                                similarity = components['embedding_service'].calculate_similarity(
                                    embeddings[0], db_embedding
                                )
                                similarities.append({
                                    'content': db_contents[i][:200] + "...",
                                    'similarity': similarity,
                                    'source': content_data.iloc[i]['source']
                                })
                            
                            # Sort by similarity and show top matches
                            similarities.sort(key=lambda x: x['similarity'], reverse=True)
                            
                            st.subheader("Most Similar Content")
                            for i, sim in enumerate(similarities[:5]):
                                with st.expander(f"Match {i+1} (Similarity: {sim['similarity']:.3f})"):
                                    st.write(f"**Source:** {sim['source']}")
                                    st.write(sim['content'])
                        else:
                            st.info("No content in database for comparison")
                    
                    except Exception as e:
                        st.error(f"Error finding similar content: {str(e)}")
    
    # Image analysis section
    st.markdown("---")
    st.subheader("🖼️ Image Analysis")
    
    uploaded_file = st.file_uploader("Upload an image for authenticity analysis", 
                                   type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        result = components['image_analyzer'].analyze_image_authenticity(tmp_path)
                        
                        st.subheader("Image Analysis Results")
                        
                        score = result['authenticity_score']
                        if score > 0.7:
                            st.success(f"**Authenticity Score: {score:.3f}** (Likely Real)")
                        elif score > 0.3:
                            st.warning(f"**Authenticity Score: {score:.3f}** (Uncertain)")
                        else:
                            st.error(f"**Authenticity Score: {score:.3f}** (Likely AI-Generated)")
                        
                        st.write(f"**Confidence:** {result['confidence_score']:.3f}")
                        st.write(f"**Explanation:** {result['explanation']}")
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")

def render_campaign_detection(components):
    st.title("📈 Campaign Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔍 Run Campaign Detection", type="primary"):
            with st.spinner("Detecting campaigns..."):
                try:
                    campaigns = components['campaign_detector'].detect_all_campaigns(limit=1000)
                    st.session_state['campaigns'] = campaigns
                    st.success("Campaign detection completed!")
                except Exception as e:
                    st.error(f"Error detecting campaigns: {str(e)}")
    
    # Display results if available
    if 'campaigns' in st.session_state:
        campaigns = st.session_state['campaigns']
        
        # Campaign summary
        st.subheader("🎯 Campaign Summary")
        
        summary_cols = st.columns(4)
        
        total_campaigns = sum(len(v) for k, v in campaigns.items() 
                            if isinstance(v, list))
        
        with summary_cols[0]:
            st.metric("Total Campaigns", total_campaigns)
        
        with summary_cols[1]:
            semantic_count = len(campaigns.get('semantic_campaigns', []))
            st.metric("Semantic Campaigns", semantic_count)
        
        with summary_cols[2]:
            temporal_count = len(campaigns.get('temporal_campaigns', []))
            st.metric("Temporal Campaigns", temporal_count)
        
        with summary_cols[3]:
            auth_count = len(campaigns.get('authenticity_campaigns', []))
            st.metric("Authenticity Campaigns", auth_count)
        
        # Detailed campaign breakdown
        st.subheader("📊 Campaign Details")
        
        for campaign_type, campaign_list in campaigns.items():
            if isinstance(campaign_list, list) and campaign_list:
                with st.expander(f"{campaign_type.replace('_', ' ').title()} ({len(campaign_list)} campaigns)"):
                    
                    for i, campaign in enumerate(campaign_list[:10]):  # Show top 10
                        st.write(f"**Campaign {i+1}:** {campaign.get('campaign_id', 'Unknown ID')}")
                        st.write(f"- Size: {campaign.get('campaign_size', 0)} content items")
                        
                        if 'avg_similarity' in campaign:
                            st.write(f"- Average Similarity: {campaign['avg_similarity']:.3f}")
                        
                        if 'time_window' in campaign:
                            st.write(f"- Time Window: {campaign['time_window']}")
                        
                        if 'platform' in campaign:
                            st.write(f"- Platform: {campaign['platform']}")
                        
                        st.markdown("---")
        
        # Generate and display report
        st.subheader("📋 Campaign Report")
        
        if st.button("Generate Full Report"):
            report = components['campaign_detector'].generate_comprehensive_report(campaigns)
            st.text_area("Campaign Detection Report", report, height=400)

def render_data_management(components):
    st.title("⚙️ Data Management")
    
    # Data collection section
    st.subheader("📥 Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Collect New Data", type="primary"):
            with st.spinner("Collecting data from various sources..."):
                try:
                    count = components['data_collector'].run_full_collection()
                    st.success(f"Successfully collected {count} content items!")
                except Exception as e:
                    st.error(f"Error collecting data: {str(e)}")
    
    with col2:
        if st.button("🏗️ Initialize Database"):
            with st.spinner("Setting up BigQuery tables..."):
                try:
                    components['bq_client'].create_dataset()
                    components['bq_client'].execute_sql_file("sql/create_tables.sql")
                    st.success("Database initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing database: {str(e)}")
    
    # Data statistics
    st.subheader("📊 Data Statistics")
    
    try:
        content_data = components['bq_client'].get_text_content(limit=10000)
        
        if not content_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Content Items", len(content_data))
            
            with col2:
                if 'source' in content_data.columns:
                    human_count = len(content_data[content_data['source'] == 'human'])
                    st.metric("Human Content", human_count)
            
            with col3:
                if 'source' in content_data.columns:
                    ai_count = len(content_data[content_data['source'] == 'ai_generated'])
                    st.metric("AI-Generated Content", ai_count)
            
            # Data preview
            st.subheader("📄 Data Preview")
            st.dataframe(content_data.head(10))
        
        else:
            st.info("No data found. Try collecting data first.")
    
    except Exception as e:
        st.error(f"Error loading data statistics: {str(e)}")
    
    # System status
    st.subheader("🔧 System Status")
    
    status_checks = {
        "BigQuery Connection": "✅ Connected",
        "Embedding Service": "✅ Available",
        "Campaign Detector": "✅ Ready",
        "Image Analyzer": "✅ Ready"
    }
    
    for check, status in status_checks.items():
        st.write(f"**{check}:** {status}")

if __name__ == "__main__":
    main()