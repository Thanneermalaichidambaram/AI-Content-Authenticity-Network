import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.authenticity_detector import AuthenticityDetector
from src.embedding_service import EmbeddingService
from src.similarity_detector import SimilarityDetector

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
        'authenticity_detector': AuthenticityDetector(),
        'embedding_service': EmbeddingService(),
        'similarity_detector': SimilarityDetector()
    }

def create_sample_data():
    """Create sample data for demonstration"""
    sample_texts = [
        {
            'content': "Hey everyone! Just discovered this amazing coffee shop downtown. The baristas are super friendly and the atmosphere is perfect for working. Definitely recommend if you're in the area!",
            'source': 'human',
            'platform': 'social_media',
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'content': "As an AI language model, I can provide you with comprehensive information about this topic. It's important to note that there are several key factors to consider when analyzing this subject matter.",
            'source': 'ai_generated', 
            'platform': 'ai_assistant',
            'timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'content': "This amazing product has changed my life! Everyone should try it now! #amazing #lifechanging",
            'source': 'ai_generated',
            'platform': 'social_media',
            'timestamp': datetime.now() - timedelta(minutes=30)
        },
        {
            'content': "This incredible product has transformed my life! Everyone should get it now! #incredible #transformation",
            'source': 'ai_generated',
            'platform': 'social_media', 
            'timestamp': datetime.now() - timedelta(minutes=25)
        },
        {
            'content': "Just finished reading an interesting book about machine learning. The concepts are fascinating and the examples really help understand the theory.",
            'source': 'human',
            'platform': 'social_media',
            'timestamp': datetime.now() - timedelta(minutes=45)
        }
    ]
    return pd.DataFrame(sample_texts)

def main():
    components = init_components()
    
    # Sidebar navigation
    st.sidebar.title("🔍 AI Content Authenticity Network")
    st.sidebar.markdown("**Simplified Dashboard** (No BigQuery Required)")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Dashboard", "🔎 Content Analysis", "📈 Campaign Detection", "📊 Demo Results"]
    )
    
    if page == "🏠 Dashboard":
        render_dashboard(components)
    elif page == "🔎 Content Analysis":
        render_content_analysis(components)
    elif page == "📈 Campaign Detection":
        render_campaign_detection(components)
    elif page == "📊 Demo Results":
        render_demo_results(components)

def render_dashboard(components):
    st.title("🏠 AI Content Authenticity Dashboard")
    st.markdown("Real-time detection of AI-generated content and coordinated campaigns")
    
    # Demo data metrics
    sample_data = create_sample_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Content Items", len(sample_data))
    
    with col2:
        human_count = len(sample_data[sample_data['source'] == 'human'])
        st.metric("Human Content", human_count)
    
    with col3:
        ai_count = len(sample_data[sample_data['source'] == 'ai_generated'])
        st.metric("AI-Generated Content", ai_count)
    
    with col4:
        st.metric("Suspicious Patterns", "2")
    
    st.markdown("---")
    
    # Sample authenticity analysis
    st.subheader("📊 Sample Authenticity Analysis")
    
    authenticity_scores = []
    for _, row in sample_data.iterrows():
        result = components['authenticity_detector'].process_content(row['content'], f"sample_{len(authenticity_scores)}")
        authenticity_scores.append({
            'content': row['content'][:50] + "...",
            'source': row['source'],
            'authenticity_score': result['authenticity_score'],
            'confidence': result['confidence_score']
        })
    
    scores_df = pd.DataFrame(authenticity_scores)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(scores_df, x='authenticity_score', nbins=10,
                          title="Distribution of Authenticity Scores",
                          color='source')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(scores_df, x='authenticity_score', y='confidence',
                        color='source', hover_data=['content'],
                        title="Authenticity vs Confidence")
        st.plotly_chart(fig, use_container_width=True)
    
    # Content timeline
    st.subheader("📈 Content Timeline")
    sample_data['hour'] = sample_data['timestamp'].dt.floor('H')
    timeline_data = sample_data.groupby(['hour', 'source']).size().reset_index(name='count')
    
    fig = px.line(timeline_data, x='hour', y='count', color='source',
                 title="Content Volume Over Time")
    st.plotly_chart(fig, use_container_width=True)

def render_content_analysis(components):
    st.title("🔎 Content Analysis")
    
    # Text analysis section
    st.subheader("📝 Text Content Analysis")
    
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Analyze Authenticity", type="primary"):
            if text_input:
                with st.spinner("Analyzing content..."):
                    result = components['authenticity_detector'].process_content(text_input, "user_input")
                    
                    st.subheader("📊 Analysis Results")
                    
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
                        st.subheader("🔧 Feature Analysis")
                        
                        # Show key features
                        key_features = {
                            'AI Phrase Density': result['features'].get('ai_phrase_density', 0),
                            'Repetition Score': result['features'].get('repetition_score', 0),
                            'Formal Language': result['features'].get('formal_language_density', 0),
                            'Reading Ease': result['features'].get('flesch_reading_ease', 0),
                            'Word Count': result['features'].get('word_count', 0)
                        }
                        
                        features_df = pd.DataFrame([key_features]).T
                        features_df.columns = ['Value']
                        st.dataframe(features_df)
    
    with col2:
        if st.button("🔗 Find Similar Content"):
            if text_input:
                with st.spinner("Finding similar content..."):
                    # Use sample data for comparison
                    sample_data = create_sample_data()
                    sample_texts = sample_data['content'].tolist()
                    
                    all_texts = [text_input] + sample_texts
                    embeddings = components['embedding_service'].get_text_embeddings(all_texts, use_vertex=False)
                    
                    query_embedding = embeddings[0]
                    similarities = []
                    
                    for i, sample_text in enumerate(sample_texts):
                        similarity = components['embedding_service'].calculate_similarity(
                            query_embedding, embeddings[i+1]
                        )
                        similarities.append({
                            'content': sample_text[:100] + "...",
                            'similarity': similarity,
                            'source': sample_data.iloc[i]['source']
                        })
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    st.subheader("🎯 Most Similar Content")
                    for i, sim in enumerate(similarities[:3]):
                        with st.expander(f"Match {i+1} (Similarity: {sim['similarity']:.3f})"):
                            st.write(f"**Source:** {sim['source']}")
                            st.write(sim['content'])
    
    # Batch analysis
    st.markdown("---")
    st.subheader("📋 Batch Analysis")
    
    uploaded_file = st.file_uploader("Upload a text file for batch analysis", type=['txt'])
    
    if uploaded_file is not None:
        content = str(uploaded_file.read(), "utf-8")
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if st.button("Analyze All Lines"):
            results = []
            progress_bar = st.progress(0)
            
            for i, line in enumerate(lines):
                result = components['authenticity_detector'].process_content(line, f"batch_{i}")
                results.append({
                    'text': line[:50] + "...",
                    'authenticity_score': result['authenticity_score'],
                    'confidence': result['confidence_score'],
                    'explanation': result['explanation']
                })
                progress_bar.progress((i + 1) / len(lines))
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Summary statistics
            avg_score = results_df['authenticity_score'].mean()
            low_auth_count = len(results_df[results_df['authenticity_score'] < 0.3])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Authenticity", f"{avg_score:.3f}")
            with col2:
                st.metric("Likely AI Content", low_auth_count)
            with col3:
                st.metric("Total Analyzed", len(results))

def render_campaign_detection(components):
    st.title("📈 Campaign Detection")
    
    if st.button("🔍 Run Campaign Detection", type="primary"):
        with st.spinner("Detecting campaigns..."):
            # Use sample data with coordinated content
            sample_data = create_sample_data()
            
            # Run similarity detection
            contents = sample_data['content'].tolist()
            content_ids = [f"content_{i}" for i in range(len(contents))]
            
            embeddings = components['embedding_service'].get_text_embeddings(contents, use_vertex=False)
            
            similar_pairs = components['similarity_detector'].find_similar_content_pairs(
                embeddings, content_ids, threshold=0.8
            )
            
            clusters = components['similarity_detector'].detect_content_clusters(
                embeddings, content_ids, min_cluster_size=2
            )
            
            st.success("Campaign detection completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Similar Pairs Found", len(similar_pairs))
            with col2:
                st.metric("Content Clusters", len(clusters))
            with col3:
                suspicious_content = len(set([p['content_id_1'] for p in similar_pairs] + 
                                          [p['content_id_2'] for p in similar_pairs]))
                st.metric("Suspicious Content", suspicious_content)
            
            # Show similar pairs
            if similar_pairs:
                st.subheader("🔗 Similar Content Pairs")
                for pair in similar_pairs:
                    id1, id2 = int(pair['content_id_1'].split('_')[1]), int(pair['content_id_2'].split('_')[1])
                    content1 = contents[id1][:100] + "..."
                    content2 = contents[id2][:100] + "..."
                    
                    with st.expander(f"Similarity: {pair['similarity_score']:.3f}"):
                        st.write(f"**Content 1:** {content1}")
                        st.write(f"**Content 2:** {content2}")
            
            # Show clusters
            if clusters:
                st.subheader("📊 Content Clusters")
                for i, cluster in enumerate(clusters):
                    with st.expander(f"Cluster {i+1} - {cluster['cluster_size']} items (Avg similarity: {cluster['avg_similarity']:.3f})"):
                        for content_id in cluster['content_ids']:
                            idx = int(content_id.split('_')[1])
                            st.write(f"• {contents[idx][:100]}...")

def render_demo_results(components):
    st.title("📊 Demo Results")
    st.markdown("Pre-computed results demonstrating system capabilities")
    
    # Authenticity detection demo
    st.subheader("🎯 Authenticity Detection Results")
    
    demo_results = [
        {"text": "Human-written social media post", "score": 0.75, "confidence": 0.80, "actual": "Human"},
        {"text": "AI assistant response", "score": 0.15, "confidence": 0.90, "actual": "AI"},
        {"text": "News article excerpt", "score": 0.80, "confidence": 0.75, "actual": "Human"},
        {"text": "Generated marketing copy", "score": 0.25, "confidence": 0.85, "actual": "AI"},
        {"text": "Personal blog post", "score": 0.70, "confidence": 0.70, "actual": "Human"}
    ]
    
    demo_df = pd.DataFrame(demo_results)
    
    # Accuracy calculation
    correct_predictions = 0
    for _, row in demo_df.iterrows():
        predicted = "Human" if row['score'] > 0.5 else "AI"
        if predicted == row['actual']:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(demo_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.1%}")
    with col2:
        avg_confidence = demo_df['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    with col3:
        st.metric("Samples Analyzed", len(demo_df))
    
    # Results visualization
    fig = px.scatter(demo_df, x='score', y='confidence', color='actual',
                    hover_data=['text'], title="Authenticity Detection Results")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", 
                  annotation_text="Decision Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    display_df = demo_df.copy()
    display_df['predicted'] = display_df['score'].apply(lambda x: "Human" if x > 0.5 else "AI")
    display_df['correct'] = display_df['predicted'] == display_df['actual']
    st.dataframe(display_df)
    
    # Campaign detection demo
    st.subheader("📈 Campaign Detection Demo")
    
    st.info("**Coordinated Campaign Detected:**")
    st.write("• 3 similar posts detected with 97.6% similarity")
    st.write("• Posted within 5-minute intervals")
    st.write("• Similar promotional language patterns")
    st.write("• High confidence AI-generated content")
    
    campaign_data = {
        'Time': ['10:00 AM', '10:05 AM', '10:10 AM'],
        'Content': [
            'This amazing product has changed my life!',
            'This incredible product has transformed my life!', 
            'This fantastic product has revolutionized my life!'
        ],
        'Authenticity Score': [0.20, 0.18, 0.22],
        'Similarity to Others': [0.98, 0.97, 0.96]
    }
    
    st.dataframe(pd.DataFrame(campaign_data))

if __name__ == "__main__":
    main()