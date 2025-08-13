#!/usr/bin/env python3
"""
AI Content Authenticity Network - Main Application
Comprehensive system for detecting AI-generated content and coordinated campaigns
"""

import argparse
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🔧 Importing modules...")
try:
    from src.bigquery_client import BigQueryClient
    print("✅ BigQueryClient imported")
except Exception as e:
    print(f"❌ Error importing BigQueryClient: {e}")

try:
    from src.data_collector import DataCollector
    print("✅ DataCollector imported")
except Exception as e:
    print(f"❌ Error importing DataCollector: {e}")

try:
    from src.authenticity_detector import AuthenticityDetector
    print("✅ AuthenticityDetector imported")
except Exception as e:
    print(f"❌ Error importing AuthenticityDetector: {e}")

try:
    from src.embedding_service import EmbeddingService
    print("✅ EmbeddingService imported")
except Exception as e:
    print(f"❌ Error importing EmbeddingService: {e}")

try:
    from src.campaign_detector import CampaignDetector
    print("✅ CampaignDetector imported")
except Exception as e:
    print(f"❌ Error importing CampaignDetector: {e}")

try:
    from src.image_analyzer import ImageAnalyzer
    print("✅ ImageAnalyzer imported")
except Exception as e:
    print(f"❌ Error importing ImageAnalyzer: {e}")

try:
    from src.similarity_detector import SimilarityDetector
    print("✅ SimilarityDetector imported")
except Exception as e:
    print(f"❌ Error importing SimilarityDetector: {e}")

print("🔧 All imports completed")

def setup_database():
    """Initialize the BigQuery database and tables"""
    print("🏗️ Setting up database...")
    
    try:
        bq_client = BigQueryClient()
        
        # Create dataset
        bq_client.create_dataset()
        
        # Create tables
        bq_client.execute_sql_file('sql/create_tables.sql')
        
        print("✅ Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def collect_data(limit=1000):
    """Collect data from various sources"""
    print(f"📥 Collecting {limit} data samples...")
    
    try:
        collector = DataCollector()
        count = collector.run_full_collection()
        
        print(f"✅ Successfully collected {count} content items!")
        return count
        
    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return 0

def analyze_content(content_text=None):
    """Analyze content for authenticity"""
    print("🔎 Analyzing content authenticity...")
    
    try:
        detector = AuthenticityDetector()
        
        if content_text:
            # Analyze provided text
            result = detector.process_content(content_text, "cli_input")
            
            print(f"\n📊 Analysis Results:")
            print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
            print(f"   Confidence Score: {result['confidence_score']:.3f}")
            print(f"   Explanation: {result['explanation']}")
            print(f"   Model Version: {result['model_version']}")
            
            # Show key features
            if result['features']:
                print(f"\n🔧 Key Features:")
                important_features = ['ai_phrase_density', 'repetition_score', 'formal_language_density']
                for feature in important_features:
                    if feature in result['features']:
                        print(f"   {feature}: {result['features'][feature]:.3f}")
        
        else:
            # Analyze recent content from database
            bq_client = BigQueryClient()
            content_data = bq_client.get_text_content(limit=10)
            
            if content_data.empty:
                print("❌ No content found in database. Run data collection first.")
                return
            
            results = []
            for _, row in content_data.iterrows():
                result = detector.process_content(row['content'], row['id'])
                results.append(result)
            
            # Store results in BigQuery
            bq_client.insert_authenticity_scores(results)
            
            print(f"✅ Analyzed {len(results)} content items from database")
            
            # Show summary statistics
            scores = [r['authenticity_score'] for r in results]
            print(f"\n📈 Summary Statistics:")
            print(f"   Average Authenticity Score: {sum(scores)/len(scores):.3f}")
            print(f"   Low Authenticity Items: {sum(1 for s in scores if s < 0.3)}")
            print(f"   High Authenticity Items: {sum(1 for s in scores if s > 0.7)}")
        
    except Exception as e:
        print(f"❌ Error analyzing content: {e}")

def detect_campaigns():
    """Detect coordinated campaigns"""
    print("📈 Detecting coordinated campaigns...")
    
    try:
        detector = CampaignDetector()
        campaigns = detector.detect_all_campaigns(limit=1000)
        
        if 'error' in campaigns:
            print(f"❌ {campaigns['error']}")
            return
        
        # Generate and display report
        report = detector.generate_comprehensive_report(campaigns)
        print(report)
        
        return campaigns
        
    except Exception as e:
        print(f"❌ Error detecting campaigns: {e}")
        return None

def analyze_image(image_path):
    """Analyze image for authenticity"""
    print(f"🖼️ Analyzing image: {image_path}")
    
    try:
        analyzer = ImageAnalyzer()
        result = analyzer.analyze_image_authenticity(image_path)
        
        print(f"\n📊 Image Analysis Results:")
        print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
        print(f"   Confidence Score: {result['confidence_score']:.3f}")
        print(f"   Explanation: {result['explanation']}")
        
        # Show key features
        if result['features']:
            print(f"\n🔧 Key Image Features:")
            important_features = ['noise_uniformity', 'pattern_repetition', 'edge_consistency']
            for feature in important_features:
                if feature in result['features']:
                    print(f"   {feature}: {result['features'][feature]:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return None

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching dashboard...")
    
    try:
        import subprocess
        dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
        subprocess.run(['streamlit', 'run', dashboard_path])
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

def main():
    print("🔧 Starting AI Content Authenticity Network...")
    print(f"🔧 Python path: {sys.path}")
    print(f"🔧 Current directory: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(
        description="AI Content Authenticity Network - Detect AI-generated content and coordinated campaigns"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Initialize the database and tables')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect data from various sources')
    collect_parser.add_argument('--limit', type=int, default=1000, 
                               help='Number of data samples to collect (default: 1000)')
    
    # Content analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze content for authenticity')
    analyze_parser.add_argument('--text', type=str, help='Text content to analyze')
    
    # Campaign detection command
    campaign_parser = subparsers.add_parser('campaigns', help='Detect coordinated campaigns')
    
    # Image analysis command
    image_parser = subparsers.add_parser('image', help='Analyze image for authenticity')
    image_parser.add_argument('path', help='Path to image file')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch the web dashboard')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the complete analysis pipeline')
    pipeline_parser.add_argument('--skip-setup', action='store_true', help='Skip database setup')
    pipeline_parser.add_argument('--skip-collect', action='store_true', help='Skip data collection')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                AI Content Authenticity Network                ║  
║                     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                      ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    if args.command == 'setup':
        setup_database()
    
    elif args.command == 'collect':
        collect_data(args.limit)
    
    elif args.command == 'analyze':
        analyze_content(args.text)
    
    elif args.command == 'campaigns':
        detect_campaigns()
    
    elif args.command == 'image':
        analyze_image(args.path)
    
    elif args.command == 'dashboard':
        run_dashboard()
    
    elif args.command == 'pipeline':
        print("🚀 Running complete analysis pipeline...")
        
        success = True
        
        # Setup database
        if not args.skip_setup:
            success = setup_database()
        
        # Collect data
        if success and not args.skip_collect:
            count = collect_data(1000)
            success = count > 0
        
        # Analyze content
        if success:
            analyze_content()
        
        # Detect campaigns
        if success:
            detect_campaigns()
        
        if success:
            print("\n✅ Complete analysis pipeline finished successfully!")
            print("🚀 Launch the dashboard with: python main.py dashboard")
        else:
            print("\n❌ Pipeline failed. Check the errors above.")

if __name__ == "__main__":
    main()