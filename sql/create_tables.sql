-- Create dataset
CREATE SCHEMA IF NOT EXISTS `{project_id}.authenticity_network`
OPTIONS(
  description="AI Content Authenticity Network Dataset",
  location="US"
);

-- Text content table
CREATE OR REPLACE TABLE `{project_id}.authenticity_network.text_content` (
  id STRING NOT NULL,
  content STRING NOT NULL,
  source STRING NOT NULL, -- 'human', 'ai_generated', 'unknown'
  source_platform STRING, -- 'hackernews', 'reddit', 'twitter', etc.
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  language STRING,
  word_count INT64,
  char_count INT64,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Image content table
CREATE OR REPLACE TABLE `{project_id}.authenticity_network.image_content` (
  id STRING NOT NULL,
  image_url STRING NOT NULL,
  source STRING NOT NULL, -- 'human', 'ai_generated', 'unknown'
  source_platform STRING,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  width INT64,
  height INT64,
  file_size INT64,
  format STRING,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Authenticity scores table
CREATE OR REPLACE TABLE `{project_id}.authenticity_network.authenticity_scores` (
  content_id STRING NOT NULL,
  content_type STRING NOT NULL, -- 'text', 'image'
  authenticity_score FLOAT64 NOT NULL,
  confidence_score FLOAT64 NOT NULL,
  model_version STRING NOT NULL,
  features JSON, -- extracted features used for scoring
  explanation STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Embeddings table
CREATE OR REPLACE TABLE `{project_id}.authenticity_network.embeddings` (
  content_id STRING NOT NULL,
  content_type STRING NOT NULL,
  embedding ARRAY<FLOAT64>,
  model_name STRING NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Campaigns table for coordinated content detection
CREATE OR REPLACE TABLE `{project_id}.authenticity_network.campaigns` (
  campaign_id STRING NOT NULL,
  content_ids ARRAY<STRING>,
  similarity_score FLOAT64 NOT NULL,
  campaign_size INT64 NOT NULL,
  detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  campaign_type STRING, -- 'text_similarity', 'image_similarity', 'temporal_pattern'
  metadata JSON
);

-- Create indexes for better performance
CREATE OR REPLACE TABLE `{project_id}.authenticity_network.content_similarity` (
  content_id_1 STRING NOT NULL,
  content_id_2 STRING NOT NULL,
  similarity_score FLOAT64 NOT NULL,
  similarity_type STRING NOT NULL, -- 'semantic', 'visual', 'temporal'
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
) CLUSTER BY content_id_1, content_id_2;