-- Create text embedding model using Vertex AI
CREATE OR REPLACE MODEL `{project_id}.authenticity_network.text_embedding_model`
REMOTE WITH CONNECTION `{project_id}.us.vertex-ai-connection`
OPTIONS (
  ENDPOINT = 'textembedding-gecko@001'
);

-- Create authenticity classification model
CREATE OR REPLACE MODEL `{project_id}.authenticity_network.authenticity_classifier`
OPTIONS(
  MODEL_TYPE='LOGISTIC_REG',
  INPUT_LABEL_COLS=['is_authentic'],
  AUTO_CLASS_WEIGHTS=TRUE
) AS
SELECT
  * EXCEPT(id, content, source, created_at)
FROM (
  SELECT 
    tc.id,
    tc.content,
    tc.source,
    CASE WHEN tc.source = 'human' THEN TRUE ELSE FALSE END as is_authentic,
    tc.word_count,
    tc.char_count,
    -- Text features
    CHAR_LENGTH(tc.content) / tc.word_count as avg_word_length,
    (CHAR_LENGTH(tc.content) - CHAR_LENGTH(REPLACE(tc.content, ' ', ''))) / CHAR_LENGTH(tc.content) as space_ratio,
    (CHAR_LENGTH(tc.content) - CHAR_LENGTH(REPLACE(tc.content, '.', ''))) / CHAR_LENGTH(tc.content) as period_ratio,
    (CHAR_LENGTH(tc.content) - CHAR_LENGTH(REPLACE(tc.content, ',', ''))) / CHAR_LENGTH(tc.content) as comma_ratio,
    tc.created_at
  FROM `{project_id}.authenticity_network.text_content` tc
  WHERE tc.source IN ('human', 'ai_generated')
);

-- Create similarity detection model
CREATE OR REPLACE MODEL `{project_id}.authenticity_network.similarity_detector`
OPTIONS(
  MODEL_TYPE='KMEANS',
  NUM_CLUSTERS=50,
  STANDARDIZE_FEATURES=TRUE
) AS
SELECT
  * EXCEPT(content_id, created_at)
FROM `{project_id}.authenticity_network.embeddings`
WHERE content_type = 'text';

-- Function to calculate semantic similarity
CREATE OR REPLACE FUNCTION `{project_id}.authenticity_network.cosine_similarity`(
  vec1 ARRAY<FLOAT64>, 
  vec2 ARRAY<FLOAT64>
) 
RETURNS FLOAT64
LANGUAGE js AS """
  if (vec1.length !== vec2.length) return null;
  
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  
  if (norm1 === 0 || norm2 === 0) return 0;
  
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
""";

-- Stored procedure to detect campaigns
CREATE OR REPLACE PROCEDURE `{project_id}.authenticity_network.detect_campaigns`()
BEGIN
  -- Find similar content pairs
  CREATE OR REPLACE TEMP TABLE similar_pairs AS
  SELECT 
    e1.content_id as content_id_1,
    e2.content_id as content_id_2,
    `{project_id}.authenticity_network.cosine_similarity`(e1.embedding, e2.embedding) as similarity
  FROM `{project_id}.authenticity_network.embeddings` e1
  CROSS JOIN `{project_id}.authenticity_network.embeddings` e2
  WHERE e1.content_id < e2.content_id
    AND e1.content_type = e2.content_type
    AND `{project_id}.authenticity_network.cosine_similarity`(e1.embedding, e2.embedding) > 0.85;
  
  -- Insert detected campaigns
  INSERT INTO `{project_id}.authenticity_network.campaigns` (
    campaign_id, content_ids, similarity_score, campaign_size, campaign_type
  )
  SELECT
    GENERATE_UUID() as campaign_id,
    ARRAY_AGG(DISTINCT content_id) as content_ids,
    AVG(similarity) as similarity_score,
    COUNT(DISTINCT content_id) as campaign_size,
    'text_similarity' as campaign_type
  FROM (
    SELECT content_id_1 as content_id, similarity FROM similar_pairs
    UNION ALL
    SELECT content_id_2 as content_id, similarity FROM similar_pairs
  )
  GROUP BY similarity
  HAVING COUNT(DISTINCT content_id) >= 5;
END;