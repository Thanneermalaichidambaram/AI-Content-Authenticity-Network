import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import re
import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
from langdetect import detect
import json

from .config import Config

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class AuthenticityDetector:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        if not text:
            return {}
        
        features = {}
        
        # Basic metrics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
        
        # Language detection
        try:
            features['language'] = detect(text)
            features['is_english'] = 1.0 if features['language'] == 'en' else 0.0
        except:
            features['language'] = 'unknown'
            features['is_english'] = 0.0
        
        # Linguistic patterns
        features.update(self._extract_linguistic_features(text))
        
        # AI-specific patterns
        features.update(self._extract_ai_patterns(text))
        
        return features
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        if not text:
            return features
        
        # Punctuation patterns
        features['exclamation_ratio'] = text.count('!') / len(text)
        features['question_ratio'] = text.count('?') / len(text)
        features['period_ratio'] = text.count('.') / len(text)
        features['comma_ratio'] = text.count(',') / len(text)
        
        # Case patterns
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
        
        # Word length patterns
        words = text.split()
        if words:
            word_lengths = [len(word) for word in words]
            features['avg_word_length'] = np.mean(word_lengths)
            features['word_length_std'] = np.std(word_lengths) if len(word_lengths) > 1 else 0
        
        # Sentence length patterns
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
            if sentence_lengths:
                features['avg_sentence_length'] = np.mean(sentence_lengths)
                features['sentence_length_std'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # POS tagging if spacy is available
        if self.nlp:
            try:
                doc = self.nlp(text[:1000])  # Limit for performance
                pos_counts = {}
                for token in doc:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                
                total_tokens = len(doc)
                if total_tokens > 0:
                    features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
                    features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
                    features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
                    features['adv_ratio'] = pos_counts.get('ADV', 0) / total_tokens
            except:
                pass
        
        return features
    
    def _extract_ai_patterns(self, text: str) -> Dict[str, float]:
        features = {}
        
        # Common AI phrases
        ai_phrases = [
            "as an ai", "i'm an ai", "as a language model",
            "i don't have personal", "i can't browse", "i don't have access",
            "as of my last update", "i'm sorry, but i", "i understand you're looking",
            "it's important to note", "please note that", "i'd be happy to help",
            "furthermore", "moreover", "additionally", "in conclusion"
        ]
        
        text_lower = text.lower()
        ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text_lower)
        features['ai_phrase_density'] = ai_phrase_count / len(text.split()) if text.split() else 0
        
        # Repetitive patterns
        features['repetition_score'] = self._calculate_repetition_score(text)
        
        # Formal language indicators
        formal_words = ['utilize', 'facilitate', 'implement', 'furthermore', 'therefore', 'however']
        formal_count = sum(1 for word in formal_words if word in text_lower)
        features['formal_language_density'] = formal_count / len(text.split()) if text.split() else 0
        
        # Perfect grammar indicators (less natural)
        features['perfect_punctuation'] = 1.0 if self._has_perfect_punctuation(text) else 0.0
        
        return features
    
    def _calculate_repetition_score(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only consider longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return 0.0
        
        # Calculate repetition score
        max_freq = max(word_freq.values())
        return max_freq / len(words)
    
    def _has_perfect_punctuation(self, text: str) -> bool:
        # Check for overly perfect punctuation patterns
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return False
        
        # Check if every sentence starts with capital letter
        proper_caps = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[0].isupper():
                proper_caps += 1
        
        return proper_caps / len(sentences) > 0.95
    
    def calculate_authenticity_score(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        # Simple rule-based authenticity scoring
        # This would be replaced with a trained ML model in production
        
        score = 0.5  # Start neutral
        confidence = 0.5
        explanation_parts = []
        
        # AI phrase detection
        if features.get('ai_phrase_density', 0) > 0.01:
            score -= 0.3
            confidence += 0.2
            explanation_parts.append("Contains AI-specific phrases")
        
        # Repetition patterns
        repetition = features.get('repetition_score', 0)
        if repetition > 0.1:
            score -= 0.2
            confidence += 0.1
            explanation_parts.append("High repetition patterns")
        
        # Perfect punctuation (unnatural)
        if features.get('perfect_punctuation', 0) > 0.5:
            score -= 0.15
            confidence += 0.1
            explanation_parts.append("Overly perfect punctuation")
        
        # Formal language density
        formal_density = features.get('formal_language_density', 0)
        if formal_density > 0.05:
            score -= 0.1
            confidence += 0.05
            explanation_parts.append("High formal language density")
        
        # Reading ease (AI tends to be more readable)
        reading_ease = features.get('flesch_reading_ease', 50)
        if reading_ease > 70:  # Very easy to read
            score -= 0.1
        elif reading_ease < 30:  # Very hard to read (more human-like)
            score += 0.1
        
        # Natural variation in sentence length
        sentence_std = features.get('sentence_length_std', 0)
        if sentence_std > 5:  # High variation is more human-like
            score += 0.1
            explanation_parts.append("Natural sentence length variation")
        
        # Clamp values
        score = max(0.0, min(1.0, score))
        confidence = max(0.1, min(1.0, confidence))
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "No clear indicators"
        
        return score, confidence, explanation
    
    def process_content(self, content: str, content_id: str) -> Dict[str, Any]:
        features = self.extract_text_features(content)
        authenticity_score, confidence, explanation = self.calculate_authenticity_score(features)
        
        return {
            'content_id': content_id,
            'content_type': 'text',
            'authenticity_score': authenticity_score,
            'confidence_score': confidence,
            'features': features,
            'explanation': explanation,
            'model_version': 'rule_based_v1.0'
        }
    
    def batch_process(self, contents: List[str], content_ids: List[str]) -> List[Dict[str, Any]]:
        results = []
        for content, content_id in zip(contents, content_ids):
            result = self.process_content(content, content_id)
            results.append(result)
        return results