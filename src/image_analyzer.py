import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageStat
import requests
from typing import Dict, List, Any, Tuple, Optional
import io
import hashlib
import uuid
from datetime import datetime
import base64

from .config import Config

class ImageAnalyzer:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def analyze_image_authenticity(self, image_path_or_url: str) -> Dict[str, Any]:
        """Analyze an image for authenticity markers"""
        try:
            image = self._load_image(image_path_or_url)
            if image is None:
                return self._create_error_result("Failed to load image")
            
            features = self._extract_image_features(image)
            authenticity_score, confidence, explanation = self._calculate_image_authenticity(features)
            
            return {
                'content_id': f"img_{hashlib.md5(str(image_path_or_url).encode()).hexdigest()[:8]}",
                'content_type': 'image',
                'authenticity_score': authenticity_score,
                'confidence_score': confidence,
                'features': features,
                'explanation': explanation,
                'model_version': 'image_analyzer_v1.0'
            }
            
        except Exception as e:
            return self._create_error_result(f"Error analyzing image: {str(e)}")
    
    def _load_image(self, image_path_or_url: str) -> Optional[np.ndarray]:
        """Load image from local path or URL"""
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url, stream=True)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_path_or_url)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features that may indicate AI generation"""
        features = {}
        
        # Basic image properties
        height, width, channels = image.shape
        features['width'] = float(width)
        features['height'] = float(height)
        features['aspect_ratio'] = width / height
        features['total_pixels'] = float(width * height)
        
        # Color distribution analysis
        features.update(self._analyze_color_distribution(image))
        
        # Texture and pattern analysis
        features.update(self._analyze_texture_patterns(image))
        
        # Compression artifacts
        features.update(self._analyze_compression_artifacts(image))
        
        # Noise analysis
        features.update(self._analyze_noise_patterns(image))
        
        # Edge and frequency analysis
        features.update(self._analyze_edges_and_frequency(image))
        
        return features
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution characteristics"""
        features = {}
        
        # Convert to PIL for easier analysis
        pil_image = Image.fromarray(image)
        
        # Color statistics
        stat = ImageStat.Stat(pil_image)
        features['mean_brightness'] = np.mean(stat.mean)
        features['brightness_stddev'] = np.mean(stat.stddev)
        
        # Color histogram analysis
        hist_r, _ = np.histogram(image[:,:,0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(image[:,:,1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(image[:,:,2], bins=256, range=(0, 256))
        
        # Color distribution uniformity
        features['red_uniformity'] = np.std(hist_r) / np.mean(hist_r) if np.mean(hist_r) > 0 else 0
        features['green_uniformity'] = np.std(hist_g) / np.mean(hist_g) if np.mean(hist_g) > 0 else 0
        features['blue_uniformity'] = np.std(hist_b) / np.mean(hist_b) if np.mean(hist_b) > 0 else 0
        
        # Saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features['mean_saturation'] = np.mean(hsv[:,:,1])
        features['saturation_stddev'] = np.std(hsv[:,:,1])
        
        return features
    
    def _analyze_texture_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture patterns that might indicate AI generation"""
        features = {}
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern-like analysis
        features['texture_variance'] = float(np.var(gray))
        features['texture_mean'] = float(np.mean(gray))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Pattern repetition (simplified)
        features['pattern_repetition'] = self._calculate_pattern_repetition(gray)
        
        return features
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze compression artifacts"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # DCT analysis for JPEG-like artifacts
        dct = cv2.dct(np.float32(gray))
        features['dct_energy'] = float(np.sum(np.abs(dct)))
        features['high_freq_energy'] = float(np.sum(np.abs(dct[32:, 32:]))) if dct.shape[0] > 32 and dct.shape[1] > 32 else 0
        
        # Blocking artifacts detection
        features['blocking_artifacts'] = self._detect_blocking_artifacts(gray)
        
        return features
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze noise patterns"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Noise estimation using Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['noise_estimate'] = float(laplacian_var)
        
        # Uniform noise vs natural noise
        features['noise_uniformity'] = self._calculate_noise_uniformity(gray)
        
        return features
    
    def _analyze_edges_and_frequency(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze edge characteristics and frequency domain"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features['edge_strength'] = float(np.mean(sobel_magnitude))
        features['edge_consistency'] = float(np.std(sobel_magnitude))
        
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        features['frequency_spread'] = float(np.std(magnitude_spectrum))
        features['high_frequency_content'] = self._calculate_high_freq_content(magnitude_spectrum)
        
        return features
    
    def _calculate_pattern_repetition(self, gray: np.ndarray) -> float:
        """Calculate pattern repetition score"""
        try:
            # Simple pattern repetition using template matching
            h, w = gray.shape
            if h < 32 or w < 32:
                return 0.0
            
            # Take a small template from the center
            template = gray[h//2-8:h//2+8, w//2-8:w//2+8]
            
            # Match template across the image
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Count high correlation areas
            high_corr = np.sum(result > 0.8)
            return float(high_corr) / result.size
            
        except Exception:
            return 0.0
    
    def _detect_blocking_artifacts(self, gray: np.ndarray) -> float:
        """Detect JPEG-like blocking artifacts"""
        try:
            h, w = gray.shape
            blocking_score = 0.0
            
            # Check for 8x8 block boundaries (JPEG standard)
            for i in range(8, h-8, 8):
                diff = np.abs(np.mean(gray[i-1, :]) - np.mean(gray[i, :]))
                blocking_score += diff
            
            for j in range(8, w-8, 8):
                diff = np.abs(np.mean(gray[:, j-1]) - np.mean(gray[:, j]))
                blocking_score += diff
            
            return blocking_score / (h * w)
            
        except Exception:
            return 0.0
    
    def _calculate_noise_uniformity(self, gray: np.ndarray) -> float:
        """Calculate noise uniformity across the image"""
        try:
            # Divide image into blocks and calculate noise in each
            h, w = gray.shape
            block_size = 32
            noise_levels = []
            
            for i in range(0, h-block_size, block_size):
                for j in range(0, w-block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_noise = cv2.Laplacian(block, cv2.CV_64F).var()
                    noise_levels.append(block_noise)
            
            if len(noise_levels) < 2:
                return 0.0
            
            return float(np.std(noise_levels) / np.mean(noise_levels)) if np.mean(noise_levels) > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_high_freq_content(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate high frequency content ratio"""
        try:
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Define high frequency region (outer portion)
            mask = np.zeros((h, w))
            cv2.circle(mask, (center_w, center_h), min(center_h, center_w) // 3, 1, -1)
            
            total_energy = np.sum(magnitude_spectrum)
            high_freq_energy = np.sum(magnitude_spectrum * (1 - mask))
            
            return float(high_freq_energy / total_energy) if total_energy > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_image_authenticity(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """Calculate authenticity score based on extracted features"""
        score = 0.5  # Start neutral
        confidence = 0.5
        explanation_parts = []
        
        # AI-generated images often have:
        # 1. Very uniform noise patterns
        noise_uniformity = features.get('noise_uniformity', 0.5)
        if noise_uniformity < 0.1:  # Very uniform noise
            score -= 0.2
            confidence += 0.1
            explanation_parts.append("Uniform noise pattern")
        
        # 2. Unnatural saturation levels
        saturation_std = features.get('saturation_stddev', 50)
        if saturation_std < 20:  # Very consistent saturation
            score -= 0.15
            confidence += 0.1
            explanation_parts.append("Consistent saturation levels")
        
        # 3. Lack of compression artifacts (if supposed to be real photo)
        blocking_artifacts = features.get('blocking_artifacts', 0.1)
        if blocking_artifacts < 0.01:  # Too clean
            score -= 0.1
            explanation_parts.append("Lack of natural compression artifacts")
        
        # 4. Unnatural edge consistency
        edge_consistency = features.get('edge_consistency', 50)
        if edge_consistency < 10:  # Very consistent edges
            score -= 0.1
            explanation_parts.append("Unnaturally consistent edges")
        
        # 5. Pattern repetition
        pattern_repetition = features.get('pattern_repetition', 0)
        if pattern_repetition > 0.1:
            score -= 0.15
            confidence += 0.1
            explanation_parts.append("Repetitive patterns detected")
        
        # Positive indicators for authentic images
        # Natural noise variation
        if noise_uniformity > 0.3:
            score += 0.1
            explanation_parts.append("Natural noise variation")
        
        # Natural compression artifacts
        if 0.01 < blocking_artifacts < 0.1:
            score += 0.05
            explanation_parts.append("Natural compression artifacts")
        
        # Clamp values
        score = max(0.0, min(1.0, score))
        confidence = max(0.1, min(1.0, confidence))
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "No clear indicators"
        
        return score, confidence, explanation
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'content_id': f"error_{uuid.uuid4().hex[:8]}",
            'content_type': 'image',
            'authenticity_score': 0.0,
            'confidence_score': 0.0,
            'features': {},
            'explanation': error_message,
            'model_version': 'image_analyzer_v1.0'
        }
    
    def batch_analyze_images(self, image_paths_or_urls: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple images for authenticity"""
        results = []
        for image_path in image_paths_or_urls:
            result = self.analyze_image_authenticity(image_path)
            results.append(result)
        return results