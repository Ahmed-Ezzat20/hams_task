"""
Arabic End-of-Utterance Detector Plugin
========================================

Simple, production-ready Arabic EOU detector using ONNX for fast inference.

Features:
- ONNX inference (2-3x faster than PyTorch)
- Quantized model support (75% smaller)
- Simple API (single file, easy to use)
- LiveKit Protocol compatible
- Structured logging

Usage:
    from arabic_turn_detector_plugin import ArabicEOUDetector
    
    detector = ArabicEOUDetector(
        model_path="./eou_model/models/eou_model_quantized.onnx",
        confidence_threshold=0.7
    )
    
    # In AgentSession:
    session = AgentSession(
        ...
        turn_detection=detector,
        ...
    )

Author: Based on MrEzzat/arabic-eou-detector
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ArabicEOUDetector:
    """
    Arabic End-of-Utterance Detector
    
    Uses ONNX for fast, efficient inference on CPU.
    Compatible with LiveKit agents framework.
    """
    
    # Model metadata
    model = "MrEzzat/arabic-eou-detector"
    provider = "huggingface"
    
    def __init__(
        self,
        model_path: str = "./eou_model/models/eou_model_quantized.onnx",
        tokenizer_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize Arabic EOU Detector
        
        Args:
            model_path: Path to ONNX model file (quantized recommended)
            tokenizer_path: Path to tokenizer directory (auto-detected if None)
            confidence_threshold: Threshold for EOU detection (0.0-1.0)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect tokenizer path
        if tokenizer_path is None:
            # Try: ./eou_model/models/eou_model/
            tokenizer_path = self.model_path.parent / "eou_model"
            if not tokenizer_path.exists():
                # Try: ./eou_model/models/
                tokenizer_path = self.model_path.parent
        
        self.tokenizer_path = Path(tokenizer_path)
        
        # Model components (lazy loaded)
        self._session = None
        self._tokenizer = None
        self._loaded = False
        
        logger.info(
            f"Initialized Arabic EOU Detector "
            f"(model={self.model_path.name}, threshold={confidence_threshold})"
        )
    
    def _load_model(self):
        """Load ONNX model and tokenizer (called on first use)"""
        if self._loaded:
            return
        
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            logger.info(f"Loading Arabic EOU detector model: {self.model_path}")
            
            # Check model exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}\n"
                    "Please ensure the ONNX model is in the correct location."
                )
            
            # Load ONNX model (CPU optimized)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1
            
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path),
                local_files_only=True
            )
            
            logger.info(f"✓ Successfully loaded {self.model_path.name}")
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def __call__(self, transcript) -> float:
        """
        Main method for LiveKit turn detection
        
        Args:
            transcript: ChatContext or string containing conversation text
            
        Returns:
            EOU probability (0.0-1.0)
        """
        # Ensure model is loaded
        self._load_model()
        
        # Extract text from ChatContext or use as string
        if hasattr(transcript, "text"):
            text = transcript.text
        else:
            text = str(transcript)
        
        if not text or not text.strip():
            return 0.0
        
        try:
            start_time = time.perf_counter()
            
            # Truncate to last 256 characters (or tokens)
            truncated_text = text[-256:] if len(text) > 256 else text
            
            # Tokenize
            inputs = self._tokenizer(
                truncated_text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="np"
            )
            
            # Prepare ONNX inputs
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            # Run ONNX inference (async to avoid blocking)
            outputs = await asyncio.to_thread(
                self._session.run, None, onnx_inputs
            )
            
            # Get prediction
            logits = outputs[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            
            # EOU is class 1 (LABEL_1)
            eou_probability = float(probs[0, 1])
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Structured logging (LiveKit pattern)
            result = {
                "eou_probability": round(eou_probability, 4),
                "is_eou": eou_probability > self.confidence_threshold,
                "threshold": self.confidence_threshold,
                "duration": round(duration, 3),
                "input": truncated_text[:100]
            }
            
            logger.debug("eou prediction", extra=result)
            
            # Return probability
            return eou_probability
            
        except Exception as e:
            logger.error(f"Error in EOU detection: {e}")
            return 0.0
    
    async def predict_end_of_turn(self, transcript) -> float:
        """Required by LiveKit 1.3.6+"""
        return await self.__call__(transcript)
    
    def supports_language(self, language: str | None) -> bool:
        """Check if detector supports the language"""
        if language is None:
            return True  # Default to Arabic
        return language.lower() in ["ar", "arabic", "ara", "ar-sa", "ar-eg"]
    
    def unlikely_threshold(self, language: str | None = None) -> float | None:
        """Return the unlikely threshold for EOU detection"""
        if self.supports_language(language):
            return self.confidence_threshold
        return None
    
    async def analyze_turn(self, transcript) -> float:
        """Alias for __call__ (compatibility)"""
        return await self.__call__(transcript)
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold"""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
        logger.info(f"Updated threshold to {threshold}")
    
    def get_model_info(self) -> dict:
        """Get model information and performance metrics"""
        return {
            "model_id": self.model,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "performance": {
                "accuracy": 0.90,
                "f1_score": 0.92,
                "precision": 0.90,
                "recall": 0.93,
            },
        }
