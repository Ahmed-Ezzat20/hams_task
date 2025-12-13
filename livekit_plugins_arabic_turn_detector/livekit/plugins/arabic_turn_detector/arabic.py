"""
Arabic Turn Detector for LiveKit Agents
========================================

This module provides an Arabic-specific End-of-Utterance (EOU) detection model
for LiveKit agents, optimized for Arabic language conversations.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from typing import Any

import numpy as np
import onnxruntime as ort
from livekit.agents.inference_runner import _InferenceRunner

try:
    from arabert.preprocess import ArabertPreprocessor
    ARABERT_AVAILABLE = True
except ImportError:
    ARABERT_AVAILABLE = False
    logging.warning(
        "arabert not installed. Install with: pip install arabert\n"
        "Falling back to basic Arabic normalization."
    )


logger = logging.getLogger(__name__)


class _EOURunnerAr(_InferenceRunner):
    """
    Arabic End-of-Utterance Runner
    
    Implements EOU detection for Arabic language using a transformer-based model.
    """
    
    INFERENCE_METHOD = "arabic_end_of_utterance"
    
    # Model configuration
    HG_MODEL = "Ahmed-Ezzat20/arabic-eou-detector"  # TODO: Update with actual HuggingFace model ID
    ONNX_FILENAME = "eou_model_quantized.onnx"
    MODEL_REVISION = "main"
    
    # Context limits
    MAX_HISTORY_TOKENS = 512
    MAX_HISTORY_TURNS = 6
    
    def __init__(
        self,
        *,
        model_path: str | None = None,
        unlikely_threshold: float = 0.7,
    ):
        """
        Initialize Arabic EOU Runner
        
        Args:
            model_path: Path to local ONNX model (optional, will download from HF if not provided)
            unlikely_threshold: Confidence threshold for EOU detection (default: 0.7)
        """
        super().__init__()
        
        self.model_path = model_path
        self.unlikely_threshold = unlikely_threshold
        
        self._session: ort.InferenceSession | None = None
        self._tokenizer = None
        self._preprocessor = None
        
        logger.info(f"Initialized Arabic EOU Runner (threshold={unlikely_threshold})")
    
    @classmethod
    def model_type(cls) -> str:
        """Return model type identifier"""
        return "ar"
    
    @classmethod
    def model_revision(cls) -> str:
        """Return model revision"""
        return cls.MODEL_REVISION
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text using AraBERT preprocessing
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        if ARABERT_AVAILABLE and self._preprocessor is not None:
            # Use AraBERT preprocessor
            return self._preprocessor.preprocess(text)
        else:
            # Fallback: Basic Arabic normalization
            return self._basic_arabic_normalization(text)
    
    def _basic_arabic_normalization(self, text: str) -> str:
        """
        Basic Arabic text normalization (fallback when arabert not available)
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized text
        """
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Normalize Arabic characters
        # Normalize Alef variations
        text = re.sub("[إأآا]", "ا", text)
        # Normalize Yeh variations
        text = re.sub("ى", "ي", text)
        # Normalize Teh Marbuta
        text = re.sub("ة", "ه", text)
        # Remove diacritics (tashkeel)
        text = re.sub("[\u064B-\u065F\u0670]", "", text)
        # Remove tatweel
        text = re.sub("\u0640", "", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def _format_chat_ctx(self, chat_ctx: list[dict[str, Any]]) -> str:
        """
        Format chat context for model input
        
        Args:
            chat_ctx: List of chat messages with 'role' and 'content' keys
            
        Returns:
            Formatted text for model input
        """
        new_chat_ctx = []
        last_msg: dict[str, Any] | None = None
        
        for msg in chat_ctx:
            if not msg.get("content"):
                continue
            
            content = self._normalize_text(msg["content"])
            
            # Combine adjacent turns from same role
            if last_msg and last_msg["role"] == msg["role"]:
                last_msg["content"] += f" {content}"
            else:
                msg_copy = msg.copy()
                msg_copy["content"] = content
                new_chat_ctx.append(msg_copy)
                last_msg = msg_copy
        
        # Apply chat template if tokenizer available
        if self._tokenizer is not None:
            try:
                convo_text = self._tokenizer.apply_chat_template(
                    new_chat_ctx,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                    tokenize=False
                )
                
                # Remove EOU token from current utterance if present
                eou_token = "<|im_end|>"
                if eou_token in convo_text:
                    ix = convo_text.rfind(eou_token)
                    text = convo_text[:ix]
                else:
                    text = convo_text
                
                return text
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
        
        # Fallback: Simple concatenation
        return " ".join(msg["content"] for msg in new_chat_ctx)
    
    def initialize(self) -> None:
        """Initialize the model and tokenizer"""
        logger.info("Initializing Arabic EOU model...")
        
        try:
            # Initialize AraBERT preprocessor
            if ARABERT_AVAILABLE:
                self._preprocessor = ArabertPreprocessor(
                    model_name="aubmindlab/bert-base-arabertv2"
                )
                logger.info("✓ AraBERT preprocessor initialized")
            
            # Load ONNX model
            if self.model_path:
                model_path = self.model_path
                logger.info(f"Loading model from local path: {model_path}")
            else:
                # TODO: Implement HuggingFace Hub download
                raise RuntimeError(
                    "Model path not provided. Please specify model_path parameter.\n"
                    "Example: ArabicTurnDetector(model_path='./models/eou_model_quantized.onnx')"
                )
            
            # Create ONNX Runtime session
            sess_options = ort.SessionOptions()
            
            # Optimize for CPU
            try:
                import psutil
                cpu_count = psutil.cpu_count(logical=False) or 4
                sess_options.intra_op_num_threads = max(1, min(math.ceil(cpu_count / 2), 4))
            except ImportError:
                sess_options.intra_op_num_threads = 2
            
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.dynamic_block_base", "4")
            
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
            
            logger.info("✓ ONNX model loaded")
            
            # Load tokenizer
            try:
                from transformers import AutoTokenizer
                
                tokenizer_path = self.model_path.replace(".onnx", "").replace("_quantized", "")
                if tokenizer_path.endswith("/"):
                    tokenizer_path = tokenizer_path[:-1]
                
                # Try to load from same directory as model
                import os
                model_dir = os.path.dirname(self.model_path)
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    truncation_side="left",
                )
                logger.info("✓ Tokenizer loaded")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                logger.warning("Continuing without tokenizer (will use fallback formatting)")
            
            logger.info("✓ Arabic EOU model initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Arabic EOU model: {e}")
            raise RuntimeError(
                f"Arabic EOU model initialization failed: {e}\n"
                "Make sure you have:\n"
                "1. Converted your model to ONNX format\n"
                "2. Provided correct model_path\n"
                "3. Installed required dependencies: pip install onnxruntime transformers arabert"
            )
    
    def run(self, data: bytes) -> bytes | None:
        """
        Run EOU inference
        
        Args:
            data: JSON-encoded input data with 'chat_ctx' key
            
        Returns:
            JSON-encoded prediction result with eou_probability, duration, input, is_eou, threshold
        """
        import time
        
        try:
            # Parse input
            data_json = json.loads(data)
            chat_ctx = data_json.get("chat_ctx", [])
            
            if not chat_ctx:
                return json.dumps({
                    "eou_probability": 0.0,
                    "is_eou": False,
                    "threshold": self.unlikely_threshold,
                    "duration": 0.0,
                    "input": ""
                }).encode()
            
            # Start timing
            start_time = time.perf_counter()
            
            # Format chat context
            text = self._format_chat_ctx(chat_ctx)
            
            if not text:
                return json.dumps({
                    "eou_probability": 0.0,
                    "is_eou": False,
                    "threshold": self.unlikely_threshold,
                    "duration": 0.0,
                    "input": ""
                }).encode()
            
            # Tokenize
            if self._tokenizer is None:
                logger.error("Tokenizer not available")
                return json.dumps({
                    "eou_probability": 0.0,
                    "is_eou": False,
                    "threshold": self.unlikely_threshold,
                    "duration": 0.0,
                    "input": text[:100]
                }).encode()
            
            inputs = self._tokenizer(
                text,
                padding="max_length",
                max_length=self.MAX_HISTORY_TOKENS,
                truncation=True,
                return_tensors="np"
            )
            
            # Prepare ONNX inputs
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            # Run inference
            outputs = self._session.run(None, onnx_inputs)
            
            # Get prediction
            logits = outputs[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            
            # EOU is class 1
            eou_probability = float(probs[0, 1])
            is_eou = eou_probability > self.unlikely_threshold
            
            # End timing
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Truncate input for logging
            input_text = text[:100] if len(text) > 100 else text
            
            result = {
                "eou_probability": eou_probability,
                "is_eou": is_eou,
                "threshold": self.unlikely_threshold,
                "duration": round(duration, 3),
                "input": input_text
            }
            
            # Log with same format as LiveKit's EOUModel
            logger.debug(
                "eou prediction",
                extra=result
            )
            
            return json.dumps(result).encode()
            
        except Exception as e:
            logger.error(f"Error during EOU inference: {e}")
            return json.dumps({
                "eou_probability": 0.0,
                "is_eou": False,
                "threshold": self.unlikely_threshold,
                "duration": 0.0,
                "input": "",
                "error": str(e)
            }).encode()


class ArabicTurnDetector:
    """
    Arabic Turn Detector for LiveKit Agents
    
    Detects end-of-utterance in Arabic conversations using a transformer-based model.
    
    Example:
        ```python
        from livekit.plugins.arabic_turn_detector import ArabicTurnDetector
        
        # Create turn detector
        turn_detector = ArabicTurnDetector(
            model_path="./models/eou_model_quantized.onnx",
            unlikely_threshold=0.7
        )
        
        # Use in agent
        session = AgentSession(
            turn_detector=turn_detector,
            # ... other configurations
        )
        ```
    """
    
    def __init__(
        self,
        *,
        model_path: str,
        unlikely_threshold: float = 0.7,
    ):
        """
        Initialize Arabic Turn Detector
        
        Args:
            model_path: Path to ONNX model file
            unlikely_threshold: Confidence threshold for EOU detection (default: 0.7)
                               Lower values = more sensitive (more interruptions)
                               Higher values = less sensitive (fewer interruptions)
        """
        self._runner = _EOURunnerAr(
            model_path=model_path,
            unlikely_threshold=unlikely_threshold
        )
        
        # Store threshold for protocol methods
        self._unlikely_threshold = unlikely_threshold
        
        # Initialize the model
        self._runner.initialize()

        # Expose metadata attributes expected by LiveKit
        # Name of the underlying model (for logging / analytics)
        self.model: str = _EOURunnerAr.HG_MODEL
        # Provider string to indicate a custom third-party detector
        self.provider: str = "arabic_turn_detector"
    
    async def unlikely_threshold(self, language: str | None = None) -> float | None:
        """
        Return the threshold below which EOU is unlikely for the given language.
        Returns None if the language is not supported.
        
        This method is required by the LiveKit agents framework Protocol.
        
        Args:
            language: Language code (e.g., "ar", "ara", "ar-SA") or None
            
        Returns:
            Threshold value for supported languages (Arabic), None otherwise
        """
        if language is None:
            # Default to Arabic support
            return self._unlikely_threshold
        
        # Handle language codes (e.g., "ara", "ar", "ar-SA")
        lang_code = language.lower().split("-")[0]
        
        # Check if it's Arabic
        if lang_code in ("ar", "ara", "arabic"):
            return self._unlikely_threshold
        
        # Language not supported
        return None
    
    async def supports_language(self, language: str | None = None) -> bool:
        """
        Asynchronously check if this turn detector supports the given language.

        This method is required by the LiveKit Agents framework Protocol.

        Args:
            language: Language code (e.g., "ar", "ara", "ar-SA") or None

        Returns:
            True if the language is supported (Arabic), False otherwise
        """
        return (await self.unlikely_threshold(language)) is not None
    
    async def detect_turn(self, chat_ctx: list[dict[str, Any]]) -> bool:
        """
        Detect if the current utterance is complete
        
        Args:
            chat_ctx: List of chat messages with 'role' and 'content' keys
            
        Returns:
            True if end-of-utterance detected, False otherwise
        """
        import time
        start_time = time.time()
        
        # Prepare input
        input_data = json.dumps({"chat_ctx": chat_ctx}).encode()
        
        # Format input for logging
        formatted_input = self._runner._format_chat_ctx(chat_ctx)
        
        # Run inference
        result_data = self._runner.run(input_data)
        
        if result_data is None:
            return False
        
        # Parse result
        result = json.loads(result_data)
        is_eou = result.get("is_eou", False)
        confidence = result.get("confidence", 0.0)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log EOU prediction (matching English turn detector format)
        logger.debug(
            "eou prediction",
            extra={
                "eou_probability": confidence,
                "duration": round(duration, 3),
                "input": formatted_input[:100],  # Truncate for readability
                "is_eou": is_eou,
                "threshold": self._unlikely_threshold
            }
        )
        
        return is_eou
    
    async def predict_end_of_turn(self, chat_ctx: list[dict[str, Any]]) -> float:
        """
        Asynchronously predict the probability that the current turn has ended.

        Args:
            chat_ctx: List of chat messages with 'role' and 'content' keys.

        Returns:
            Probability (0.0–1.0) that the latest user utterance has ended.
        """
        return self.get_confidence(chat_ctx)

    def get_confidence(self, chat_ctx: list[dict[str, Any]]) -> float:
        """
        Get EOU confidence score
        
        Args:
            chat_ctx: List of chat messages with 'role' and 'content' keys or a LiveKit
                ChatContext-like object.
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Convert LiveKit ChatContext → list[dict] if necessary
        if not isinstance(chat_ctx, list):
            try:
                msgs = getattr(chat_ctx, "messages", [])
                chat_ctx = [
                    {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                    for m in msgs
                    if getattr(m, "content", None)
                ]
            except Exception as e:
                logger.error(f"Failed to serialise chat_ctx: {e}")
                return 0.0

        # Prepare input
        input_data = json.dumps({"chat_ctx": chat_ctx}, ensure_ascii=False).encode()
        
        # Run inference
        result_data = self._runner.run(input_data)
        
        if result_data is None:
            return 0.0
        
        # Parse result
        result = json.loads(result_data)
        return result.get("confidence", 0.0)
