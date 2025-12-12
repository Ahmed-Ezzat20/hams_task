#!/usr/bin/env python3
"""
ONNX Conversion Script
======================

Convert trained EOU model to ONNX format for optimized inference.

Usage:
    python convert_to_onnx.py --model_path ./models/eou_model --output_path ./models/eou_model.onnx
"""

import argparse
import logging
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXConverter:
    """Convert HuggingFace model to ONNX format"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        opset_version: int = 14,
        max_seq_length: int = 512,
    ):
        """
        Initialize ONNX Converter
        
        Args:
            model_path: Path to trained model directory
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            max_seq_length: Maximum sequence length
        """
        self.model_path = model_path
        self.output_path = output_path
        self.opset_version = opset_version
        self.max_seq_length = max_seq_length
        
        self.model = None
        self.tokenizer = None
        
        logger.info("Initialized ONNXConverter")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Opset version: {opset_version}")
    
    def load_model(self):
        """Load model and tokenizer"""
        logger.info("="*70)
        logger.info("STEP 1: Loading Model and Tokenizer")
        logger.info("="*70)
        
        try:
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.eval()
            logger.info(f"✓ Model loaded: {type(self.model).__name__}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info(f"✓ Tokenizer loaded: {type(self.tokenizer).__name__}")
            
            # Print model info
            logger.info("  Model config:")
            logger.info(f"    - Hidden size: {self.model.config.hidden_size}")
            logger.info(f"    - Number of labels: {self.model.config.num_labels}")
            logger.info(f"    - Vocab size: {self.model.config.vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_dummy_input(self):
        """
        Create dummy input for ONNX export
        
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Creating Dummy Input")
        logger.info("="*70)
        
        # Create dummy text
        dummy_text = "مرحبا كيف حالك"
        
        # Tokenize
        inputs = self.tokenizer(
            dummy_text,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        logger.info("✓ Dummy input created")
        logger.info(f"  Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"  Attention mask shape: {inputs['attention_mask'].shape}")
        
        return inputs['input_ids'], inputs['attention_mask']
    
    def export_to_onnx(self, input_ids, attention_mask):
        """
        Export model to ONNX format
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Exporting to ONNX")
        logger.info("="*70)
        
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Define input and output names
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        
        # Define dynamic axes
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }
        
        logger.info(f"Exporting model to {self.output_path}...")
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                (input_ids, attention_mask),
                self.output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                export_params=True,
            )
            
            logger.info("✓ Model exported to ONNX format")
            
            # Get file size
            file_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def validate_onnx(self):
        """Validate the exported ONNX model"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Validating ONNX Model")
        logger.info("="*70)
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(self.output_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model is valid")
            
            # Print model info
            logger.info("  Model info:")
            logger.info(f"    - IR version: {onnx_model.ir_version}")
            logger.info(f"    - Producer: {onnx_model.producer_name}")
            logger.info(f"    - Opset version: {onnx_model.opset_import[0].version}")
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise
    
    def test_onnx_runtime(self):
        """Test ONNX model with ONNX Runtime"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Testing with ONNX Runtime")
        logger.info("="*70)
        
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(
                self.output_path,
                providers=['CPUExecutionProvider']
            )
            logger.info("✓ ONNX Runtime session created")
            
            # Print session info
            logger.info(f"  Providers: {session.get_providers()}")
            
            # Test inference
            test_text = "مرحبا كيف حالك"
            inputs = self.tokenizer(
                test_text,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="np"
            )
            
            # Run inference
            outputs = session.run(
                None,
                {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
            )
            
            # Get predictions
            logits = outputs[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            predicted_class = np.argmax(probs, axis=-1)[0]
            confidence = float(probs[0, predicted_class])
            
            logger.info("✓ Inference test successful")
            logger.info(f"  Test text: {test_text}")
            logger.info(f"  Predicted class: {predicted_class}")
            logger.info(f"  Confidence: {confidence:.4f}")
            
            # Compare with PyTorch model
            logger.info("\nComparing with PyTorch model...")
            self.model.eval()
            with torch.no_grad():
                pt_inputs = self.tokenizer(
                    test_text,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                pt_outputs = self.model(**pt_inputs)
                pt_logits = pt_outputs.logits.numpy()
                pt_probs = np.exp(pt_logits) / np.sum(np.exp(pt_logits), axis=-1, keepdims=True)
                pt_predicted_class = np.argmax(pt_probs, axis=-1)[0]
                pt_confidence = float(pt_probs[0, pt_predicted_class])
            
            logger.info(f"  PyTorch predicted class: {pt_predicted_class}")
            logger.info(f"  PyTorch confidence: {pt_confidence:.4f}")
            
            # Check if outputs match
            if predicted_class == pt_predicted_class:
                logger.info("✓ ONNX and PyTorch predictions match!")
            else:
                logger.warning("⚠ ONNX and PyTorch predictions differ!")
            
            # Check logits difference
            max_diff = np.max(np.abs(logits - pt_logits))
            logger.info(f"  Max logits difference: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                logger.info("✓ Logits are very close (difference < 1e-4)")
            elif max_diff < 1e-3:
                logger.info("✓ Logits are close (difference < 1e-3)")
            else:
                logger.warning(f"⚠ Logits difference is significant: {max_diff:.6f}")
            
        except Exception as e:
            logger.error(f"ONNX Runtime test failed: {e}")
            raise
    
    def convert(self):
        """Run the complete conversion pipeline"""
        logger.info("\n" + "="*70)
        logger.info("ONNX Conversion Pipeline")
        logger.info("="*70)
        
        # Load model
        self.load_model()
        
        # Create dummy input
        input_ids, attention_mask = self.create_dummy_input()
        
        # Export to ONNX
        self.export_to_onnx(input_ids, attention_mask)
        
        # Validate ONNX
        self.validate_onnx()
        
        # Test with ONNX Runtime
        self.test_onnx_runtime()
        
        logger.info("\n" + "="*70)
        logger.info("✓ Conversion Complete!")
        logger.info(f"ONNX model saved to: {self.output_path}")
        logger.info("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert EOU model to ONNX format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save ONNX model"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = ONNXConverter(
        model_path=args.model_path,
        output_path=args.output_path,
        opset_version=args.opset_version,
        max_seq_length=args.max_seq_length,
    )
    
    # Run conversion
    converter.convert()


if __name__ == "__main__":
    main()
