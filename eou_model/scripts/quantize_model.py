#!/usr/bin/env python3
"""
Model Quantization Script
=========================

Quantize ONNX model to reduce size and improve inference speed.

Usage:
    python quantize_model.py --model_path ./models/eou_model.onnx --output_path ./models/eou_model_quantized.onnx
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantize ONNX model for optimized inference"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        tokenizer_path: str = None,
        max_seq_length: int = 512,
    ):
        """
        Initialize Model Quantizer
        
        Args:
            model_path: Path to ONNX model
            output_path: Path to save quantized model
            tokenizer_path: Path to tokenizer (optional)
            max_seq_length: Maximum sequence length
        """
        self.model_path = model_path
        self.output_path = output_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        
        self.tokenizer = None
        
        logger.info(f"Initialized ModelQuantizer")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Output path: {output_path}")
    
    def load_tokenizer(self):
        """Load tokenizer if path provided"""
        if self.tokenizer_path:
            logger.info(f"Loading tokenizer from {self.tokenizer_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                logger.info("✓ Tokenizer loaded")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                logger.warning("Continuing without tokenizer (validation will be skipped)")
    
    def get_model_info(self, model_path: str):
        """
        Get information about ONNX model
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Dictionary with model info
        """
        # File size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Load model
        model = onnx.load(model_path)
        
        # Count parameters
        total_params = 0
        for initializer in model.graph.initializer:
            dims = initializer.dims
            param_count = 1
            for dim in dims:
                param_count *= dim
            total_params += param_count
        
        return {
            "file_size_mb": file_size_mb,
            "total_params": total_params,
            "ir_version": model.ir_version,
            "opset_version": model.opset_import[0].version,
        }
    
    def quantize(self):
        """Quantize the ONNX model"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Model Information (Before Quantization)")
        logger.info("="*70)
        
        # Get original model info
        original_info = self.get_model_info(self.model_path)
        logger.info(f"Original model:")
        logger.info(f"  File size: {original_info['file_size_mb']:.2f} MB")
        logger.info(f"  Total parameters: {original_info['total_params']:,}")
        logger.info(f"  IR version: {original_info['ir_version']}")
        logger.info(f"  Opset version: {original_info['opset_version']}")
        
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Quantizing Model")
        logger.info("="*70)
        
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Quantize model
            logger.info("Applying dynamic quantization...")
            logger.info("  Quantization type: INT8")
            logger.info("  Method: Dynamic quantization")
            
            quantize_dynamic(
                model_input=self.model_path,
                model_output=self.output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True,
            )
            
            logger.info("✓ Quantization complete")
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise
        
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Model Information (After Quantization)")
        logger.info("="*70)
        
        # Get quantized model info
        quantized_info = self.get_model_info(self.output_path)
        logger.info(f"Quantized model:")
        logger.info(f"  File size: {quantized_info['file_size_mb']:.2f} MB")
        logger.info(f"  Total parameters: {quantized_info['total_params']:,}")
        logger.info(f"  IR version: {quantized_info['ir_version']}")
        logger.info(f"  Opset version: {quantized_info['opset_version']}")
        
        # Calculate reduction
        size_reduction = (1 - quantized_info['file_size_mb'] / original_info['file_size_mb']) * 100
        logger.info(f"\n✓ Size reduction: {size_reduction:.1f}%")
        logger.info(f"  Original: {original_info['file_size_mb']:.2f} MB")
        logger.info(f"  Quantized: {quantized_info['file_size_mb']:.2f} MB")
        logger.info(f"  Saved: {original_info['file_size_mb'] - quantized_info['file_size_mb']:.2f} MB")
    
    def validate(self):
        """Validate quantized model"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Validating Quantized Model")
        logger.info("="*70)
        
        try:
            # Load quantized model
            logger.info("Loading quantized model...")
            quantized_session = ort.InferenceSession(
                self.output_path,
                providers=['CPUExecutionProvider']
            )
            logger.info("✓ Quantized model loaded successfully")
            
            # If tokenizer available, test inference
            if self.tokenizer:
                logger.info("\nTesting inference...")
                
                # Test texts
                test_texts = [
                    "مرحبا كيف حالك",
                    "أنا بخير شكرا",
                    "ما هو اسمك",
                ]
                
                # Load original model for comparison
                original_session = ort.InferenceSession(
                    self.model_path,
                    providers=['CPUExecutionProvider']
                )
                
                logger.info(f"\nTesting {len(test_texts)} examples...")
                
                matches = 0
                total_diff = 0.0
                
                for i, text in enumerate(test_texts, 1):
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        padding="max_length",
                        max_length=self.max_seq_length,
                        return_tensors="np"
                    )
                    
                    onnx_inputs = {
                        'input_ids': inputs['input_ids'].astype(np.int64),
                        'attention_mask': inputs['attention_mask'].astype(np.int64)
                    }
                    
                    # Original model inference
                    orig_outputs = original_session.run(None, onnx_inputs)
                    orig_logits = orig_outputs[0]
                    orig_pred = np.argmax(orig_logits, axis=-1)[0]
                    
                    # Quantized model inference
                    quant_outputs = quantized_session.run(None, onnx_inputs)
                    quant_logits = quant_outputs[0]
                    quant_pred = np.argmax(quant_logits, axis=-1)[0]
                    
                    # Compare
                    logits_diff = np.max(np.abs(orig_logits - quant_logits))
                    total_diff += logits_diff
                    
                    if orig_pred == quant_pred:
                        matches += 1
                        status = "✓"
                    else:
                        status = "✗"
                    
                    logger.info(f"  Test {i}: {status} (logits diff: {logits_diff:.6f})")
                
                # Summary
                accuracy = (matches / len(test_texts)) * 100
                avg_diff = total_diff / len(test_texts)
                
                logger.info(f"\n✓ Validation Results:")
                logger.info(f"  Prediction accuracy: {accuracy:.1f}% ({matches}/{len(test_texts)})")
                logger.info(f"  Average logits difference: {avg_diff:.6f}")
                
                if accuracy == 100:
                    logger.info("  ✓ All predictions match!")
                elif accuracy >= 90:
                    logger.info("  ✓ Most predictions match (acceptable)")
                else:
                    logger.warning("  ⚠ Many predictions differ (may need investigation)")
            
            else:
                logger.info("✓ Model loads successfully (tokenizer not available for inference test)")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def run(self):
        """Run the complete quantization pipeline"""
        logger.info("\n" + "="*70)
        logger.info("Model Quantization Pipeline")
        logger.info("="*70)
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Quantize
        self.quantize()
        
        # Validate
        self.validate()
        
        logger.info("\n" + "="*70)
        logger.info("✓ Quantization Complete!")
        logger.info(f"Quantized model saved to: {self.output_path}")
        logger.info("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (optional, for validation)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = ModelQuantizer(
        model_path=args.model_path,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
    )
    
    # Run quantization
    quantizer.run()


if __name__ == "__main__":
    main()
