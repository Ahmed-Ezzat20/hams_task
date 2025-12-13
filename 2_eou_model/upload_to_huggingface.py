#!/usr/bin/env python3
"""
Upload EOU Model to HuggingFace Hub

This script uploads the trained EOU model (PyTorch and ONNX versions) to HuggingFace Hub
for easy sharing and inference.

Usage:
    python scripts/upload_to_huggingface.py \\
        --model_path ./models/eou_model \\
        --repo_name "your-username/arabic-eou-detector" \\
        --token "your_hf_token"
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """Upload model to HuggingFace Hub"""
    
    def __init__(
        self,
        model_path: str,
        repo_name: str,
        token: Optional[str] = None,
        private: bool = False,
        onnx_model_path: Optional[str] = None,
        quantized_model_path: Optional[str] = None,
    ):
        """
        Initialize uploader
        
        Args:
            model_path: Path to trained PyTorch model directory
            repo_name: HuggingFace repo name (username/repo-name)
            token: HuggingFace API token (or set HF_TOKEN env var)
            private: Whether to create a private repository
            onnx_model_path: Optional path to ONNX model
            quantized_model_path: Optional path to quantized ONNX model
        """
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.token = token or os.getenv("HF_TOKEN")
        self.private = private
        self.onnx_model_path = Path(onnx_model_path) if onnx_model_path else None
        self.quantized_model_path = Path(quantized_model_path) if quantized_model_path else None
        
        # Initialize HuggingFace API
        self.api = HfApi(token=self.token)
        
        # Temporary directory for organizing files
        self.temp_dir = Path("./temp_hf_upload")
        
        logger.info("Initialized HuggingFaceUploader")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Repository: {self.repo_name}")
        logger.info(f"  Private: {self.private}")
    
    def prepare_files(self):
        """Prepare files for upload"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Preparing Files")
        logger.info("="*70)
        
        # Create temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)
        
        # Copy PyTorch model files
        logger.info("Copying PyTorch model files...")
        for file in self.model_path.glob("*"):
            if file.is_file():
                shutil.copy2(file, self.temp_dir / file.name)
                logger.info(f"  ✓ {file.name}")
        
        # Copy ONNX model if provided
        if self.onnx_model_path and self.onnx_model_path.exists():
            logger.info("Copying ONNX model...")
            shutil.copy2(self.onnx_model_path, self.temp_dir / "model.onnx")
            logger.info("  ✓ model.onnx")
        
        # Copy quantized ONNX model if provided
        if self.quantized_model_path and self.quantized_model_path.exists():
            logger.info("Copying quantized ONNX model...")
            shutil.copy2(self.quantized_model_path, self.temp_dir / "model_quantized.onnx")
            logger.info("  ✓ model_quantized.onnx")
        
        logger.info(f"\n✓ Files prepared in {self.temp_dir}")
    
    def create_model_card(self):
        """Create model card (README.md)"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Creating Model Card")
        logger.info("="*70)
        
        model_card = """---
language:
- ar
license: apache-2.0
tags:
- arabic
- end-of-utterance
- eou
- turn-detection
- conversational-ai
- livekit
- bert
- arabert
datasets:
- arabic-eou-detection-10k
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: Arabic End-of-Utterance Detector
  results:
  - task:
      type: text-classification
      name: End-of-Utterance Detection
    dataset:
      name: Arabic EOU Detection
      type: arabic-eou-detection-10k
    metrics:
    - type: accuracy
      value: 0.90
      name: Accuracy
    - type: f1
      value: 0.92
      name: F1 Score (EOU)
    - type: precision
      value: 0.90
      name: Precision (EOU)
    - type: recall
      value: 0.93
      name: Recall (EOU)
---

# Arabic End-of-Utterance (EOU) Detector

**Detect when a speaker has finished their utterance in Arabic conversations.**

This model is fine-tuned from [AraBERT v2](https://huggingface.co/aubmindlab/bert-base-arabertv2) for binary classification of Arabic text to determine if an utterance is complete (EOU) or incomplete (No EOU).

## Model Description

- **Model Type**: BERT-based binary classifier
- **Base Model**: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- **Language**: Arabic (ar)
- **Task**: End-of-Utterance Detection
- **License**: Apache 2.0

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 90% |
| **Precision (EOU)** | 0.90 |
| **Recall (EOU)** | 0.93 |
| **F1-Score (EOU)** | 0.92 |
| **Test Samples** | 1,001 |

### Confusion Matrix

```
           Predicted
           No EOU  EOU
Actual No  333     62   (84.3% correct)
       EOU 42      564  (93.1% correct)
```

## Available Formats

This repository includes three model formats:

1. **PyTorch** (`pytorch_model.bin` or `model.safetensors`) - For training and fine-tuning
2. **ONNX** (`model.onnx`) - For optimized CPU/GPU inference (~2-3x faster)
3. **Quantized ONNX** (`model_quantized.onnx`) - For production (75% smaller, 2-3x faster)

## Quick Start

### Installation

```bash
pip install transformers torch onnxruntime
```

### PyTorch Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "your-username/arabic-eou-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Inference
def predict_eou(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    is_eou = torch.argmax(probs, dim=-1).item() == 1
    confidence = probs[0, 1].item()
    
    return is_eou, confidence

# Test
text = "مرحبا كيف حالك"
is_eou, conf = predict_eou(text)
print(f"Is EOU: {is_eou}, Confidence: {conf:.4f}")
```

### ONNX Inference (Recommended for Production)

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Load model and tokenizer
model_name = "your-username/arabic-eou-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load ONNX model (use model_quantized.onnx for best performance)
session = ort.InferenceSession(
    "model_quantized.onnx",  # or "model.onnx"
    providers=['CPUExecutionProvider']
)

# Inference
def predict_eou(text: str):
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="np"
    )
    
    outputs = session.run(
        None,
        {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
    )
    
    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    is_eou = np.argmax(probs, axis=-1)[0] == 1
    confidence = float(probs[0, 1])
    
    return is_eou, confidence

# Test
text = "مرحبا كيف حالك"
is_eou, conf = predict_eou(text)
print(f"Is EOU: {is_eou}, Confidence: {conf:.4f}")
```

## Use Cases

- **Voice Assistants**: Detect when user has finished speaking
- **Conversational AI**: Improve turn-taking in Arabic chatbots
- **LiveKit Agents**: Custom turn detection for Arabic conversations
- **Speech Recognition**: Post-processing for better utterance segmentation

## Integration with LiveKit

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

# Download model from HuggingFace
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/arabic-eou-detector",
    filename="model_quantized.onnx"
)

# Create turn detector
turn_detector = ArabicTurnDetector(
    model_path=model_path,
    unlikely_threshold=0.7
)

# Use in agent
session = AgentSession(
    turn_detector=turn_detector,
    # ... other config
)
```

## Training Details

### Training Data

- **Dataset**: Arabic EOU Detection (10,072 samples)
- **Train/Val/Test Split**: 80/10/10
- **Classes**: 
  - `0`: Incomplete utterance (No EOU)
  - `1`: Complete utterance (EOU)

### Training Hyperparameters

- **Base Model**: aubmindlab/bert-base-arabertv2
- **Learning Rate**: 2e-5
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Max Sequence Length**: 512

### Preprocessing

- AraBERT normalization (diacritics removal, character normalization)
- Tokenization with AraBERT tokenizer
- Padding to max length (512 tokens)

## Limitations

- **Language**: Optimized for Modern Standard Arabic (MSA)
- **Domain**: Trained on conversational Arabic text
- **Sequence Length**: Maximum 512 tokens
- **Dialects**: May have reduced accuracy on dialectal Arabic

## Citation

If you use this model, please cite:

```bibtex
@misc{arabic-eou-detector,
  author = {Your Name},
  title = {Arabic End-of-Utterance Detector},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/your-username/arabic-eou-detector}}
}
```

## License

Apache 2.0

## Acknowledgments

- **AraBERT**: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- **HuggingFace Transformers**: Model training and inference
- **ONNX Runtime**: Model optimization and deployment

## Contact

For issues or questions, please open an issue on the [GitHub repository](https://github.com/Ahmed-Ezzat20/hams_task).
"""
        
        # Write model card
        readme_path = self.temp_dir / "README.md"
        readme_path.write_text(model_card)
        logger.info(f"✓ Model card created: {readme_path}")
    
    def create_repository(self):
        """Create HuggingFace repository"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Creating Repository")
        logger.info("="*70)
        
        try:
            repo_url = create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=self.private,
                exist_ok=True,
            )
            logger.info(f"✓ Repository created/verified: {repo_url}")
            return repo_url
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise
    
    def upload_files(self):
        """Upload files to HuggingFace Hub"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Uploading Files")
        logger.info("="*70)
        
        try:
            logger.info(f"Uploading to {self.repo_name}...")
            
            upload_folder(
                folder_path=str(self.temp_dir),
                repo_id=self.repo_name,
                token=self.token,
                commit_message="Upload Arabic EOU detection model",
            )
            
            logger.info("✓ Files uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload files: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Cleanup")
        logger.info("="*70)
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"✓ Removed temporary directory: {self.temp_dir}")
    
    def run(self):
        """Run the complete upload process"""
        logger.info("\n" + "="*70)
        logger.info("HuggingFace Model Upload Pipeline")
        logger.info("="*70)
        
        try:
            # Prepare files
            self.prepare_files()
            
            # Create model card
            self.create_model_card()
            
            # Create repository and upload files
            self.create_repository()

            # Upload files
            self.upload_files()
            
            # Cleanup
            self.cleanup()
            
            # Success message
            logger.info("\n" + "="*70)
            logger.info("✓ Upload Complete!")
            logger.info("="*70)
            logger.info(f"Model URL: https://huggingface.co/{self.repo_name}")
            logger.info("\nTo use your model:")
            logger.info('  from transformers import AutoTokenizer, AutoModelForSequenceClassification')
            logger.info(f'  tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")')
            logger.info(f'  model = AutoModelForSequenceClassification.from_pretrained("{self.repo_name}")')
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            # Cleanup on error
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload EOU model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained PyTorch model directory"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="HuggingFace repository name (username/repo-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default=None,
        help="Path to ONNX model (optional)"
    )
    parser.add_argument(
        "--quantized_model_path",
        type=str,
        default=None,
        help="Path to quantized ONNX model (optional)"
    )
    
    args = parser.parse_args()
    
    # Create uploader
    uploader = HuggingFaceUploader(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        onnx_model_path=args.onnx_model_path,
        quantized_model_path=args.quantized_model_path,
    )
    
    # Run upload
    uploader.run()


if __name__ == "__main__":
    main()
