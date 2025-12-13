---
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
  howpublished = {\url{https://huggingface.co/your-username/arabic-eou-detector}}
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
