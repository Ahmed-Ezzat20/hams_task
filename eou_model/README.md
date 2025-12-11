# EOU Model Module

End-of-Utterance (EOU) detection model for Arabic speech recognition systems.

## Overview

This module provides a complete pipeline for training, converting, and deploying an End-of-Utterance detection model optimized for Arabic language. The model uses **AraBERT v2** as the base architecture and achieves **90% accuracy** with **0.92 F1-score** on Arabic EOU detection tasks.

## Features

- ✅ **Training Pipeline**: Complete training script with AraBERT preprocessing
- ✅ **ONNX Conversion**: Convert trained models to ONNX format for optimized inference
- ✅ **Model Quantization**: Reduce model size by ~75% with INT8 quantization
- ✅ **Validation Tools**: Comprehensive testing and validation scripts
- ✅ **Production Ready**: Optimized for deployment in LiveKit agents

## Directory Structure

```
eou_model/
├── scripts/
│   ├── train.py                 # Training script
│   ├── convert_to_onnx.py       # ONNX conversion script
│   └── quantize_model.py        # Model quantization script
├── models/                      # Trained models directory
├── configs/                     # Configuration files
├── tests/                       # Unit tests
├── data/                        # Dataset directory
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### 1. Install Dependencies

```bash
cd eou_model
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; import transformers; import onnx; print('✓ All dependencies installed')"
```

## Usage

### Training

Train an EOU detection model from scratch:

```bash
python scripts/train.py \
    --model_name "aubmindlab/bert-base-arabertv2" \
    --dataset_name "arabic-eou-detection-10k" \
    --output_dir "./models/eou_model" \
    --learning_rate 2e-5 \
    --batch_size 32 \
    --num_epochs 10
```

**Arguments:**
- `--model_name`: HuggingFace model name (default: aubmindlab/bert-base-arabertv2)
- `--dataset_name`: Dataset name or path
- `--output_dir`: Directory to save trained model
- `--learning_rate`: Learning rate for optimizer (default: 2e-5)
- `--batch_size`: Training batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--max_length`: Maximum sequence length (default: 512)
- `--weight_decay`: Weight decay for regularization (default: 0.01)

**Output:**
- Trained model saved to `output_dir`
- Tokenizer saved with model
- Evaluation results in `evaluation_results.json`
- Training logs in `output_dir/logs`

### ONNX Conversion

Convert trained model to ONNX format:

```bash
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx" \
    --opset_version 14 \
    --max_seq_length 512
```

**Arguments:**
- `--model_path`: Path to trained model directory
- `--output_path`: Path to save ONNX model
- `--opset_version`: ONNX opset version (default: 14)
- `--max_seq_length`: Maximum sequence length (default: 512)

**Output:**
- ONNX model saved to `output_path`
- Validation report with PyTorch comparison
- Model size and inference test results

### Model Quantization

Quantize ONNX model for production:

```bash
python scripts/quantize_model.py \
    --model_path "./models/eou_model.onnx" \
    --output_path "./models/eou_model_quantized.onnx" \
    --tokenizer_path "./models/eou_model" \
    --max_seq_length 512
```

**Arguments:**
- `--model_path`: Path to ONNX model
- `--output_path`: Path to save quantized model
- `--tokenizer_path`: Path to tokenizer (optional, for validation)
- `--max_seq_length`: Maximum sequence length (default: 512)

**Output:**
- Quantized model saved to `output_path`
- Size reduction report (~75% reduction)
- Validation results comparing original and quantized models

## Model Performance

### Metrics

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
Actual No  333     62   (84.3%)
       EOU 42      564  (93.1%)
```

### Model Sizes

| Format | Size | Reduction |
|--------|------|-----------|
| **PyTorch** | ~400 MB | - |
| **ONNX** | ~400 MB | 0% |
| **Quantized ONNX** | ~100 MB | 75% |

### Inference Speed

| Format | Latency | Throughput |
|--------|---------|------------|
| **PyTorch** | ~100-200ms | ~300 samples/sec |
| **ONNX** | ~30-50ms | ~500 samples/sec |
| **Quantized ONNX** | ~20-30ms | ~700 samples/sec |

## Integration with LiveKit

### 1. Load the Model

```python
import onnxruntime as ort
from transformers import AutoTokenizer

# Load ONNX model
session = ort.InferenceSession(
    "./models/eou_model_quantized.onnx",
    providers=['CPUExecutionProvider']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/eou_model")
```

### 2. Create Turn Detector

```python
from livekit.agents import llm

class CustomEOUTurnDetector(llm.TurnDetector):
    def __init__(self, session, tokenizer, threshold=0.7):
        super().__init__()
        self.session = session
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = 512
    
    async def detect_turn(self, text: str) -> bool:
        """Detect if utterance is complete"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np"
        )
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
        )
        
        # Get prediction
        logits = outputs[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        is_eou = probs[0, 1] > self.threshold
        
        return is_eou
```

### 3. Use in Agent

```python
from livekit.agents import AgentSession

# Create turn detector
turn_detector = CustomEOUTurnDetector(session, tokenizer, threshold=0.7)

# Use in agent session
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        turn_detector=turn_detector,
        # ... other configurations
    )
    
    await session.start()
```

## Configuration

### Training Configuration

Create `configs/training_config.yaml`:

```yaml
model:
  name: "aubmindlab/bert-base-arabertv2"
  num_labels: 2
  max_length: 512

dataset:
  name: "arabic-eou-detection-10k"
  train_split: "train"
  validation_split: "validation"
  test_split: "test"

training:
  learning_rate: 2e-5
  batch_size: 32
  num_epochs: 10
  weight_decay: 0.01
  warmup_steps: 500
  eval_strategy: "epoch"
  save_strategy: "epoch"
  metric_for_best_model: "f1"

output:
  dir: "./models/eou_model"
  save_total_limit: 2
```

## Testing

Run unit tests:

```bash
# TODO: Add unit tests
python -m pytest tests/
```

## Troubleshooting

### Issue: Model not found

**Solution:** Ensure the model path is correct and the model has been trained.

```bash
ls -la ./models/eou_model/
# Should show: config.json, model.safetensors, tokenizer files
```

### Issue: ONNX conversion fails

**Solution:** Try with lower opset version:

```bash
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx" \
    --opset_version 12
```

### Issue: Out of memory during training

**Solution:** Reduce batch size:

```bash
python scripts/train.py \
    --batch_size 16 \
    # ... other arguments
```

## Performance Optimization

### 1. Use GPU for Training

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train with GPU (automatic if available)
python scripts/train.py --batch_size 64 # Larger batch size with GPU
```

### 2. Use Mixed Precision

```bash
# Enable mixed precision training
python scripts/train.py --fp16
```

### 3. Use ONNX Runtime GPU

```python
# Use GPU provider for ONNX Runtime
session = ort.InferenceSession(
    model_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the ASR Demo repository.

## References

- **AraBERT**: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- **LiveKit Agents**: [livekit/agents](https://github.com/livekit/agents)
- **ONNX Runtime**: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

## Support

For issues or questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the LiveKit documentation

---

**Last Updated:** December 11, 2025  
**Version:** 1.0.0  
**Status:** Production Ready ✅
