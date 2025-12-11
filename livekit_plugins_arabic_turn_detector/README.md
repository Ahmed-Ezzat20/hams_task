# LiveKit Arabic Turn Detector Plugin

A LiveKit plugin for detecting end-of-utterance (EOU) in Arabic conversations using a transformer-based model optimized for Arabic language.

## Features

- ✅ **Arabic-Optimized**: Uses AraBERT preprocessing for accurate Arabic text handling
- ✅ **High Performance**: ONNX-optimized model with 90% accuracy
- ✅ **Low Latency**: 20-30ms inference time with quantized model
- ✅ **Easy Integration**: Drop-in replacement for LiveKit's default turn detector
- ✅ **Configurable**: Adjustable confidence threshold for different use cases
- ✅ **Production-Ready**: Tested and optimized for real-time conversations

## Installation

### From Source

```bash
cd livekit_plugins_arabic_turn_detector
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Model

First, ensure you have your ONNX model ready:

```bash
# Train and convert your model (see eou_model/ directory)
cd ../eou_model
python scripts/train.py --output_dir ./models/eou_model
python scripts/convert_to_onnx.py --model_path ./models/eou_model --output_path ./models/eou_model.onnx
python scripts/quantize_model.py --model_path ./models/eou_model.onnx --output_path ./models/eou_model_quantized.onnx
```

### 2. Use in Your Agent

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector
from livekit.agents import AgentSession, JobContext

# Create Arabic turn detector
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.7  # Adjust based on your needs
)

# Use in agent
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        turn_detector=turn_detector,
        # ... other configurations
    )
    
    await session.start()
```

## Configuration

### Threshold Tuning

The `unlikely_threshold` parameter controls the sensitivity of turn detection:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.5-0.6** | Very sensitive | Quick responses, may interrupt |
| **0.7** (default) | Balanced | Most conversations |
| **0.8-0.9** | Conservative | Avoid interruptions, wait longer |

```python
# More sensitive (faster responses, may interrupt)
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.6
)

# More conservative (fewer interruptions, slower responses)
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.8
)
```

## Advanced Usage

### Get Confidence Scores

```python
# Get EOU confidence for debugging
chat_ctx = [
    {"role": "user", "content": "مرحبا كيف حالك"},
    {"role": "assistant", "content": "أنا بخير شكرا"},
]

confidence = turn_detector.get_confidence(chat_ctx)
print(f"EOU confidence: {confidence:.4f}")
```

### Manual Turn Detection

```python
# Manually check if utterance is complete
is_complete = await turn_detector.detect_turn(chat_ctx)
if is_complete:
    print("Utterance is complete, agent can respond")
else:
    print("User is still speaking, wait for more input")
```

## Integration with LiveKit Agent

### Complete Example

```python
import asyncio
from livekit import rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import elevenlabs, deepgram
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

# Create Arabic turn detector
arabic_turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

async def entrypoint(ctx: JobContext):
    """Agent entrypoint"""
    await ctx.connect()
    
    # Create agent session with Arabic turn detector
    session = AgentSession(
        # STT (Speech-to-Text)
        stt=deepgram.STT(language="ar"),
        
        # TTS (Text-to-Speech)
        tts=elevenlabs.TTS(
            voice_id="your-arabic-voice-id",
            model_id="eleven_multilingual_v2"
        ),
        
        # Turn detection with Arabic model
        turn_detector=arabic_turn_detector,
        
        # Other configurations
        min_endpointing_delay=0.5,
        interrupt_speech_duration=0.3,
        interrupt_min_words=2,
    )
    
    # Start the session
    await session.start()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
```

## Model Performance

### Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 90% |
| **Precision (EOU)** | 0.90 |
| **Recall (EOU)** | 0.93 |
| **F1-Score (EOU)** | 0.92 |
| **Inference Time** | 20-30ms (quantized) |
| **Model Size** | ~100MB (quantized) |

### Confusion Matrix

```
           Predicted
           No EOU  EOU
Actual No  333     62   (84.3%)
       EOU 42      564  (93.1%)
```

## Architecture

### Plugin Structure

```
livekit_plugins_arabic_turn_detector/
├── livekit/
│   └── plugins/
│       └── arabic_turn_detector/
│           ├── __init__.py       # Plugin exports
│           ├── arabic.py         # Main implementation
│           └── version.py        # Version info
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

### How It Works

1. **Text Normalization**: Input text is normalized using AraBERT preprocessing
2. **Tokenization**: Text is tokenized using AraBERT tokenizer
3. **ONNX Inference**: Model runs inference on CPU with optimized settings
4. **Prediction**: Returns binary classification (EOU or not) with confidence score
5. **Threshold Comparison**: Confidence is compared against threshold to make final decision

```
User Speech → STT (Arabic) → Text Normalization → Tokenization
                                                       ↓
                                                  ONNX Model
                                                       ↓
                                              Confidence Score
                                                       ↓
                                            Threshold Comparison
                                                       ↓
                                              EOU Decision
                                                       ↓
                                          Agent Response Trigger
```

## Troubleshooting

### Issue: Model not found

**Error:**
```
RuntimeError: Model path not provided
```

**Solution:**
Ensure you provide the correct path to your ONNX model:

```python
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx"  # Correct path
)
```

### Issue: AraBERT not installed

**Warning:**
```
arabert not installed. Install with: pip install arabert
Falling back to basic Arabic normalization.
```

**Solution:**
Install arabert for better Arabic text preprocessing:

```bash
pip install arabert
```

### Issue: Tokenizer not found

**Warning:**
```
Failed to load tokenizer: ...
Continuing without tokenizer (will use fallback formatting)
```

**Solution:**
Ensure tokenizer files are in the same directory as your ONNX model:

```
models/
├── eou_model_quantized.onnx
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### Issue: Slow inference

**Problem:** Inference takes too long (>100ms)

**Solutions:**
1. Use quantized model (75% faster)
2. Reduce `MAX_HISTORY_TOKENS` in code
3. Use GPU provider (if available):
   ```python
   # Modify arabic.py to use GPU
   providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
   ```

### Issue: Too many interruptions

**Problem:** Agent interrupts user too frequently

**Solution:** Increase threshold:

```python
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.8  # Higher = less sensitive
)
```

### Issue: Agent waits too long

**Problem:** Agent takes too long to respond

**Solution:** Decrease threshold:

```python
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.6  # Lower = more sensitive
)
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Formatting

```bash
# Format code
black livekit/
isort livekit/

# Type checking
mypy livekit/
```

## Comparison with Other Turn Detectors

| Feature | Arabic Turn Detector | LiveKit Default | Deepgram VAD |
|---------|---------------------|-----------------|--------------|
| **Language** | Arabic-optimized | English | Multilingual |
| **Accuracy** | 90% (Arabic) | 99.3% (English) | 85% |
| **Latency** | 20-30ms | 50-160ms | <50ms |
| **Model Size** | 100MB | 396MB | Small |
| **Offline** | ✅ Yes | ✅ Yes | ❌ No (API) |
| **Cost** | Free | Free | Paid |
| **Customizable** | ✅ Yes | ❌ No | ❌ No |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Apache-2.0 License

## References

- **LiveKit Agents**: https://github.com/livekit/agents
- **AraBERT**: https://huggingface.co/aubmindlab/bert-base-arabertv2
- **ONNX Runtime**: https://github.com/microsoft/onnxruntime

## Support

For issues or questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the LiveKit documentation

---

**Version:** 0.1.0  
**Status:** Alpha  
**Last Updated:** December 11, 2025
