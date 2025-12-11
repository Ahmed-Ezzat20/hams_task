# LiveKit Arabic Turn Detector Plugin

**Production-ready LiveKit plugin for Arabic end-of-utterance detection in real-time voice conversations.**

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/accuracy-90%25-blue)]()
[![Latency](https://img.shields.io/badge/latency-20--30ms-green)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Integration](#basic-integration)
  - [Complete Example](#complete-example)
  - [Advanced Usage](#advanced-usage)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Architecture](#architecture)
- [Integration Guide](#integration-guide)
- [Troubleshooting](#troubleshooting)
- [Comparison](#comparison-with-other-turn-detectors)

---

## Overview

A LiveKit plugin that detects **end-of-utterance (EOU)** in Arabic conversations using a transformer-based model optimized for Arabic language. Built on **AraBERT v2** with **90% accuracy** and **20-30ms latency**.

### Features

- ✅ **Arabic-Optimized**: AraBERT preprocessing for accurate Arabic text handling
- ✅ **High Performance**: 90% accuracy, 0.92 F1-score
- ✅ **Low Latency**: 20-30ms inference with quantized ONNX model
- ✅ **Easy Integration**: Drop-in replacement for LiveKit's default turn detector
- ✅ **Configurable**: Adjustable confidence threshold (0.5-0.9)
- ✅ **Production-Ready**: Tested and optimized for real-time conversations
- ✅ **Offline**: No external API calls, runs on-device

### Directory Structure

```
livekit_plugins_arabic_turn_detector/
├── livekit/plugins/arabic_turn_detector/
│   ├── __init__.py              # Plugin exports
│   ├── arabic.py                # Main implementation (450+ lines)
│   ├── version.py               # Version info
│   └── py.typed                 # Type hints marker
├── examples/
│   └── simple_agent.py          # Complete working example
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Quick Start

Get started in **10 minutes**!

### Prerequisites

- Python 3.9+
- LiveKit server (cloud or self-hosted)
- Trained Arabic EOU model (see `../eou_model/`)
- STT/TTS providers (Deepgram, ElevenLabs, etc.)

### Step 1: Prepare Model (5 minutes)

```bash
# Train and convert model
cd ../eou_model
python scripts/train.py --output_dir ./models/eou_model
python scripts/convert_to_onnx.py --model_path ./models/eou_model --output_path ./models/eou_model.onnx
python scripts/quantize_model.py --model_path ./models/eou_model.onnx --output_path ./models/eou_model_quantized.onnx
```

### Step 2: Install Plugin (1 minute)

```bash
cd ../livekit_plugins_arabic_turn_detector
pip install -e .
```

### Step 3: Use in Agent (2 minutes)

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector
from livekit.agents import AgentSession, JobContext

# Create turn detector
turn_detector = ArabicTurnDetector(
    model_path="../eou_model/models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

# Use in agent
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        turn_detector=turn_detector,
        stt=your_stt_provider,
        tts=your_tts_provider,
    )
    
    await session.start()
```

### Step 4: Run Agent

```bash
# Set environment variables
export LIVEKIT_URL="wss://your-server.com"
export LIVEKIT_API_KEY="your-key"
export LIVEKIT_API_SECRET="your-secret"

# Run agent
python examples/simple_agent.py
```

---

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

**Required packages:**
- livekit-agents >= 0.8.0
- onnxruntime >= 1.15.0
- numpy >= 1.24.0
- transformers >= 4.30.0
- arabert >= 1.0.1
- pyarabic >= 0.6.15

### Verify Installation

```python
python -c "from livekit.plugins.arabic_turn_detector import ArabicTurnDetector; print('✓ Plugin installed')"
```

---

## Usage

### Basic Integration

Minimal example to get started:

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector
from livekit.agents import AgentSession, JobContext

# Create turn detector
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

# Use in agent
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        turn_detector=turn_detector,
        stt=your_stt_provider,
        tts=your_tts_provider,
    )
    
    await session.start()
```

### Complete Example

Full agent with Deepgram STT and ElevenLabs TTS:

```python
import os
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, elevenlabs
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

# Configuration
MODEL_PATH = "./models/eou_model_quantized.onnx"
EOU_THRESHOLD = 0.7
ELEVENLABS_VOICE_ID = "your-arabic-voice-id"

async def entrypoint(ctx: JobContext):
    """Agent entrypoint"""
    await ctx.connect()
    
    # Create Arabic turn detector
    turn_detector = ArabicTurnDetector(
        model_path=MODEL_PATH,
        unlikely_threshold=EOU_THRESHOLD
    )
    
    # Configure STT (Speech-to-Text)
    stt = deepgram.STT(
        language="ar",
        model="nova-2",
    )
    
    # Configure TTS (Text-to-Speech)
    tts = elevenlabs.TTS(
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_multilingual_v2",
        language="ar",
    )
    
    # Create agent session
    session = AgentSession(
        stt=stt,
        tts=tts,
        turn_detector=turn_detector,
        
        # Endpointing configuration
        min_endpointing_delay=0.5,
        interrupt_speech_duration=0.3,
        interrupt_min_words=2,
        allow_interruptions=True,
    )
    
    # Define agent behavior
    @session.on("user_speech_committed")
    async def on_user_speech(text: str):
        # Your agent logic here
        response = f"سمعتك تقول: {text}"
        await session.say(response)
    
    # Start session
    await session.start()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Advanced Usage

#### Get Confidence Scores

```python
# Get EOU confidence for debugging
chat_ctx = [
    {"role": "user", "content": "مرحبا كيف حالك"},
    {"role": "assistant", "content": "أنا بخير شكرا"},
]

confidence = turn_detector.get_confidence(chat_ctx)
print(f"EOU confidence: {confidence:.4f}")
```

#### Manual Turn Detection

```python
# Manually check if utterance is complete
is_complete = await turn_detector.detect_turn(chat_ctx)
if is_complete:
    print("Utterance is complete, agent can respond")
else:
    print("User is still speaking, wait for more input")
```

#### Dynamic Threshold Adjustment

```python
class AdaptiveTurnDetector:
    def __init__(self, base_model_path: str):
        self.base_threshold = 0.7
        self.current_threshold = self.base_threshold
        self.detector = ArabicTurnDetector(
            model_path=base_model_path,
            unlikely_threshold=self.current_threshold
        )
    
    def adjust_for_context(self, context: str):
        """Adjust threshold based on conversation context"""
        if "urgent" in context.lower():
            self.current_threshold = 0.6  # Faster responses
        elif "careful" in context.lower():
            self.current_threshold = 0.85  # More careful
        else:
            self.current_threshold = self.base_threshold
        
        # Recreate detector with new threshold
        self.detector = ArabicTurnDetector(
            model_path=self.detector._runner.model_path,
            unlikely_threshold=self.current_threshold
        )
```

---

## Configuration

### Threshold Tuning

The `unlikely_threshold` parameter controls turn detection sensitivity:

| Threshold | Behavior | Use Case | Description |
|-----------|----------|----------|-------------|
| **0.5-0.6** | Very sensitive | Quick Q&A | Fast responses, may interrupt user |
| **0.7** | Balanced (default) | General conversation | Good for most use cases |
| **0.8-0.9** | Conservative | Customer service | Avoid interruptions, wait longer |

#### Examples

```python
# Fast responses (may interrupt)
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.6
)

# Balanced (recommended)
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

# Conservative (fewer interruptions)
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.85
)
```

### Environment Variables

```bash
# Model configuration
export ARABIC_EOU_MODEL_PATH="./models/eou_model_quantized.onnx"
export EOU_THRESHOLD="0.7"

# LiveKit credentials
export LIVEKIT_URL="wss://your-livekit-server.com"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"

# STT/TTS API keys
export DEEPGRAM_API_KEY="your-deepgram-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

---

## Model Performance

### Accuracy Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 90% | ✅ Excellent |
| **Precision (EOU)** | 0.90 | ✅ Excellent |
| **Recall (EOU)** | 0.93 | ✅ Excellent |
| **F1-Score (EOU)** | 0.92 | ✅ Excellent |
| **Inference Time** | 20-30ms | ✅ Real-time |
| **Model Size** | ~100MB | ✅ Compact |

### Confusion Matrix

```
           Predicted
           No EOU  EOU
Actual No  333     62   (84.3% correct)
       EOU 42      564  (93.1% correct)
```

**Analysis:**
- **High Recall (93%)**: Excellent at detecting true end-of-utterance
- **Good Precision (90%)**: Low false positive rate
- **Balanced**: Works well for both EOU and non-EOU cases

### Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency (CPU)** | 20-30ms | Quantized model |
| **Latency (GPU)** | 10-15ms | With CUDA |
| **Throughput** | ~700 samples/sec | CPU |
| **Memory Usage** | ~200MB | Runtime |
| **Model Size** | 100MB | Quantized |

---

## Architecture

### How It Works

```
User Speech → STT (Arabic) → Text Normalization (AraBERT)
                                        ↓
                                  Tokenization
                                        ↓
                                  ONNX Inference
                                        ↓
                              Confidence Score (0-1)
                                        ↓
                            Threshold Comparison
                                        ↓
                               EOU Decision (bool)
                                        ↓
                          Agent Response Trigger
```

### Class Hierarchy

```
_InferenceRunner (LiveKit base class)
    ↓
_EOURunnerAr (Internal runner)
    - initialize()
    - _normalize_text()
    - _format_chat_ctx()
    - run()
    ↓
ArabicTurnDetector (Public API)
    - detect_turn()
    - get_confidence()
```

### Key Components

1. **Text Normalization**: AraBERT preprocessing for Arabic text
2. **Tokenization**: AraBERT tokenizer with 512 max tokens
3. **ONNX Inference**: CPU-optimized with 2-4 threads
4. **Prediction**: Binary classification with confidence score
5. **Threshold Logic**: Configurable decision boundary

---

## Integration Guide

### Prerequisites

- Python 3.9+
- LiveKit server running
- Trained Arabic EOU model
- STT provider (Deepgram, Google, Azure)
- TTS provider (ElevenLabs, Google, Azure)

### Step-by-Step Integration

#### 1. Prepare Model

```bash
cd ../eou_model

# Train
python scripts/train.py --output_dir ./models/eou_model

# Convert
python scripts/convert_to_onnx.py \
    --model_path ./models/eou_model \
    --output_path ./models/eou_model.onnx

# Quantize
python scripts/quantize_model.py \
    --model_path ./models/eou_model.onnx \
    --output_path ./models/eou_model_quantized.onnx
```

#### 2. Install Plugin

```bash
cd ../livekit_plugins_arabic_turn_detector
pip install -e .
```

#### 3. Configure Agent

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

turn_detector = ArabicTurnDetector(
    model_path="../eou_model/models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)
```

#### 4. Test Integration

```bash
# Set environment
export LIVEKIT_URL="wss://your-server.com"
export LIVEKIT_API_KEY="your-key"

# Run example
python examples/simple_agent.py
```

#### 5. Deploy to Production

See [Deployment](#deployment) section below.

---

## Troubleshooting

### Common Issues

#### Issue: Model not found

**Error:**
```
RuntimeError: Model path not provided
```

**Solution:**
```python
# Ensure correct path
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx"  # Absolute or relative path
)

# Verify file exists
import os
assert os.path.exists("./models/eou_model_quantized.onnx"), "Model file not found"
```

#### Issue: AraBERT not installed

**Warning:**
```
arabert not installed. Falling back to basic Arabic normalization.
```

**Solution:**
```bash
pip install arabert pyarabic
```

#### Issue: Tokenizer not found

**Warning:**
```
Failed to load tokenizer
```

**Solution:**
Ensure tokenizer files are in the same directory as ONNX model:
```bash
ls -la models/
# Should show:
# - eou_model_quantized.onnx
# - config.json
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
```

#### Issue: Slow inference (>100ms)

**Solutions:**

1. **Use quantized model:**
   ```bash
   python ../eou_model/scripts/quantize_model.py \
       --model_path ./models/eou_model.onnx \
       --output_path ./models/eou_model_quantized.onnx
   ```

2. **Reduce sequence length:**
   Modify `MAX_HISTORY_TOKENS` in `arabic.py` from 512 to 256

3. **Use GPU:**
   Modify `arabic.py`:
   ```python
   providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
   ```

#### Issue: Too many interruptions

**Problem:** Agent interrupts user frequently

**Solution:** Increase threshold
```python
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.85  # Higher = less sensitive
)
```

#### Issue: Agent waits too long

**Problem:** Agent takes too long to respond

**Solution:** Decrease threshold
```python
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.6  # Lower = more sensitive
)
```

---

## Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy plugin and model
COPY livekit_plugins_arabic_turn_detector/ /app/plugin/
COPY eou_model/models/ /app/models/

# Install plugin
RUN pip install -e /app/plugin/

# Copy agent
COPY agent.py /app/

# Set environment
ENV ARABIC_EOU_MODEL_PATH=/app/models/eou_model_quantized.onnx
ENV EOU_THRESHOLD=0.7

# Run
CMD ["python", "agent.py"]
```

Build and run:

```bash
docker build -t arabic-voice-agent .
docker run -e LIVEKIT_URL=$LIVEKIT_URL \
           -e LIVEKIT_API_KEY=$LIVEKIT_API_KEY \
           -e LIVEKIT_API_SECRET=$LIVEKIT_API_SECRET \
           arabic-voice-agent
```

### Production Checklist

- [ ] Model quantized for performance
- [ ] Threshold tuned for use case
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Fallback mechanisms in place
- [ ] Load testing completed
- [ ] Documentation updated

---

## Comparison with Other Turn Detectors

| Feature | Arabic Turn Detector | LiveKit Default | Deepgram VAD |
|---------|---------------------|-----------------|--------------|
| **Language** | Arabic-optimized | English | Multilingual |
| **Accuracy** | 90% (Arabic) | 99.3% (English) | 85% |
| **Latency** | 20-30ms | 50-160ms | <50ms |
| **Model Size** | 100MB (quantized) | 396MB | Small |
| **Offline** | ✅ Yes | ✅ Yes | ❌ No (API) |
| **Cost** | Free | Free | Paid |
| **Customizable** | ✅ Yes | ❌ No | ❌ No |
| **Threshold** | Configurable | Fixed | Fixed |

**Advantages:**
- ✅ Optimized for Arabic language
- ✅ No external API calls
- ✅ Full control and customization
- ✅ Can fine-tune on your data
- ✅ Runs on-device (privacy)

---

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

---

## References

- **LiveKit Agents**: https://github.com/livekit/agents
- **AraBERT**: https://huggingface.co/aubmindlab/bert-base-arabertv2
- **ONNX Runtime**: https://github.com/microsoft/onnxruntime
- **EOU Model Module**: `../eou_model/README.md`

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [LiveKit documentation](https://docs.livekit.io/agents/)
3. Open an issue on GitHub

---

## License

Apache-2.0 License

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**Version:** 1.0.0  
**Last Updated:** December 11, 2025  
**Status:** ✅ Production Ready  
**Accuracy:** 90% | **F1-Score:** 0.92 | **Latency:** 20-30ms
