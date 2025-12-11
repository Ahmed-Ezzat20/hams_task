# Arabic Turn Detector Integration Guide

Complete guide for integrating the Arabic Turn Detector plugin with your LiveKit agent.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Model Preparation](#model-preparation)
4. [Basic Integration](#basic-integration)
5. [Advanced Configuration](#advanced-configuration)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- Python 3.9+
- LiveKit server (cloud or self-hosted)
- Trained Arabic EOU model (see `eou_model/` directory)
- STT provider (Deepgram, Google, Azure, etc.)
- TTS provider (ElevenLabs, Google, Azure, etc.)

### Environment Variables

```bash
# LiveKit credentials
export LIVEKIT_URL="wss://your-livekit-server.com"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"

# Model path
export ARABIC_EOU_MODEL_PATH="./models/eou_model_quantized.onnx"

# Optional: Threshold tuning
export EOU_THRESHOLD="0.7"

# STT/TTS API keys
export DEEPGRAM_API_KEY="your-deepgram-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

---

## Installation

### Step 1: Install the Plugin

```bash
cd livekit_plugins_arabic_turn_detector
pip install -e .
```

### Step 2: Verify Installation

```python
python -c "from livekit.plugins.arabic_turn_detector import ArabicTurnDetector; print('✓ Plugin installed')"
```

---

## Model Preparation

### Step 1: Train the Model

```bash
cd ../eou_model

# Train model
python scripts/train.py \
    --model_name "aubmindlab/bert-base-arabertv2" \
    --dataset_name "arabic-eou-detection-10k" \
    --output_dir "./models/eou_model" \
    --num_epochs 10
```

### Step 2: Convert to ONNX

```bash
# Convert to ONNX
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx"
```

### Step 3: Quantize for Production

```bash
# Quantize model (75% size reduction, 2-3x faster)
python scripts/quantize_model.py \
    --model_path "./models/eou_model.onnx" \
    --output_path "./models/eou_model_quantized.onnx" \
    --tokenizer_path "./models/eou_model"
```

### Step 4: Verify Model Files

```bash
ls -lh models/
# Should show:
# - eou_model_quantized.onnx (~100MB)
# - config.json
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
```

---

## Basic Integration

### Minimal Example

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
        stt=your_stt_provider,
        tts=your_tts_provider,
        turn_detector=turn_detector,  # Use Arabic turn detector
    )
    
    await session.start()
```

### Complete Example with STT/TTS

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
        min_endpointing_delay=0.5,
        interrupt_speech_duration=0.3,
        interrupt_min_words=2,
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

---

## Advanced Configuration

### Threshold Tuning

Different thresholds for different scenarios:

```python
# Customer service (avoid interruptions)
turn_detector = ArabicTurnDetector(
    model_path=MODEL_PATH,
    unlikely_threshold=0.85  # High threshold = wait longer
)

# Quick Q&A (fast responses)
turn_detector = ArabicTurnDetector(
    model_path=MODEL_PATH,
    unlikely_threshold=0.6  # Low threshold = respond faster
)

# Balanced (default)
turn_detector = ArabicTurnDetector(
    model_path=MODEL_PATH,
    unlikely_threshold=0.7  # Balanced
)
```

### Dynamic Threshold Adjustment

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

### Confidence Monitoring

```python
# Monitor EOU confidence for debugging
@session.on("user_speech_interim")
async def on_interim_speech(text: str):
    chat_ctx = [{"role": "user", "content": text}]
    confidence = turn_detector.get_confidence(chat_ctx)
    
    if confidence > 0.9:
        logger.info(f"High confidence EOU: {confidence:.4f}")
    elif confidence < 0.3:
        logger.info(f"Low confidence (user still speaking): {confidence:.4f}")
```

### Fallback to VAD

```python
# Combine with VAD for robustness
from livekit.plugins import silero

class HybridTurnDetector:
    def __init__(self, arabic_model_path: str):
        self.arabic_detector = ArabicTurnDetector(
            model_path=arabic_model_path,
            unlikely_threshold=0.7
        )
        self.vad = silero.VAD()
    
    async def detect_turn(self, chat_ctx: list, audio_data: bytes) -> bool:
        # Check Arabic EOU model
        eou_detected = await self.arabic_detector.detect_turn(chat_ctx)
        
        # Check VAD
        vad_detected = await self.vad.detect_silence(audio_data)
        
        # Both must agree
        return eou_detected and vad_detected
```

---

## Testing

### Unit Tests

```python
import pytest
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

@pytest.fixture
def turn_detector():
    return ArabicTurnDetector(
        model_path="./models/eou_model_quantized.onnx",
        unlikely_threshold=0.7
    )

@pytest.mark.asyncio
async def test_complete_utterance(turn_detector):
    chat_ctx = [
        {"role": "user", "content": "مرحبا كيف حالك"}
    ]
    is_complete = await turn_detector.detect_turn(chat_ctx)
    assert is_complete == True

@pytest.mark.asyncio
async def test_incomplete_utterance(turn_detector):
    chat_ctx = [
        {"role": "user", "content": "أنا أريد أن"}
    ]
    is_complete = await turn_detector.detect_turn(chat_ctx)
    assert is_complete == False

def test_confidence_score(turn_detector):
    chat_ctx = [
        {"role": "user", "content": "شكرا جزيلا"}
    ]
    confidence = turn_detector.get_confidence(chat_ctx)
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.7  # Should be high for complete utterance
```

### Integration Tests

```bash
# Run example agent
python examples/simple_agent.py

# In another terminal, connect to the room and test:
# 1. Say: "مرحبا" (should detect EOU and respond)
# 2. Say: "أنا أريد أن..." (should wait for more)
# 3. Say: "شكرا" (should detect EOU and respond)
```

### Performance Testing

```python
import time
import numpy as np

def benchmark_inference(turn_detector, num_iterations=100):
    """Benchmark inference speed"""
    chat_ctx = [{"role": "user", "content": "مرحبا كيف حالك"}]
    
    times = []
    for _ in range(num_iterations):
        start = time.time()
        turn_detector.get_confidence(chat_ctx)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    print(f"Average inference time: {np.mean(times):.2f}ms")
    print(f"Median inference time: {np.median(times):.2f}ms")
    print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
    print(f"99th percentile: {np.percentile(times, 99):.2f}ms")

# Run benchmark
turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)
benchmark_inference(turn_detector)
```

---

## Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy plugin
COPY livekit_plugins_arabic_turn_detector/ /app/livekit_plugins_arabic_turn_detector/
COPY eou_model/models/ /app/models/

# Install plugin
RUN pip install -e /app/livekit_plugins_arabic_turn_detector/

# Copy agent code
COPY agent.py /app/

# Set environment variables
ENV ARABIC_EOU_MODEL_PATH=/app/models/eou_model_quantized.onnx
ENV EOU_THRESHOLD=0.7

# Run agent
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
- [ ] Threshold tuned for your use case
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Fallback mechanisms in place
- [ ] Load testing completed
- [ ] Documentation updated

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails

**Symptoms:**
```
RuntimeError: Model path not provided
```

**Solutions:**
- Check model path exists: `ls -la $ARABIC_EOU_MODEL_PATH`
- Verify model format: `file $ARABIC_EOU_MODEL_PATH` (should be ONNX)
- Check permissions: `chmod 644 $ARABIC_EOU_MODEL_PATH`

#### 2. Slow Inference

**Symptoms:**
- Inference time > 100ms
- Agent feels laggy

**Solutions:**
- Use quantized model (75% faster)
- Check CPU usage: `top` or `htop`
- Reduce `MAX_HISTORY_TOKENS` in code
- Consider GPU inference (modify `arabic.py`)

#### 3. Too Many Interruptions

**Symptoms:**
- Agent interrupts user frequently
- Conversations feel rushed

**Solutions:**
- Increase threshold: `unlikely_threshold=0.8`
- Increase `min_endpointing_delay` in AgentSession
- Increase `interrupt_speech_duration`

#### 4. Agent Waits Too Long

**Symptoms:**
- Long pauses before agent responds
- Conversations feel slow

**Solutions:**
- Decrease threshold: `unlikely_threshold=0.6`
- Decrease `min_endpointing_delay`
- Check STT latency

#### 5. AraBERT Not Available

**Symptoms:**
```
arabert not installed. Falling back to basic Arabic normalization.
```

**Solutions:**
- Install arabert: `pip install arabert`
- Verify installation: `python -c "import arabert; print('OK')"`
- If still failing, check dependencies: `pip install pyarabic farasapy`

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("livekit.plugins.arabic_turn_detector")
logger.setLevel(logging.DEBUG)
```

### Performance Monitoring

```python
import time

class MonitoredTurnDetector:
    def __init__(self, base_detector):
        self.detector = base_detector
        self.inference_times = []
    
    async def detect_turn(self, chat_ctx):
        start = time.time()
        result = await self.detector.detect_turn(chat_ctx)
        elapsed = (time.time() - start) * 1000
        
        self.inference_times.append(elapsed)
        
        if elapsed > 50:
            logger.warning(f"Slow inference: {elapsed:.2f}ms")
        
        return result
    
    def get_stats(self):
        if not self.inference_times:
            return {}
        
        return {
            "avg": np.mean(self.inference_times),
            "median": np.median(self.inference_times),
            "p95": np.percentile(self.inference_times, 95),
            "p99": np.percentile(self.inference_times, 99),
            "max": max(self.inference_times),
        }
```

---

## Next Steps

1. **Optimize threshold** for your specific use case
2. **Collect metrics** on turn detection accuracy
3. **Fine-tune model** with your own conversation data
4. **Implement monitoring** for production deployment
5. **Add error recovery** mechanisms
6. **Scale horizontally** with multiple agent instances

---

## Support

For additional help:
- Check the [README.md](README.md)
- Review [examples/](examples/)
- Open an issue on GitHub
- Check LiveKit documentation

---

**Last Updated:** December 11, 2025  
**Version:** 0.1.0
