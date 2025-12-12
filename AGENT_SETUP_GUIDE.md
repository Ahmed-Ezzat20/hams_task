# Arabic Voice Agent Setup Guide

**Complete guide for running the LiveKit agent with custom Arabic turn detection.**

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Agent](#running-the-agent)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

This agent integrates:

✅ **Custom Arabic Turn Detection** - Your trained EOU model  
✅ **Arabic STT** - ElevenLabs with Arabic support  
✅ **Multilingual LLM** - Google Gemini 2.5 Flash  
✅ **High-Quality TTS** - Cartesia Sonic  
✅ **Voice Activity Detection** - Silero VAD  
✅ **Noise Cancellation** - BVC/BVC Telephony  

---

## Prerequisites

### 1. Python Environment

```bash
python --version  # Python 3.9 or higher
```

### 2. Required API Keys

- **LiveKit**: Server URL, API Key, API Secret
- **ElevenLabs**: API Key (for Arabic STT)
- **Google**: API Key (for Gemini LLM)

### 3. Arabic EOU Model

You need the quantized ONNX model:
```bash
ls eou_model/models/eou_model_quantized.onnx
```

If you don't have it yet:
```bash
cd eou_model

# Convert to ONNX
python scripts/convert_to_onnx.py \
    --model_path ./models/eou_model \
    --output_path ./models/eou_model.onnx

# Quantize
python scripts/quantize_model.py \
    --model_path ./models/eou_model.onnx \
    --output_path ./models/eou_model_quantized.onnx
```

---

## Installation

### Step 1: Install Dependencies

```bash
# Core LiveKit packages
pip install livekit-agents livekit-plugins-silero

# STT provider
pip install livekit-plugins-elevenlabs

# LLM provider
pip install livekit-plugins-google

# Noise cancellation
pip install livekit-plugins-noise-cancellation

# For ONNX model inference
pip install onnxruntime transformers

# Environment variables
pip install python-dotenv
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### Step 2: Install Arabic Turn Detector Plugin

```bash
# Install the custom plugin
cd livekit_plugins_arabic_turn_detector
pip install -e .

# Verify installation
python -c "from livekit.plugins.arabic_turn_detector import ArabicTurnDetector; print('✓ Plugin installed')"
```

---

## Configuration

### Step 1: Copy Environment Template

```bash
cp .env.local.example .env.local
```

### Step 2: Edit Configuration

Edit `.env.local`:

```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Google Configuration
GOOGLE_API_KEY=your_google_api_key

# Arabic EOU Model Path
ARABIC_EOU_MODEL_PATH=./eou_model/models/eou_model_quantized.onnx

# Turn Detector Threshold (optional)
ARABIC_EOU_THRESHOLD=0.7
```

### Step 3: Verify Model Path

```bash
# Check model exists
ls -lh ./eou_model/models/eou_model_quantized.onnx

# Should show ~130 MB file
```

---

## Running the Agent

### Development Mode

```bash
# Run with auto-reload
python agent_with_arabic_turn_detector.py dev
```

### Production Mode

```bash
# Run in production
python agent_with_arabic_turn_detector.py start
```

### With Custom Configuration

```bash
# Specify custom env file
python agent_with_arabic_turn_detector.py start --env-file .env.production
```

### Expected Output

```
INFO:agent:Prewarming models...
INFO:agent:✓ VAD model loaded
INFO:agent:✓ Arabic turn detector loaded from ./eou_model/models/eou_model_quantized.onnx
INFO:livekit:Agent server started
INFO:livekit:Listening on port 8080
```

---

## Testing

### Test 1: Verify Agent Starts

```bash
python agent_with_arabic_turn_detector.py start
```

**Expected:**
- No errors
- "Agent server started" message
- "Arabic turn detector loaded" message

### Test 2: Test Turn Detector Directly

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

# Load detector
detector = ArabicTurnDetector(
    model_path="./eou_model/models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

# Test with Arabic text
text = "مرحبا كيف حالك"
is_eou, confidence = detector.predict(text)
print(f"Is EOU: {is_eou}, Confidence: {confidence:.4f}")
```

### Test 3: Test in LiveKit Room

1. Create a test room in LiveKit
2. Join the room with a client
3. Speak in Arabic
4. Verify agent responds correctly
5. Check turn detection timing

---

## Troubleshooting

### Issue 1: Arabic Turn Detector Not Loading

**Error:**
```
WARNING:agent:ARABIC_EOU_MODEL_PATH not set or file not found
WARNING:agent:Will use default turn detector as fallback
```

**Solution:**
```bash
# Check environment variable
echo $ARABIC_EOU_MODEL_PATH

# Check file exists
ls -la ./eou_model/models/eou_model_quantized.onnx

# Set correct path in .env.local
ARABIC_EOU_MODEL_PATH=./eou_model/models/eou_model_quantized.onnx
```

### Issue 2: ElevenLabs Connection Issues

**Error:**
```
livekit.agents._exceptions.APIStatusError: connection closed
```

**Solution:**
Already fixed in the adapted agent! The agent uses `use_realtime=False` to avoid WebSocket issues.

If still having issues:
```python
# In agent_with_arabic_turn_detector.py, line 127
stt=elevenlabs.STT(
    language_code="ar",
    use_realtime=False,  # ← Already set to False
)
```

### Issue 3: Import Error for Arabic Turn Detector

**Error:**
```
ModuleNotFoundError: No module named 'livekit.plugins.arabic_turn_detector'
```

**Solution:**
```bash
# Install the plugin
cd livekit_plugins_arabic_turn_detector
pip install -e .

# Verify
python -c "from livekit.plugins.arabic_turn_detector import ArabicTurnDetector"
```

### Issue 4: ONNX Runtime Error

**Error:**
```
Error loading ONNX model
```

**Solution:**
```bash
# Install/upgrade onnxruntime
pip install --upgrade onnxruntime

# Verify model file
python -c "import onnxruntime as ort; ort.InferenceSession('./eou_model/models/eou_model_quantized.onnx')"
```

### Issue 5: Agent Not Detecting Turn Properly

**Problem:** Agent interrupts too early or waits too long

**Solution:** Adjust the threshold in `.env.local`:

```bash
# More sensitive (detects EOU earlier)
ARABIC_EOU_THRESHOLD=0.5

# Less sensitive (waits longer)
ARABIC_EOU_THRESHOLD=0.9

# Default (balanced)
ARABIC_EOU_THRESHOLD=0.7
```

Or modify in code:
```python
# In agent_with_arabic_turn_detector.py, line 94
proc.userdata["arabic_turn_detector"] = ArabicTurnDetector(
    model_path=model_path,
    unlikely_threshold=0.7,  # ← Adjust this value
)
```

---

## Advanced Configuration

### 1. Using HuggingFace Model

If you uploaded your model to HuggingFace:

```python
# In agent_with_arabic_turn_detector.py, add at top:
from huggingface_hub import hf_hub_download

# In prewarm function:
def prewarm(proc: JobProcess):
    logger.info("Prewarming models...")
    
    # Load VAD
    proc.userdata["vad"] = silero.VAD.load()
    
    # Download model from HuggingFace
    model_path = hf_hub_download(
        repo_id="your-username/arabic-eou-detector",
        filename="model_quantized.onnx",
        cache_dir="./models"
    )
    
    # Load Arabic turn detector
    proc.userdata["arabic_turn_detector"] = ArabicTurnDetector(
        model_path=model_path,
        unlikely_threshold=0.7,
    )
```

### 2. Dynamic Threshold Adjustment

```python
# In agent_with_arabic_turn_detector.py
class ArabicAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(...)
        self.eou_threshold = 0.7
    
    async def on_message(self, message):
        # Adjust threshold based on conversation context
        if "urgent" in message.lower():
            self.eou_threshold = 0.5  # More sensitive
        else:
            self.eou_threshold = 0.7  # Normal
```

### 3. Logging Configuration

```python
# Add at top of agent_with_arabic_turn_detector.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
```

### 4. Multiple Language Support

```python
# Detect language and use appropriate turn detector
async def arabic_agent(ctx: JobContext):
    # ... existing code ...
    
    # Detect language from room metadata
    language = ctx.room.metadata.get("language", "ar")
    
    if language == "ar":
        turn_detector = ctx.proc.userdata.get("arabic_turn_detector")
    else:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel
        turn_detector = MultilingualModel()
    
    session = AgentSession(
        turn_detection=turn_detector,
        # ... rest of config ...
    )
```

### 5. Performance Monitoring

```python
# Add metrics tracking
import time

class ArabicAssistant(Agent):
    def __init__(self):
        super().__init__(...)
        self.turn_detection_times = []
    
    async def on_turn_detected(self):
        start_time = time.time()
        # ... process turn ...
        elapsed = time.time() - start_time
        self.turn_detection_times.append(elapsed)
        
        # Log average
        if len(self.turn_detection_times) % 10 == 0:
            avg_time = sum(self.turn_detection_times) / len(self.turn_detection_times)
            logger.info(f"Average turn detection time: {avg_time:.3f}s")
```

---

## Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install Arabic turn detector plugin
RUN cd livekit_plugins_arabic_turn_detector && pip install -e .

# Run agent
CMD ["python", "agent_with_arabic_turn_detector.py", "start"]
```

Build and run:
```bash
docker build -t arabic-voice-agent .
docker run -d --env-file .env.local arabic-voice-agent
```

### Kubernetes Deployment

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arabic-voice-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arabic-voice-agent
  template:
    metadata:
      labels:
        app: arabic-voice-agent
    spec:
      containers:
      - name: agent
        image: arabic-voice-agent:latest
        envFrom:
        - secretRef:
            name: agent-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Performance Tips

### 1. Model Optimization

- ✅ Use quantized model (already done)
- ✅ Cache model in memory (already done in prewarm)
- ✅ Use ONNX Runtime (already done)

### 2. Agent Configuration

```python
session = AgentSession(
    # ... existing config ...
    preemptive_generation=True,  # ← Enable for faster responses
    max_concurrent_tasks=5,      # ← Adjust based on your server
)
```

### 3. Resource Allocation

- **CPU**: 2 cores minimum
- **Memory**: 2GB minimum (for model + agent)
- **Network**: Low latency connection to LiveKit server

---

## Summary

### ✅ What Changed from Original Agent

| Feature | Original | Adapted |
|---------|----------|---------|
| **Turn Detection** | MultilingualModel | ArabicTurnDetector |
| **Language** | English | Arabic + English |
| **STT** | English | Arabic (ElevenLabs) |
| **use_realtime** | True | False (fixed issue) |
| **Instructions** | English only | Bilingual |
| **Model Loading** | None | Prewarm with ONNX |

### 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Arabic turn detector
cd livekit_plugins_arabic_turn_detector && pip install -e . && cd ..

# 3. Configure environment
cp .env.local.example .env.local
# Edit .env.local with your API keys

# 4. Run agent
python agent_with_arabic_turn_detector.py start
```

### 📊 Expected Performance

- **Turn Detection Latency**: 20-30ms
- **Model Size**: 130 MB (quantized)
- **Memory Usage**: ~500 MB
- **Accuracy**: 90% (from training)

---

## Resources

- **LiveKit Agents**: https://docs.livekit.io/agents/
- **ElevenLabs Plugin**: https://docs.livekit.io/agents/plugins/elevenlabs/
- **Google Plugin**: https://docs.livekit.io/agents/plugins/google/
- **Turn Detection**: https://docs.livekit.io/agents/build/turns/

---

**Your agent is now ready to use with custom Arabic turn detection!** 🎉
