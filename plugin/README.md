# LiveKit Plugin

This directory contains the Arabic EOU detection plugin for LiveKit agents.

## Files

- **arabic_turn_detector.py** - Main plugin implementing Arabic EOU detection
- **agent.py** - Example LiveKit agent using the plugin
- **requirements.txt** - Plugin dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env.local` file in project root:

```bash
# LiveKit
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# ElevenLabs (Arabic STT)
ELEVENLABS_API_KEY=your-elevenlabs-key

# Google (Gemini LLM)
GOOGLE_API_KEY=your-google-key

# Cartesia (Arabic TTS) - Optional
CARTESIA_API_KEY=your-cartesia-key
```

### 3. Run Agent

```bash
# Development mode (console)
python agent.py dev

# Production mode
python agent.py start

# Connect to specific room
python agent.py connect --room my-room
```

## Usage

### Using the Plugin

```python
from arabic_turn_detector import ArabicEOUDetector
from livekit.agents import AgentSession
from livekit.plugins import elevenlabs, google, inference, silero

# Initialize detector
detector = ArabicEOUDetector(
    model_path="../eou_model/eou_model_quantized.onnx",
    confidence_threshold=0.7
)

# Create agent session
session = AgentSession(
    stt=elevenlabs.STT(language_code="ar", use_realtime=False),
    llm=google.LLM(model="models/gemini-2.5-flash-lite"),
    tts=inference.TTS(model="cartesia/sonic-3", language="ar"),
    turn_detection=detector,  # ← Arabic EOU detection
    vad=silero.VAD.load(),
)
```

### Configuration

#### ArabicEOUDetector Parameters

```python
detector = ArabicEOUDetector(
    model_path="path/to/model.onnx",     # Path to ONNX model
    confidence_threshold=0.7,             # EOU probability threshold
    min_utterance_length=3,               # Minimum tokens
    max_utterance_length=128,             # Maximum tokens
    log_predictions=True                  # Enable debug logging
)
```

#### Threshold Tuning

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.5 | Balanced | General conversations |
| 0.7 | Conservative | Avoid interruptions (recommended) |
| 0.9 | Very conservative | Formal settings |
| 0.3 | Aggressive | Fast-paced conversations |

## Architecture

### Plugin Interface

The plugin implements LiveKit's turn detection protocol:

```python
class ArabicEOUDetector:
    def supports_language(self, language: str) -> bool:
        """Check if language is supported (ar, ar-*)"""
        
    def unlikely_threshold(self, language: str) -> float:
        """Return threshold for unlikely EOU"""
        
    async def detect_turn(self, chat_ctx: ChatContext) -> float:
        """Detect EOU probability (0.0 to 1.0)"""
```

### Detection Flow

```
User speaks → STT → Text → EOU Detector → Probability
                                    ↓
                            Is probability > threshold?
                                    ↓
                        Yes → Agent responds
                        No  → Wait for more input
```

## Testing

### Test the Plugin

```python
# test_plugin.py
from arabic_turn_detector import ArabicEOUDetector
from livekit.agents import ChatContext, ChatMessage

detector = ArabicEOUDetector("../eou_model/eou_model_quantized.onnx")

# Test utterances
test_cases = [
    ("مرحباً، كيف حالك؟", True),      # Complete
    ("أنا أريد أن", False),            # Incomplete
    ("الحمد لله بخير", True),         # Complete
    ("هل يمكنك أن", False),            # Incomplete
]

for utterance, expected_complete in test_cases:
    # Create mock context
    ctx = ChatContext(messages=[ChatMessage(content=utterance, role="user")])
    
    # Detect
    probability = await detector.detect_turn(ctx)
    is_complete = probability > 0.7
    
    print(f"{utterance}: {probability:.3f} ({'✓' if is_complete == expected_complete else '✗'})")
```

### Run Test

```bash
python test_plugin.py
```

## Monitoring

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### View EOU Predictions

When running the agent, you'll see logs like:

```
DEBUG - EOU prediction: {
    'text': 'مرحباً، كيف حالك؟',
    'eou_probability': 0.92,
    'is_eou': True,
    'threshold': 0.7,
    'inference_time_ms': 25.3
}
```

## Performance

### Latency

- **Tokenization:** 2-3ms
- **ONNX Inference:** 20-30ms (CPU), 15-25ms (GPU)
- **Postprocessing:** 1-2ms
- **Total:** ~25-35ms

### Memory

- **Model:** 130MB
- **Inference buffers:** 50MB
- **Tokenizer:** 20MB
- **Total:** ~200MB

## Troubleshooting

### Model Not Found

```
FileNotFoundError: eou_model_quantized.onnx not found
```

**Solution:** Update `model_path` to correct location:
```python
detector = ArabicEOUDetector(
    model_path="../eou_model/eou_model_quantized.onnx"
)
```

### WebSocket Connection Error

```
ERROR - ElevenLabs WebSocket connection failed
```

**Solution:** Set `use_realtime=False` in STT:
```python
stt=elevenlabs.STT(language_code="ar", use_realtime=False)
```

### Low Detection Accuracy

**Solution:** Adjust threshold:
```python
detector = ArabicEOUDetector(
    model_path="...",
    confidence_threshold=0.6  # Lower threshold
)
```

### Agent Not Responding

**Check:**
1. VAD is detecting speech
2. STT is transcribing correctly
3. EOU probability exceeds threshold
4. LLM is generating responses

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Deployment

### Docker

```dockerfile
FROM python:3.11

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy plugin and model
COPY arabic_turn_detector.py .
COPY ../eou_model/eou_model_quantized.onnx ./models/

# Copy agent
COPY agent.py .

# Run
CMD ["python", "agent.py", "start"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arabic-eou-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: arabic-eou-agent:latest
        env:
        - name: LIVEKIT_URL
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## Next Steps

1. Test the plugin with different Arabic utterances
2. Tune the confidence threshold for your use case
3. Monitor performance in production
4. Collect user feedback
5. Fine-tune model with real conversation data

## Support

For issues or questions:
- Check the main README.md
- Review TECHNICAL_REPORT.md for methodology
- See docs/HOW_TO_RUN.md for detailed setup
