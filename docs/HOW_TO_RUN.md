# How to Run the Arabic Voice Agent

Complete step-by-step guide for running the agent on Windows.

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Pull Latest Code

```powershell
cd C:\Work\hams
git pull origin main
```

### Step 2: Activate Virtual Environment

```powershell
# If using the existing 'hams' environment
.\.venv\Scripts\Activate.ps1

# Or if you named it differently
.\hams\Scripts\Activate.ps1
```

**If you get execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install/Update Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Configure API Keys

```powershell
# Copy template if you don't have .env.local yet
Copy-Item .env.local.example .env.local

# Edit with your keys
notepad .env.local
```

**Add your API keys:**
```bash
LIVEKIT_URL=wss://your-server.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
ELEVENLABS_API_KEY=your_elevenlabs_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Step 5: Run the Agent

```powershell
python agent.py dev
```

**That's it!** 🎉

---

## 📋 Detailed Setup Guide

### Prerequisites

1. **Python 3.11+** installed
2. **Git** installed
3. **Virtual environment** created
4. **API Keys** from:
   - LiveKit (https://cloud.livekit.io/)
   - ElevenLabs (https://elevenlabs.io/)
   - Google AI Studio (https://aistudio.google.com/apikey)

---

## 🔧 Complete Setup (First Time)

### 1. Clone Repository (if not done)

```powershell
cd C:\Work
git clone https://github.com/Ahmed-Ezzat20/hams_task.git
cd hams_task
```

### 2. Create Virtual Environment

```powershell
# Create venv
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1
```

**Verify activation:**
```powershell
# You should see (.venv) in your prompt
(.venv) PS C:\Work\hams>
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:**
```
Installing collected packages: ...
Successfully installed livekit-agents-0.x.x ...
```

### 4. Get API Keys

#### LiveKit API Keys

1. Go to https://cloud.livekit.io/
2. Sign up / Log in
3. Create a new project
4. Go to **Settings** → **Keys**
5. Copy:
   - API Key
   - API Secret
   - WebSocket URL (wss://...)

#### ElevenLabs API Key

1. Go to https://elevenlabs.io/
2. Sign up / Log in
3. Go to **Profile** → **API Keys**
4. Click **Create API Key**
5. Copy the key

#### Google API Key

1. Go to https://aistudio.google.com/apikey
2. Sign in with Google account
3. Click **Create API Key**
4. Copy the key

### 5. Configure Environment

```powershell
# Copy template
Copy-Item .env.local.example .env.local

# Edit
notepad .env.local
```

**Fill in ALL keys:**
```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ElevenLabs API Key
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google AI API Key
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Save and close.**

### 6. Verify Model Files

```powershell
# Check if model exists
dir eou_model\models\

# You should see:
# - eou_model\ (directory)
# - eou_model.onnx
# - eou_model_quantized.onnx
```

**If models are missing:**
```powershell
# You need to train or download the model first
# See eou_model/README.md for instructions
```

---

## ▶️ Running the Agent

### Development Mode (Console Testing)

```powershell
python agent.py dev
```

**What happens:**
1. Agent starts in console mode
2. Loads VAD model
3. Loads Arabic EOU detector
4. Starts HTTP server
5. Waits for connections

**Expected output:**
```
INFO - Starting console mode 🚀
INFO - Prewarming models...
INFO - ✓ VAD model loaded
DEBUG - Initialized Arabic EOU Runner (threshold=0.7)
DEBUG - ✓ ONNX model loaded
INFO - ✓ Arabic turn detector loaded
INFO - starting worker
INFO - HTTP server listening on :XXXXX
```

### Production Mode

```powershell
python agent.py start
```

**Difference from dev mode:**
- More robust error handling
- Production logging
- Auto-restart on errors

### Connect Mode (Join Specific Room)

```powershell
python agent.py connect --room <room_name>
```

---

## 🎯 Testing the Agent

### 1. Check Logs

After running `python agent.py dev`, you should see:

```
✓ VAD model loaded
✓ Arabic turn detector loaded
HTTP server listening
```

### 2. Connect from LiveKit Client

Use LiveKit's web client or mobile app to connect to your room.

### 3. Speak in Arabic

Try these phrases:

| Arabic | English | Expected EOU |
|--------|---------|--------------|
| مرحبا | Hello | Low (incomplete) |
| مرحبا كيف حالك؟ | Hello, how are you? | High (complete) |
| أنا بخير | I'm fine | Low (incomplete) |
| أنا بخير والحمد لله | I'm fine, thank God | High (complete) |

### 4. Monitor EOU Logs

You should see:

```
DEBUG - eou prediction
DEBUG - {
    'eou_probability': 0.92,
    'is_eou': True,
    'threshold': 0.7,
    'duration': 0.025,
    'input': 'مرحبا كيف حالك'
}
```

**Understanding the log:**
- `eou_probability`: 0.92 = 92% confident it's end of utterance
- `is_eou`: True = Detected as complete
- `threshold`: 0.7 = Your configured threshold
- `duration`: 0.025 = 25ms inference time
- `input`: The Arabic text analyzed

---

## 🔧 Configuration Options

### Adjust EOU Threshold

Edit `agent.py` (around line 50):

```python
detector = ArabicEOUDetector(
    model_path="./eou_model/models/eou_model_quantized.onnx",
    confidence_threshold=0.7,  # ← Change this (0.5-0.9)
)
```

**Guidelines:**
- **0.5-0.6**: Very responsive (interrupts less, faster responses)
- **0.7**: Balanced (recommended)
- **0.8-0.9**: Conservative (waits longer, fewer interruptions)

### Adjust Endpointing Delay

Edit `agent.py` (around line 80):

```python
session = AgentSession(
    ...
    min_endpointing_delay=0.8,  # ← Change this (0.5-1.2 seconds)
    ...
)
```

**Guidelines:**
- **0.5s**: Fast-paced conversations
- **0.8s**: Normal (recommended)
- **1.2s**: Formal, slower conversations

### Change Voice

Edit `agent.py` (around line 90):

```python
tts=inference.TTS(
    model="cartesia/sonic-3",
    voice="79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e",  # ← Change voice ID
    language="ar",
)
```

**Find voice IDs:**
- Cartesia voices: https://docs.cartesia.ai/voices
- Or use ElevenLabs TTS instead

---

## 🐛 Troubleshooting

### Issue 1: "Module not found" Error

```
ModuleNotFoundError: No module named 'livekit'
```

**Solution:**
```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 2: "Model not found" Error

```
FileNotFoundError: eou_model/models/eou_model_quantized.onnx
```

**Solution:**
```powershell
# Check if model exists
dir eou_model\models\

# If missing, you need to:
# 1. Train the model (see eou_model/README.md)
# 2. Or download pre-trained model
# 3. Or use PyTorch model instead
```

**Quick fix - use PyTorch model:**

Edit `arabic_turn_detector_plugin.py`:
```python
# Change model_path to PyTorch model
model_path = "./eou_model/models/eou_model"
```

### Issue 3: No EOU Probability Logs

```
# Logs show agent running but no "eou prediction" messages
```

**Solution:**

Check if DEBUG logging is enabled in `agent.py`:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,  # ← Should be DEBUG, not INFO
    ...
)
```

### Issue 4: API Key Errors

```
AuthenticationError: Invalid API key
```

**Solution:**
```powershell
# Check .env.local exists
dir .env.local

# Verify keys are correct
notepad .env.local

# Common issues:
# - Extra spaces in keys
# - Missing quotes
# - Wrong key format
```

### Issue 5: WebSocket Connection Errors

```
APIStatusError: connection closed (status_code=-1)
```

**Solution:**

This is already fixed in the code. Verify `agent.py` has:

```python
stt=elevenlabs.STT(
    language_code="ar",
    use_realtime=False,  # ← Must be False!
),
```

### Issue 6: Slow Inference (>100ms)

```
DEBUG - {'duration': 0.150, ...}  # Too slow!
```

**Solution:**

1. **Use quantized model:**
   ```python
   model_path="./eou_model/models/eou_model_quantized.onnx"
   ```

2. **Check CPU usage:**
   ```powershell
   # Open Task Manager
   # Check if CPU is maxed out
   ```

3. **Close other applications**

### Issue 7: Agent Crashes

```
Error: ...
Process finished with exit code 1
```

**Solution:**

1. **Check logs** for error message
2. **Verify all API keys** are valid
3. **Check internet connection**
4. **Restart agent:**
   ```powershell
   # Stop: Ctrl+C
   # Start: python agent.py dev
   ```

---

## 📊 Performance Monitoring

### Check Inference Speed

Look for `duration` in logs:

```
DEBUG - {'duration': 0.025, ...}  # 25ms - Good!
DEBUG - {'duration': 0.150, ...}  # 150ms - Too slow
```

**Good performance:**
- Duration: 20-30ms
- EOU probability: 0.0-1.0 (varies by input)

**Poor performance:**
- Duration: >100ms
- Frequent errors

### Monitor Accuracy

Watch for false positives/negatives:

**False Positive** (detected EOU when incomplete):
```
Input: "مرحبا"  # Just "hello"
is_eou: True  # ❌ Should be False
```

**False Negative** (missed EOU when complete):
```
Input: "مرحبا كيف حالك؟"  # Complete question
is_eou: False  # ❌ Should be True
```

**If accuracy is poor:**
- Adjust threshold
- Retrain model with more data
- Check input text quality

---

## 🎛️ Advanced Usage

### Custom Model Path

```python
detector = ArabicEOUDetector(
    model_path="C:\\path\\to\\your\\model.onnx",
    confidence_threshold=0.7
)
```

### Multiple Agents

Run multiple agents on different ports:

```powershell
# Terminal 1
python agent.py dev

# Terminal 2 (different port)
$env:PORT=8081
python agent.py dev
```

### Production Deployment

For production, use:

```powershell
# Install production server
pip install gunicorn

# Run with gunicorn (Linux/Mac)
gunicorn agent:app

# Or use Docker (recommended)
docker build -t arabic-agent .
docker run -p 8080:8080 arabic-agent
```

---

## 📝 Command Reference

### Agent Commands

```powershell
# Development mode (console)
python agent.py dev

# Production mode
python agent.py start

# Connect to specific room
python agent.py connect --room my-room

# Help
python agent.py --help
```

### Environment Management

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Deactivate
deactivate

# Update dependencies
pip install --upgrade -r requirements.txt

# Check installed packages
pip list
```

### Git Commands

```powershell
# Pull latest
git pull origin main

# Check status
git status

# View changes
git log --oneline -5
```

---

## ✅ Success Checklist

Before running the agent, verify:

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list | findstr livekit`)
- [ ] `.env.local` exists with all API keys
- [ ] Model files exist (`dir eou_model\models\`)
- [ ] No errors in previous runs

When agent is running:

- [ ] "VAD model loaded" message appears
- [ ] "Arabic turn detector loaded" message appears
- [ ] "HTTP server listening" message appears
- [ ] No error messages in logs
- [ ] EOU prediction logs appear when speaking

---

## 🎯 Quick Reference

### Start Agent (Most Common)

```powershell
cd C:\Work\hams
.\.venv\Scripts\Activate.ps1
python agent.py dev
```

### Stop Agent

```
Ctrl+C
```

### Check Logs

Logs appear in terminal in real-time.

### Update Code

```powershell
git pull origin main
pip install -r requirements.txt
python agent.py dev
```

---

## 📞 Getting Help

### Check Documentation

1. **USAGE_GUIDE.md** - Complete usage guide
2. **eou_model/README.md** - Model training guide
3. **README.md** - Project overview

### Common Issues

- API key errors → Check `.env.local`
- Model not found → Check `eou_model/models/`
- No logs → Check DEBUG logging enabled
- Slow inference → Use quantized model

### Report Issues

GitHub Issues: https://github.com/Ahmed-Ezzat20/hams_task/issues

---

## 🎉 You're Ready!

**To run the agent:**

```powershell
cd C:\Work\hams
.\.venv\Scripts\Activate.ps1
python agent.py dev
```

**Expected output:**
```
✓ VAD model loaded
✓ Arabic turn detector loaded
HTTP server listening
```

**Now speak Arabic and watch the EOU predictions!** 🚀
