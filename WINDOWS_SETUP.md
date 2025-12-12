# Windows Setup Guide - Arabic Voice Agent

**Quick setup guide for running the agent on Windows.**

---

## Quick Start (Automated)

### Option 1: Using PowerShell Script (Recommended)

```powershell
# Run the automated setup script
.\setup_windows.ps1
```

This will:
1. Check Python installation
2. Create virtual environment
3. Install all dependencies
4. Install Arabic turn detector plugin
5. Create `.env.local` from template

---

## Manual Setup

### Step 1: Check Python

```powershell
python --version
# Should show Python 3.9 or higher
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv hams

# Activate it
.\hams\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 4: Install Arabic Turn Detector Plugin

```powershell
cd livekit_plugins_arabic_turn_detector
pip install -e .
cd ..
```

### Step 5: Configure Environment

```powershell
# Copy template
Copy-Item .env.local.example .env.local

# Edit with your API keys
notepad .env.local
```

**Required variables:**
```bash
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
ELEVENLABS_API_KEY=your_elevenlabs_api_key
GOOGLE_API_KEY=your_google_api_key
ARABIC_EOU_MODEL_PATH=.\eou_model\models\eou_model_quantized.onnx
```

### Step 6: Verify Model

```powershell
# Check model exists
Get-Item .\eou_model\models\eou_model_quantized.onnx

# Should show ~130 MB file
```

If model doesn't exist, train it first:
```powershell
cd eou_model

# Train model
python scripts\train.py --output_dir .\models\eou_model

# Convert to ONNX
python scripts\convert_to_onnx.py `
    --model_path .\models\eou_model `
    --output_path .\models\eou_model.onnx

# Quantize
python scripts\quantize_model.py `
    --model_path .\models\eou_model.onnx `
    --output_path .\models\eou_model_quantized.onnx

cd ..
```

### Step 7: Run Agent

```powershell
# Development mode (with auto-reload)
python agent_with_arabic_turn_detector.py dev

# Production mode
python agent_with_arabic_turn_detector.py start
```

---

## Troubleshooting

### Issue 1: "python: command not found"

**Solution:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart PowerShell

### Issue 2: "Execution Policy" Error

**Error:**
```
.\hams\Scripts\Activate.ps1 : File cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 3: "No module named 'livekit'"

**Error:**
```
ModuleNotFoundError: No module named 'livekit'
```

**Solution:**
```powershell
# Make sure virtual environment is activated
.\hams\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Issue 4: Virtual Environment Not Activating

**Solution:**

Try using Command Prompt instead of PowerShell:
```cmd
# Create virtual environment
python -m venv hams

# Activate (Command Prompt)
hams\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Issue 5: Path Issues with Model

**Error:**
```
FileNotFoundError: eou_model/models/eou_model_quantized.onnx
```

**Solution:**

Use Windows-style paths in `.env.local`:
```bash
# Use backslashes or forward slashes
ARABIC_EOU_MODEL_PATH=.\eou_model\models\eou_model_quantized.onnx
# OR
ARABIC_EOU_MODEL_PATH=./eou_model/models/eou_model_quantized.onnx
```

### Issue 6: SSL Certificate Errors

**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**
```powershell
# Install certifi
pip install --upgrade certifi

# Or disable SSL verification (not recommended for production)
$env:PYTHONHTTPSVERIFY=0
```

---

## Using UV (Alternative Package Manager)

If you're using `uv` instead of `pip`:

```powershell
# Create project
uv init

# Add dependencies
uv add livekit-agents livekit-plugins-silero livekit-plugins-elevenlabs livekit-plugins-google livekit-plugins-noise-cancellation onnxruntime transformers python-dotenv

# Install Arabic turn detector
cd livekit_plugins_arabic_turn_detector
uv pip install -e .
cd ..

# Run agent
uv run agent_with_arabic_turn_detector.py dev
```

---

## Verification

### Check Installation

```powershell
# Activate virtual environment
.\hams\Scripts\Activate.ps1

# Check Python packages
pip list | Select-String "livekit"

# Should show:
# livekit
# livekit-agents
# livekit-plugins-elevenlabs
# livekit-plugins-google
# livekit-plugins-noise-cancellation
# livekit-plugins-silero
```

### Test Arabic Turn Detector

```powershell
python -c "from livekit.plugins.arabic_turn_detector import ArabicTurnDetector; print('✓ Plugin installed')"
```

### Test Model Loading

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

detector = ArabicTurnDetector(
    model_path="./eou_model/models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

text = "مرحبا كيف حالك"
is_eou, confidence = detector.predict(text)
print(f"Is EOU: {is_eou}, Confidence: {confidence:.4f}")
```

---

## Running the Agent

### Development Mode

```powershell
# Activate virtual environment
.\hams\Scripts\Activate.ps1

# Run agent
python agent_with_arabic_turn_detector.py dev
```

**Expected output:**
```
INFO:agent:Prewarming models...
INFO:agent:✓ VAD model loaded
INFO:agent:✓ Arabic turn detector loaded from .\eou_model\models\eou_model_quantized.onnx
INFO:livekit:Agent server started
INFO:livekit:Listening on port 8080
```

### Production Mode

```powershell
python agent_with_arabic_turn_detector.py start
```

### Stop the Agent

Press `Ctrl+C` to stop the agent.

---

## Common Commands

### Activate Virtual Environment

```powershell
# PowerShell
.\hams\Scripts\Activate.ps1

# Command Prompt
hams\Scripts\activate.bat
```

### Deactivate Virtual Environment

```powershell
deactivate
```

### Update Dependencies

```powershell
pip install --upgrade -r requirements.txt
```

### Clean Installation

```powershell
# Remove virtual environment
Remove-Item -Recurse -Force hams

# Create new one
python -m venv hams
.\hams\Scripts\Activate.ps1
pip install -r requirements.txt
cd livekit_plugins_arabic_turn_detector
pip install -e .
cd ..
```

---

## Environment Variables

### Set Temporarily (Current Session)

```powershell
$env:LIVEKIT_URL="wss://your-server.com"
$env:ELEVENLABS_API_KEY="your_key"
```

### Set Permanently (User Level)

```powershell
[System.Environment]::SetEnvironmentVariable('LIVEKIT_URL', 'wss://your-server.com', 'User')
```

### Use .env.local (Recommended)

Just edit `.env.local` file - the agent will load it automatically.

---

## IDE Setup

### VS Code

1. Install Python extension
2. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose: `.\hams\Scripts\python.exe`
4. Create launch configuration (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Arabic Voice Agent",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/agent_with_arabic_turn_detector.py",
            "args": ["dev"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.local"
        }
    ]
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add interpreter → Existing environment
3. Select: `.\hams\Scripts\python.exe`
4. Run → Edit Configurations
5. Add environment file: `.env.local`

---

## Performance Tips

### Windows-Specific

1. **Disable Windows Defender** for the project folder (speeds up Python)
2. **Use SSD** for better model loading performance
3. **Close unnecessary apps** to free up memory
4. **Use PowerShell 7** (faster than PowerShell 5.1)

### Agent Configuration

In `.env.local`:
```bash
# Adjust based on your system
ARABIC_EOU_THRESHOLD=0.7  # Lower = more sensitive
```

---

## Summary

### Quick Setup

```powershell
# 1. Run automated setup
.\setup_windows.ps1

# 2. Edit configuration
notepad .env.local

# 3. Run agent
.\hams\Scripts\Activate.ps1
python agent_with_arabic_turn_detector.py dev
```

### Manual Setup

```powershell
# 1. Create virtual environment
python -m venv hams
.\hams\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
cd livekit_plugins_arabic_turn_detector
pip install -e .
cd ..

# 3. Configure
Copy-Item .env.local.example .env.local
notepad .env.local

# 4. Run
python agent_with_arabic_turn_detector.py dev
```

---

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. See `AGENT_SETUP_GUIDE.md` for detailed information
3. Check logs for error messages
4. Verify all API keys are correct in `.env.local`

---

**You're all set! Happy coding!** 🚀
