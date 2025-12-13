# Arabic Voice Agent & EOU Dataset Generator

Production-ready toolkit for Arabic voice AI with custom End-of-Utterance (EOU) detection and dataset generation.

## 🎯 What's Included

### 1. Arabic Voice Agent
LiveKit voice agent with custom Arabic EOU detection for natural conversation flow.

### 2. EOU Model Training Pipeline
Complete pipeline for training, converting, and deploying Arabic EOU detection models.

### 3. HAMS Dataset Generator
Toolkit for generating synthetic Arabic conversational data for EOU model training.

---

## 📊 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **EOU Model** | Accuracy | 90% |
| **EOU Model** | F1-Score | 0.92 |
| **EOU Model** | Inference Time | 20-30ms |
| **EOU Model** | Model Size | 130MB (quantized) |

---

## 🚀 Quick Start

### Option 1: Use Pre-trained Voice Agent

```bash
# 1. Clone repository
git clone https://github.com/Ahmed-Ezzat20/hams_task.git
cd hams_task

# 2. Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API keys
cp .env.local.example .env.local
# Edit .env.local with your API keys

# 4. Run agent
python agent.py dev
```

### Option 2: Generate Dataset & Train Model

```bash
# 1. Generate dataset
hams-build --num-conversations 1000 --output-dir data

# 2. Train EOU model
cd eou_model
python scripts/train.py --dataset_name "your-dataset" --output_dir "./models/eou_model"

# 3. Convert to ONNX and quantize
python scripts/convert_to_onnx.py --model_path "./models/eou_model" --output_path "./models/eou_model.onnx"
python scripts/quantize_model.py --model_path "./models/eou_model.onnx" --output_path "./models/eou_model_quantized.onnx"
```

---

## 📁 Project Structure

```
hams_task/
├── agent.py                              # Main voice agent
├── arabic_turn_detector_plugin.py        # EOU detector plugin
├── USAGE_GUIDE.md                        # Complete usage guide
├── requirements.txt                      # Dependencies
│
├── eou_model/                            # EOU Model Module
│   ├── README.md                         # Model documentation
│   ├── scripts/
│   │   ├── train.py                      # Training
│   │   ├── convert_to_onnx.py            # ONNX conversion
│   │   ├── quantize_model.py             # Quantization
│   │   └── upload_to_huggingface.py      # HF upload
│   └── models/                           # Trained models
│
└── hams/                                 # Dataset Generator
    ├── cli/                              # CLI tools
    ├── core/                             # Core modules
    ├── prompts.yaml                      # Conversation prompts
    └── tests/                            # Unit tests
```

---

## 📖 Documentation

### Voice Agent

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete voice agent guide
  - Installation (Windows, macOS, Linux)
  - Configuration
  - Running instructions
  - Performance tuning
  - Troubleshooting

### EOU Model

- **[eou_model/README.md](eou_model/README.md)** - Model training guide
  - Training pipeline
  - ONNX conversion
  - Model quantization
  - HuggingFace deployment

### Dataset Generation

See sections below for HAMS dataset generator usage.

---

## 🎙️ Voice Agent Features

- ✅ **Arabic Speech Recognition** - ElevenLabs STT
- ✅ **Custom EOU Detection** - 90% accuracy
- ✅ **ONNX Inference** - 2-3x faster (20-30ms)
- ✅ **Quantized Model** - 75% smaller
- ✅ **Google Gemini LLM** - Fast responses
- ✅ **Noise Cancellation** - Enhanced audio
- ✅ **Debug Logging** - Real-time monitoring

### Configuration

```bash
# .env.local
LIVEKIT_URL=wss://your-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
ELEVENLABS_API_KEY=your_elevenlabs_key
GOOGLE_API_KEY=your_google_key
```

### Usage Example

```python
from arabic_turn_detector_plugin import ArabicEOUDetector

detector = ArabicEOUDetector(
    model_path="./eou_model/models/eou_model_quantized.onnx",
    confidence_threshold=0.7
)
```

---

## 📊 HAMS: Dataset Generator

**HAMS** generates synthetic Arabic conversational data for EOU detection training.

### Features

- **Modular Architecture**: Clean, extensible modules
- **ASR-Style Degradation**: Simulate realistic ASR noise
- **YAML-based Prompts**: Easy to manage and extend
- **JSONL Output**: Streamable and mergeable
- **CLI Commands**: Simple entry points
- **Production-Ready**: Tested and documented

### Installation

```bash
# Install with dev dependencies
pip install -e .[dev]

# Or with uv
uv sync --dev

# Set API key
export NEBIUS_API_KEY="your-nebius-api-key"
```

### Quick Start

**Build a complete dataset:**

```bash
hams-build --num-conversations 100 --output-dir data
```

**Output:**
- `data/conversations_clean.jsonl`
- `data/conversations_asr.jsonl`

### CLI Commands

#### 1. `hams-generate`: Generate Conversations

```bash
# Generate 50 clean conversations
hams-generate \
    --num-conversations 50 \
    --output-file data/clean.jsonl \
    --style clean

# Generate 50 ASR-like conversations
hams-generate \
    --num-conversations 50 \
    --output-file data/asr.jsonl \
    --style asr_like

# Generate for specific domains
hams-generate \
    --num-conversations 20 \
    --output-file data/restaurant.jsonl \
    --domains restaurant hospitality
```

#### 2. `hams-asr-augment`: Add ASR Noise

```bash
hams-asr-augment \
    --input-file data/clean.jsonl \
    --output-file data/augmented.jsonl
```

#### 3. `hams-build`: Build Complete Datasets

```bash
# Build 200 conversations (clean + ASR)
hams-build \
    --num-conversations 200 \
    --output-dir datasets/eou_v1

# Build only clean conversations
hams-build \
    --num-conversations 100 \
    --output-dir datasets/eou_v1_clean \
    --style clean
```

### Architecture

```
[PromptBuilder] -> [ConversationGenerator] -> [PostProcessor] -> [DatasetWriter]
   (YAML)           (LLM API)             (Normalize)         (JSONL)
```

**4 Core Modules:**
1. **PromptBuilder**: Loads prompts from `prompts.yaml`
2. **ConversationGenerator**: Generates conversations via LLM
3. **PostProcessor**: Normalizes and applies ASR noise
4. **DatasetWriter**: Writes to JSONL files

### Prompts

Prompts are in `hams/prompts.yaml`:

```yaml
- id: restaurant_booking
  domain: restaurant
  description: "Customer booking a table"
  scenario: "..."
```

---

## 🔧 Complete Workflow

### 1. Generate Dataset

```bash
hams-build --num-conversations 1000 --output-dir datasets/arabic_eou
```

### 2. Train EOU Model

```bash
cd eou_model
python scripts/train.py \
    --dataset_name "../datasets/arabic_eou/conversations_clean.jsonl" \
    --output_dir "./models/eou_model"
```

### 3. Convert & Quantize

```bash
# Convert to ONNX
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx"

# Quantize (75% smaller)
python scripts/quantize_model.py \
    --model_path "./models/eou_model.onnx" \
    --output_path "./models/eou_model_quantized.onnx"
```

### 4. Deploy Voice Agent

```bash
cd ..
cp .env.local.example .env.local
# Edit .env.local with API keys
python agent.py start
```

---

## 🐛 Troubleshooting

### Voice Agent Issues

1. **"Model not found"**
   - Check: `ls eou_model/models/`
   - Verify path in `.env.local`

2. **No EOU logs**
   - DEBUG logging enabled by default
   - Check terminal output

3. **WebSocket errors**
   - Already fixed: `use_realtime=False`

### Dataset Generation Issues

1. **API key errors**
   - Set: `export NEBIUS_API_KEY="your-key"`

2. **Import errors**
   - Install: `pip install -e .[dev]`

See **[USAGE_GUIDE.md](USAGE_GUIDE.md)** for detailed troubleshooting.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest hams/tests/test_core.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Make changes
4. Commit (`git commit -m "feat: description"`)
5. Push (`git push origin feature/name`)
6. Open pull request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) file.

---

## 🙏 Credits

- **Model**: AraBERT v2 (aubmindlab/bert-base-arabertv2)
- **Framework**: LiveKit Agents
- **Author**: Ahmed Ezzat

---

## 📞 Support

- **Issues**: https://github.com/Ahmed-Ezzat20/hams_task/issues
- **Docs**: [USAGE_GUIDE.md](USAGE_GUIDE.md)

---

**Ready to start?**

- **Voice Agent**: `python agent.py dev`
- **Dataset**: `hams-build --num-conversations 100 --output-dir data`
- **Training**: See `eou_model/README.md`

🚀
