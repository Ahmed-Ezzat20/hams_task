# Arabic End-of-Utterance Detection for LiveKit

Complete solution for detecting when Arabic speakers finish their utterances in real-time conversations.

## 🎯 Overview

This project provides a simple, modular pipeline for Arabic EOU detection:

1. **Dataset Generation** (`data_prep/`) - Generate Arabic conversational data
2. **Model Training** (`eou_model/`) - Train and optimize EOU detection model  
3. **LiveKit Plugin** (`plugin/`) - Deploy as voice agent

**Performance:**
- ✅ 90% accuracy, 0.92 F1-score
- ✅ 20-30ms inference latency
- ✅ 130MB model size (quantized)
- ✅ Saudi dialect emphasis

**Links:**
- 📊 **Dataset:** https://huggingface.co/datasets/MrEzzat/arabic-eou-detection-10k
- 🤖 **Model:** https://huggingface.co/MrEzzat/arabic-eou-detector

---

## 📁 Simple Structure

```
arabic-eou-detection/
│
├── data_prep/              # Generate dataset
│   ├── generate_dataset.py
│   └── prompts.yaml
│
├── eou_model/              # Train model
│   ├── train.py
│   ├── convert_to_onnx.py
│   ├── quantize_model.py
│   └── upload_to_huggingface.py
│
├── plugin/                 # LiveKit plugin
│   ├── arabic_turn_detector.py
│   └── agent.py
│
└── docs/                     # Documentation
    ├── HOW_TO_RUN.md
    └── USAGE_GUIDE.md
```

**No complex packages, just simple Python scripts!**

---

## 🚀 Quick Start

### Step 1: Generate Dataset

```bash
cd data_prep
pip install openai pyyaml
export OPENAI_API_KEY="your-key"

python generate_dataset.py --num-samples 10000 --split --output-dir ./data
```

### Step 2: Train Model

```bash
cd ../eou_model
pip install -r requirements.txt

python train.py --train_file ../data_prep/data/train.csv --output_dir ./models
python convert_to_onnx.py --model_path ./models/eou_model
python quantize_model.py --model_path ./models/eou_model.onnx
```

### Step 3: Run Agent

```bash
cd ../plugin
pip install -r requirements.txt

# Configure .env.local with API keys
python agent.py dev
```

Open http://localhost:8081 and speak Arabic!

---

## 📖 Documentation

- **Quick Start:** See above
- **Detailed Guide:** [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)
- **Configuration:** [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- **Technical Report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

Each directory has its own README:
- [data_prep/README.md](data_prep/README.md)
- [eou_model/README.md](eou_model/README.md)
- [plugin/README.md](plugin/README.md)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 90.0% |
| F1-Score (Complete) | 0.92 |
| Recall | 0.93 |
| Inference Time | 20-30ms |
| Model Size | 130MB (quantized) |

---

## 🎓 What's Different?

This refactored version is **simpler and more intuitive**:

✅ **No complex package structure** - Just simple Python scripts  
✅ **Clear separation** - Three main parts (data, model, plugin)  
✅ **No __init__.py files** - Not a package, easier to understand  
✅ **Standalone scripts** - Each can be run independently  
✅ **Better organization** - Logical workflow: data → model → plugin

---

## 🤝 Contributing

Contributions welcome! The simple structure makes it easy to:
- Add new dataset generation methods
- Experiment with different models
- Extend the LiveKit plugin

---

## 📄 License

MIT License

---

## 📞 Contact

**Ahmed Ezzat**  
- LinkedIn: https://eg.linkedin.com/in/mrezzat
- GitHub: https://github.com/Ahmed-Ezzat20

---

**Built with ❤️ for the Arabic NLP community**
