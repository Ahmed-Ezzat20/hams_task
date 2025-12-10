# Hugging Face Dataset Upload Guide

This guide provides instructions for uploading your finalized Arabic EOU dataset to the Hugging Face Hub.

---

## 🎯 **Goal**

Create a public dataset on Hugging Face with train, validation, and test splits, along with a comprehensive dataset card.

---

## 🚀 **Step 1: Prepare Your Files**

After running the data generation and splitting scripts, you should have the following files in your `datasets/arabic_eou_5k/` directory:

- `train.csv`
- `validation.csv`
- `test.csv`
- `report.txt`

---

## 🤗 **Step 2: Create a New Dataset on Hugging Face**

### **Option A: Web Interface (Recommended)**
1. Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
2. **Dataset name:** `your-username/arabic-eou-detection-5k`
3. **License:** `mit` (or your preferred license)
4. **Public/Private:** Public
5. Click **Create dataset**

### **Option B: CLI**
```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli repo create your-username/arabic-eou-detection-5k --type dataset
```

---

## 📤 **Step 3: Upload Your Files**

### **Option A: Web Interface**
1. In your new dataset repository, click **Add file** > **Upload files**
2. Drag and drop `train.csv`, `validation.csv`, and `test.csv`
3. Commit the changes

### **Option B: CLI**
```bash
huggingface-cli upload your-username/arabic-eou-detection-5k datasets/arabic_eou_5k
```

---

## 📝 **Step 4: Create the Dataset Card (README.md)**

A good dataset card is essential for discoverability and usability. Create a `README.md` file in your dataset repository with the following content:

```markdown
---
license: mit
language:
- ar
- ar-SA
tags:
- audio-classification
- text-classification
- end-of-utterance
- saudi-arabic
- conversational-ai
pretty_name: "Arabic End-of-Utterance (EOU) Detection Dataset"
---

# Arabic End-of-Utterance (EOU) Detection Dataset

This dataset contains **5,000 synthetic Saudi Arabic utterances** designed for training End-of-Utterance (EOU) detection models. The data is generated using state-of-the-art language models and is formatted for text classification tasks.

## Dataset Structure

The dataset is provided in CSV format with three columns:

- `utterance`: The Arabic utterance text.
- `style`: The conversational style (`informal`, `formal`, or `asr_like`).
- `label`: The EOU label (`1` for end-of-utterance, `0` for not end-of-utterance).

## Splits

The dataset is divided into three splits:

| Split | Samples |
|---|---|
| **Train** | 3,500 |
| **Validation** | 750 |
| **Test** | 750 |

## Label Distribution

| Label | Percentage |
|---|---|
| **EOU (1)** | ~60% |
| **Non-EOU (0)** | ~40% |

## Style Distribution

| Style | Percentage |
|---|---|
| **Informal** | ~40% |
| **Formal** | ~40% |
| **ASR-like** | ~20% |

## Use Cases

This dataset is ideal for training text classification models for:
- Real-time EOU detection in voice agents
- Conversational AI turn-taking
- Spoken language understanding

## Generation Process

The data was generated using the `hams` data generation package, which uses a combination of few-shot prompting and style-controlled generation with large language models. The process ensures a balanced distribution of labels and styles, with high diversity and low redundancy.

## Curation and Quality

- **No Ellipsis Bias:** The dataset is free from punctuation crutches (e.g., "…") for non-EOU samples.
- **Diverse Patterns:** Non-EOU samples include a wide variety of semantic incompleteness patterns.
- **Low Duplicates:** The dataset has < 1% duplicate utterances.

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@misc{your-name-2025-arabic-eou,
  author = {Your Name},
  title = {Arabic End-of-Utterance (EOU) Detection Dataset},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/datasets/your-username/arabic-eou-detection-5k}}
}
```
```

---

This guide and template will help you create a high-quality, well-documented dataset on the Hugging Face Hub.
