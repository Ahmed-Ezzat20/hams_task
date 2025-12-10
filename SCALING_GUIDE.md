# Production-Scale Dataset Generation Guide

This guide provides a comprehensive walkthrough for scaling your Arabic EOU dataset generation to a production-ready size (5,000-10,000 samples), creating proper train/validation/test splits, and uploading to Hugging Face.

---

## 🎯 **Target Dataset Size: 5,000 Samples**

We recommend a target of **5,000 high-quality samples** for your initial production dataset. This provides a strong balance of diversity, quality, and cost-effectiveness.

| Metric | 5,000 Samples |
|---|---|
| **Total Samples** | 5,000 |
| **EOU (label=1)** | ~3,000 (60%) |
| **Non-EOU (label=0)** | ~2,000 (40%) |
| **Train Split (70%)** | 3,500 samples |
| **Validation Split (15%)** | 750 samples |
| **Test Split (15%)** | 750 samples |
| **Estimated Cost** | ~$35-50 |
| **Estimated Time** | ~10-15 hours |

---

## 🚀 **Step 1: Automated Batch Generation (5,000 Samples)**

Use the automated batch generation script to create the full dataset. This script will:
- Generate 5,000 samples in 10 batches of 500 each
- Save to `data/batch_1.csv` through `data/batch_10.csv`
- Merge all batches into `data/arabic_eou_dataset_5000.csv`
- Provide progress and statistics

### **Commands:**

```bash
# Set your API key
export NEBIUS_API_KEY="your-nebius-api-key"

# Run the automated batch generation script
cd C:\Work\hams
./generate_production_dataset.sh
```

---

## 📊 **Step 2: Data Splitting and Validation**

After generation, use the data splitting script to create train/validation/test splits and validate the data quality.

### **Commands:**

```bash
# Run the data splitting and validation script
python -m hams.cli.split_dataset \
    --input-file data/arabic_eou_dataset_5000.csv \
    --output-dir datasets/arabic_eou_5k \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### **Expected Output:**
- `datasets/arabic_eou_5k/train.csv` (3,500 samples)
- `datasets/arabic_eou_5k/validation.csv` (750 samples)
- `datasets/arabic_eou_5k/test.csv` (750 samples)
- `datasets/arabic_eou_5k/report.txt` (validation report)

---

## 🤗 **Step 3: Upload to Hugging Face**

Upload your finalized dataset to the Hugging Face Hub.

### **Option A: Web Interface (Recommended)**
1. Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
2. Create a new dataset (e.g., `your-username/arabic-eou-detection-5k`)
3. Upload `train.csv`, `validation.csv`, and `test.csv`
4. Create a `README.md` (dataset card) with the provided template
5. Make the dataset public

### **Option B: CLI**
```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli upload your-username/arabic-eou-detection-5k datasets/arabic_eou_5k
```

---

## ⏰ **Time & Cost Estimates**

| Samples | Batches | Time | Cost |
|---|---|---|---|
| **1,000** | 2x500 | 2-3 hours | ~$7-10 |
| **5,000** | 10x500 | 10-15 hours | ~$35-50 |
| **10,000** | 20x500 | 20-30 hours | ~$70-100 |

---

## 💡 **Best Practices**

- **Run overnight:** Start the generation script and let it run overnight to save time.
- **Check logs:** Monitor the log files in the `logs/` directory for any errors.
- **Validate splits:** Review the `report.txt` to ensure the splits are balanced and diverse.
- **Start small:** Generate 1,000 samples first to verify everything works before scaling to 5,000.

---

This guide provides a clear path to creating a high-quality, production-scale dataset for your EOU detection model. Good luck!
