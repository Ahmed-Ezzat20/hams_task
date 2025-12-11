# Phase 2: Model Selection and Fine-Tuning Strategy

**Project:** HAMS - Arabic EOU Detection for LiveKit Voice Agents  
**Date:** December 10, 2025  
**Status:** Phase 1 Complete ✅ | Phase 2 Planning  

---

## Phase 1 Completion Summary ✅

### Achievements
- ✅ **Dataset Generated:** 10,000 high-quality Arabic EOU samples
- ✅ **Quality Score:** 85.8/100 (GOOD - Ready for training)
- ✅ **Dataset Uploaded:** [Hugging Face Dataset](https://huggingface.co/datasets/MrEzzat/arabic-eou-detection-10k)
- ✅ **Comprehensive Documentation:** Dataset card, analysis report, scaling guides
- ✅ **Zero Ellipsis Bias:** Eliminated punctuation crutches
- ✅ **Perfect Label Balance:** 60.55% EOU / 39.45% non-EOU

### Key Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total Samples | 10,000 | 10,000 | ✅ |
| Label Balance | 60.55% / 39.45% | 60% / 40% | ✅ |
| Ellipsis Bias | 0% | 0% | ✅ |
| Unique Last Words | 3,885 (38.85%) | High diversity | ✅ |
| Duplicates | 5.55% | <1% | ⚠️ Acceptable |
| Quality Score | 85.8/100 | >80 | ✅ |

---

## Phase 2: Model Selection and Fine-Tuning

### Objective
Select and fine-tune an Arabic language model for **real-time End-of-Utterance (EOU) detection** in LiveKit voice agents, optimized for Saudi Arabic dialect.

### Requirements

#### Functional Requirements
1. **Real-time Inference:** <50ms latency per utterance
2. **Saudi Arabic Support:** Optimized for Najdi dialect
3. **Binary Classification:** EOU (1) vs non-EOU (0)
4. **High Accuracy:** >90% F1-score on test set
5. **LiveKit Integration:** Compatible with LiveKit agent framework

#### Technical Requirements
1. **Model Size:** <1GB for efficient deployment
2. **Framework:** PyTorch or TensorFlow
3. **Input:** Short text utterances (1-12 words)
4. **Output:** Binary classification logits
5. **Deployment:** CPU-friendly for cost-effective scaling

---

## Model Selection Criteria

### Evaluation Matrix

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Inference Speed** | 30% | Latency per utterance (<50ms target) |
| **Arabic Support** | 25% | Pre-training on Arabic corpus, dialect coverage |
| **Model Size** | 20% | Parameters and disk size (<1GB target) |
| **Fine-tuning Ease** | 15% | Available tooling, documentation, examples |
| **Community Support** | 10% | Active development, issue resolution, examples |

---

## Candidate Models

### 1. **AraBERT (aubmindlab/bert-base-arabertv2)**

**Overview:**
- Pre-trained BERT model for Arabic
- 136M parameters
- Trained on 77GB of Arabic text
- Widely used in Arabic NLP research

**Pros:**
- ✅ Strong Arabic language understanding
- ✅ Proven performance on Arabic classification tasks
- ✅ Extensive documentation and examples
- ✅ Active community support
- ✅ Compatible with Hugging Face Transformers

**Cons:**
- ⚠️ Larger model size (~540MB)
- ⚠️ Slower inference compared to smaller models
- ⚠️ Not specifically optimized for Saudi dialect

**Expected Performance:**
- Inference speed: ~30-50ms per utterance (CPU)
- F1-score: 92-95% (estimated)
- Model size: 540MB

**Use Case Fit:** ⭐⭐⭐⭐ (4/5)

---

### 2. **CAMeL-Lab/bert-base-arabic-camelbert-msa**

**Overview:**
- BERT model for Modern Standard Arabic (MSA)
- 110M parameters
- Trained on MSA corpus
- Developed by NYU Abu Dhabi

**Pros:**
- ✅ Optimized for MSA
- ✅ Good performance on formal Arabic
- ✅ Smaller than AraBERT
- ✅ Well-documented

**Cons:**
- ⚠️ Less coverage of dialectal Arabic
- ⚠️ May struggle with informal Saudi dialect
- ⚠️ Limited community examples for Saudi Arabic

**Expected Performance:**
- Inference speed: ~25-40ms per utterance (CPU)
- F1-score: 88-92% (estimated, lower on informal)
- Model size: 440MB

**Use Case Fit:** ⭐⭐⭐ (3/5)

---

### 3. **Qwen/Qwen2.5-0.5B-Instruct**

**Overview:**
- Lightweight instruction-tuned LLM
- 500M parameters
- Multilingual support (includes Arabic)
- Optimized for instruction following

**Pros:**
- ✅ Multilingual support including Arabic
- ✅ Instruction-tuned for task understanding
- ✅ Modern architecture (2024)
- ✅ Can be fine-tuned for classification

**Cons:**
- ⚠️ Larger model size (~1GB)
- ⚠️ Slower inference for simple classification
- ⚠️ Overkill for binary classification task
- ⚠️ Not specifically optimized for Arabic dialects

**Expected Performance:**
- Inference speed: ~100-200ms per utterance (CPU)
- F1-score: 90-94% (estimated)
- Model size: ~1GB

**Use Case Fit:** ⭐⭐ (2/5) - Too large for real-time requirements

---

### 4. **aubmindlab/bert-base-arabertv02-twitter**

**Overview:**
- AraBERT variant trained on Twitter data
- 136M parameters
- Optimized for informal, dialectal Arabic
- Includes emoji and social media patterns

**Pros:**
- ✅ **Optimized for dialectal Arabic** (including Saudi)
- ✅ Handles informal language patterns
- ✅ Trained on conversational data
- ✅ Good for short utterances
- ✅ Same architecture as AraBERT (easy to use)

**Cons:**
- ⚠️ May have social media bias (emojis, hashtags)
- ⚠️ Similar size to AraBERT (~540MB)
- ⚠️ Moderate inference speed

**Expected Performance:**
- Inference speed: ~30-50ms per utterance (CPU)
- F1-score: 93-96% (estimated, better on informal)
- Model size: 540MB

**Use Case Fit:** ⭐⭐⭐⭐⭐ (5/5) - **Best fit for Saudi dialect**

---

### 5. **distilbert-base-multilingual-cased**

**Overview:**
- Distilled version of multilingual BERT
- 66M parameters (50% smaller than BERT)
- Supports 104 languages including Arabic
- Optimized for speed

**Pros:**
- ✅ **Fastest inference** (~15-25ms per utterance)
- ✅ **Smallest model size** (~270MB)
- ✅ Multilingual support
- ✅ Well-documented and widely used

**Cons:**
- ⚠️ Weaker Arabic understanding compared to Arabic-specific models
- ⚠️ Not optimized for Saudi dialect
- ⚠️ May require more training data

**Expected Performance:**
- Inference speed: ~15-25ms per utterance (CPU)
- F1-score: 85-90% (estimated, lower on dialect)
- Model size: 270MB

**Use Case Fit:** ⭐⭐⭐ (3/5) - Good for speed, but weaker on Arabic

---

## Recommended Model: **aubmindlab/bert-base-arabertv02-twitter**

### Rationale

**Why AraBERT-Twitter is the best choice:**

1. **Dialectal Arabic Optimization** ⭐⭐⭐⭐⭐
   - Trained on Twitter data with heavy dialectal content
   - Handles informal Saudi Arabic patterns naturally
   - Better suited for conversational AI than MSA-only models

2. **Proven Architecture** ⭐⭐⭐⭐⭐
   - BERT-base architecture is well-established for classification
   - Extensive tooling and examples available
   - Easy integration with Hugging Face Transformers

3. **Balanced Performance** ⭐⭐⭐⭐
   - Inference speed: ~30-50ms (acceptable for real-time)
   - Expected F1-score: 93-96% (high accuracy)
   - Model size: 540MB (deployable)

4. **Community Support** ⭐⭐⭐⭐⭐
   - Active development by aubmindlab
   - Extensive documentation and examples
   - Proven success in Arabic NLP tasks

### Alternative: **distilbert-base-multilingual-cased** (if speed is critical)

If real-time performance is the top priority and accuracy can be slightly lower, use DistilBERT for:
- ✅ Fastest inference (~15-25ms)
- ✅ Smallest model size (270MB)
- ⚠️ Lower accuracy on Saudi dialect (85-90% F1)

---

## Fine-Tuning Strategy

### Training Configuration

```python
# Model
model_name = "aubmindlab/bert-base-arabertv02-twitter"
num_labels = 2  # Binary classification

# Training hyperparameters
learning_rate = 2e-5
batch_size = 16
num_epochs = 3
warmup_steps = 500
weight_decay = 0.01

# Optimizer
optimizer = AdamW
scheduler = linear_with_warmup

# Evaluation
eval_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
metric_for_best_model = "f1"
```

### Data Preparation

**Dataset Splits:**
- Train: 7,000 samples (70%)
- Validation: 1,500 samples (15%)
- Test: 1,500 samples (15%)

**Preprocessing:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")

def preprocess_function(examples):
    return tokenizer(
        examples["utterance"],
        truncation=True,
        padding="max_length",
        max_length=64  # Short utterances (1-12 words)
    )
```

### Training Pipeline

**Step 1: Environment Setup**
```bash
pip install transformers datasets torch scikit-learn
```

**Step 2: Load Dataset**
```python
from datasets import load_dataset

dataset = load_dataset("MrEzzat/arabic-eou-detection-10k")
```

**Step 3: Tokenize**
```python
tokenized_datasets = dataset.map(preprocess_function, batched=True)
```

**Step 4: Train**
```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv02-twitter",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

**Step 5: Evaluate**
```python
results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test F1-score: {results['eval_f1']:.4f}")
```

---

## Evaluation Metrics

### Primary Metrics
1. **F1-score** (macro-averaged)
2. **Precision** (per class)
3. **Recall** (per class)
4. **Accuracy**

### Secondary Metrics
1. **Inference latency** (ms per utterance)
2. **Confusion matrix** (EOU vs non-EOU)
3. **Per-style performance** (formal, informal, asr_like)

### Target Performance
| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| F1-score | >93% | >90% |
| Precision (EOU) | >92% | >88% |
| Recall (EOU) | >92% | >88% |
| Inference Latency | <50ms | <100ms |

---

## LiveKit Integration Plan

### Architecture

```
User Speech → STT → Text Utterance → EOU Model → Decision
                                                    ↓
                                            [Continue Listening]
                                                    ↓
                                            [Process & Respond]
```

### Integration Steps

**Step 1: Export Model**
```python
# Save fine-tuned model
model.save_pretrained("./arabic-eou-model")
tokenizer.save_pretrained("./arabic-eou-model")
```

**Step 2: Create Inference Module**
```python
class EOUDetector:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def predict(self, utterance: str) -> bool:
        """Returns True if EOU, False if non-EOU"""
        inputs = self.tokenizer(utterance, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        return prediction == 1  # 1 = EOU
```

**Step 3: Replace LiveKit Default EOU**
```python
from livekit.agents import Agent
from eou_detector import EOUDetector

eou_model = EOUDetector("./arabic-eou-model")

class ArabicVoiceAgent(Agent):
    def on_transcript_update(self, transcript: str):
        if eou_model.predict(transcript):
            # End of utterance detected
            self.process_and_respond(transcript)
        else:
            # Continue listening
            self.continue_listening()
```

---

## Timeline and Milestones

### Week 1: Model Fine-Tuning
- [ ] Set up training environment
- [ ] Load dataset from Hugging Face
- [ ] Fine-tune AraBERT-Twitter model
- [ ] Evaluate on test set
- [ ] Optimize hyperparameters if needed

### Week 2: Optimization and Testing
- [ ] Benchmark inference latency
- [ ] Optimize model for deployment (quantization, ONNX)
- [ ] Test on edge cases (very short/long utterances)
- [ ] Validate per-style performance

### Week 3: LiveKit Integration
- [ ] Create inference module
- [ ] Integrate with LiveKit agent
- [ ] End-to-end testing with voice input
- [ ] Performance tuning

### Week 4: Demo and Documentation
- [ ] Create demo video
- [ ] Write integration guide
- [ ] Publish model to Hugging Face
- [ ] Project completion report

---

## Risk Mitigation

### Risk 1: Low Accuracy on Informal Dialect
**Mitigation:**
- Use AraBERT-Twitter (optimized for dialectal Arabic)
- Monitor per-style performance during training
- Consider data augmentation if informal F1 < 85%

### Risk 2: Slow Inference Speed
**Mitigation:**
- Benchmark latency early in training
- Use model quantization (INT8) if needed
- Consider DistilBERT as fallback
- Deploy on GPU if CPU latency > 100ms

### Risk 3: Overfitting on Synthetic Data
**Mitigation:**
- Use early stopping with validation monitoring
- Apply dropout (0.1) during fine-tuning
- Test on real voice transcripts (if available)
- Consider collecting small human-annotated validation set

---

## Success Criteria

### Must-Have (P0)
- ✅ F1-score > 90% on test set
- ✅ Inference latency < 100ms per utterance
- ✅ Successfully integrated with LiveKit
- ✅ Model published to Hugging Face

### Should-Have (P1)
- ✅ F1-score > 93% on test set
- ✅ Inference latency < 50ms per utterance
- ✅ Per-style F1 > 88% (formal, informal, asr_like)
- ✅ Demo video showcasing real-time EOU detection

### Nice-to-Have (P2)
- ✅ F1-score > 95% on test set
- ✅ Inference latency < 30ms per utterance
- ✅ Quantized model for edge deployment
- ✅ Benchmark comparison with other Arabic EOU models

---

## Next Steps

1. **Confirm Model Selection:** AraBERT-Twitter (recommended) or DistilBERT (speed-optimized)
2. **Set Up Training Environment:** Install dependencies, configure GPU/CPU
3. **Start Fine-Tuning:** Follow training pipeline outlined above
4. **Evaluate and Iterate:** Monitor metrics, optimize hyperparameters
5. **Prepare for LiveKit Integration:** Export model, create inference module

---

**Phase 2 Status:** Ready to Begin  
**Recommended Model:** `aubmindlab/bert-base-arabertv02-twitter`  
**Expected Timeline:** 3-4 weeks  
**Success Probability:** High (85%+)

---

**Document Version:** 1.0  
**Last Updated:** December 10, 2025  
**Author:** HAMS Project Team
