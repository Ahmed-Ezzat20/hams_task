# HAMS: Arabic EOU Dataset Generator

**HAMS** is a production-ready toolkit for generating synthetic Arabic conversational data for End-of-Utterance (EOU) detection. It provides a modular, extensible, and easy-to-use system for creating high-quality datasets for EOU model training.

## Features

- **Modular Architecture**: 4 clean modules for prompts, generation, post-processing, and writing.
- **ASR-Style Degradation**: Simulate realistic ASR noise with one flag.
- **YAML-based Prompts**: Easily manage and extend conversation prompts.
- **JSONL Output**: Streamable and easy to merge.
- **CLI Commands**: Simple entry points for generating and building datasets.
- **Production-Ready**: Clean, tested, and documented.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ahmed-Ezzat20/hams_task.git
cd hams_task
```

2. **Install dependencies:**
```bash
# Using pip
pip install -e .[dev]

# Or with uv
uv sync --dev
```

3. **Set your API key:**
```bash
export NEBIUS_API_KEY="your-nebius-api-key"
```

---

## Quick Start

### Build a Complete Dataset (Recommended)

This command generates 100 conversations in both `clean` and `asr_like` styles and saves them to the `data/` directory.

```bash
hams-build --num-conversations 100 --output-dir data
```

**Output:**
- `data/conversations_clean.jsonl`
- `data/conversations_asr.jsonl`

---

## Usage

### 1. `hams-generate`: Generate Conversations

Generate conversations with specific styles and domains.

**Generate 50 clean conversations:**
```bash
hams-generate \
    --num-conversations 50 \
    --output-file data/clean_conversations.jsonl \
    --style clean
```

**Generate 50 ASR-like conversations:**
```bash
hams-generate \
    --num-conversations 50 \
    --output-file data/asr_conversations.jsonl \
    --style asr_like
```

**Generate conversations for specific domains:**
```bash
hams-generate \
    --num-conversations 20 \
    --output-file data/restaurant_conversations.jsonl \
    --domains restaurant hospitality
```

### 2. `hams-asr-augment`: Add ASR Noise to Existing Data

Apply ASR-style noise to an existing dataset.

```bash
hams-asr-augment \
    --input-file data/clean_conversations.jsonl \
    --output-file data/augmented_conversations.jsonl
```

### 3. `hams-build`: Build Complete Datasets

Wrapper for `hams-generate` and `hams-asr-augment`.

**Build a dataset with 200 conversations (clean + ASR):**
```bash
hams-build \
    --num-conversations 200 \
    --output-dir datasets/eou_v1
```

**Build a dataset with only clean conversations:**
```bash
hams-build \
    --num-conversations 100 \
    --output-dir datasets/eou_v1_clean \
    --style clean
```

---

## Architecture

The system is composed of 4 core modules:

1. **`PromptBuilder`**: Loads prompts from `prompts.yaml` and builds formatted prompt strings.
2. **`ConversationGenerator`**: Generates raw JSON conversations from prompts using the LLM API.
3. **`PostProcessor`**: Normalizes raw JSON to `Conversation` and `Turn` dataclasses and applies ASR noise.
4. **`DatasetWriter`**: Writes `Conversation` objects to JSONL files.

### Data Flow

```
[PromptBuilder] -> [ConversationGenerator] -> [PostProcessor] -> [DatasetWriter]
   (YAML)           (LLM API)             (Normalize)         (JSONL)
```

---

## Prompts

Conversation prompts are stored in `hams/prompts.yaml`. You can easily add or modify prompts in this file.

Each prompt has:
- `id`: Unique identifier
- `domain`: Conversation domain (e.g., restaurant, banking)
- `description`: Brief description
- `scenario`: Detailed scenario for the LLM

---

## Testing

Run unit tests to ensure core functionality works as expected.

```bash
pytest
```

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m "feat: add your feature"`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
