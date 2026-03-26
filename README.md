# GraphRAG-FI

A Graph-based Retrieval-Augmented Generation framework for Knowledge Graph Question Answering (KGQA).


## Requirements

- Python 3.8+
- PyTorch
- CUDA (for GPU acceleration)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── config/              # DeepSpeed and training configurations
├── datasets/            # Dataset files (WebQSP, CWQ)
├── prompts/             # Prompt templates for different LLMs
├── results/             # Output results and predictions
├── scripts/             # Shell scripts for training and inference
│   ├── planning.sh      # Generate rule paths
│   ├── rog-reasoning.sh # Run reasoning with RoG model
│   └── train.sh         # Training script
└── src/                 # Source code
    ├── align_kg/        # Knowledge graph alignment
    ├── joint_training/  # Joint finetuning modules
    ├── llms/            # LLM implementations (ChatGPT, Llama, etc.)
    ├── qa_prediction/   # Question answering prediction
    └── utils/           # Utility functions
```

## Usage

### 1. Generate Rule Paths

```bash
./scripts/planning.sh
```

### 2. Run Reasoning

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/rog-reasoning.sh FI 0.2
```


## Datasets

- **WebQSP**: Web Questions Semantic Parses dataset
- **CWQ**: ComplexWebQuestions dataset

## Supported Models

- RoG (Reasoning on Graphs)
- GPT-3.5 / GPT-4
- Llama 2
- Alpaca
- FLAN-T5

## Acknowledgements

This project is built upon the framework provided by [Reasoning on Graphs (RoG)](https://github.com/RManLuo/reasoning-on-graphs):

> **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**  
> Luo et al., ICLR 2024

We thank the authors for their excellent work and open-source contribution.

