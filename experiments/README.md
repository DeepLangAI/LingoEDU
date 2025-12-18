# LLM Benchmark Evaluation Suite

## 📋 Project Introduction

This project is a comprehensive Large Language Model (LLM) evaluation suite, containing three benchmarks across different dimensions to comprehensively evaluate the capabilities of large models in Chinese retrieval reasoning, high-difficulty problem solving, and long text understanding.

## 🎯 Evaluation System

This suite includes the following three benchmarks:

### 1️⃣ BrowseComp-ZH 

[Detailed Documentation](./BrowseComp-ZH/README.md)

### 2️⃣ HLE 

[Detailed Documentation](./HLE/README.md)

### 3️⃣ LongBench - Long Text Understanding Evaluation

[Detailed Documentation](./LongBench/README.md)

## 📊 Evaluation Comparison

| Benchmark | Evaluation Dimension | Data Scale | Core Technology | Key Metrics |
|-----------|---------------------|-----------|-----------------|-------------|
| **BrowseComp-ZH** | Chinese Retrieval Reasoning | 289 questions | Query-focused Summarization<br/>LingoEDU Structural Decomposition | Accuracy, Calibration Error |
| **HLE** | High-Difficulty Problem Solving | Custom | Solver-Selector Architecture<br/>LingoEDU Structural Decomposition | Correctness |
| **LongBench** | Long Text Understanding | Custom | LingoEDU Structural Decomposition<br/>Rerank Semantic Compression | Accuracy, Compression Rate |

## 🚀 Quick Start

### Environment Requirements

Each benchmark has independent dependency management:

```bash
# BrowseComp-ZH
cd BrowseComp-ZH
pip install -r requirements.txt

# HLE
cd HLE
pip install -r requirements.txt

# LongBench
cd LongBench
pip install requests tiktoken
```

### Configure API Keys

Each benchmark requires configuration of corresponding API keys:

1. **BrowseComp-ZH**: Configure model API keys in `run.py`
2. **HLE**: Configure model and search API keys in `main.py`
3. **LongBench**: Create `config.py` to configure EDU and Rerank API keys

### Run Evaluation

```bash
# BrowseComp-ZH Standard Evaluation
cd BrowseComp-ZH
bash run.sh

# HLE Full Enhanced Evaluation
cd HLE
python main.py --batch_file hle.json --enable_search --enable_edu

# LongBench Simple Example
cd LongBench
python simple_example.py
```

## 📁 Project Structure

```
.
├── BrowseComp-ZH/              # Chinese Web Browsing Capability Evaluation
│   ├── data/                      # Dataset
│   ├── raw_data/                  # Decrypted data
│   ├── run.py                     # Standard evaluation
│   ├── rag.py                     # Query-focused Summarization
│   ├── deepsearch.py              # LingoEDU Structural Decomposition
│   ├── edu.py                     # LingoEDU tool
│   └── README.md
│
├── HLE/                        # High-Difficulty Problem Solving Evaluation
│   ├── llm_agent/                 # Agent core module
│   ├── prompt/                    # Prompt templates
│   ├── main.py                    # Main program
│   ├── hle_score.py               # Scoring script
│   └── README.md
│
├── LongBench/                  # Long Text Understanding Evaluation
│   ├── edu_rerank_example.py      # Core implementation
│   ├── simple_example.py          # Usage example
│   ├── config_example.py          # Configuration example
│   └── README.md
│
└── README.md                   # This document
```

## 🎓 Usage Scenarios

### Scenario 1: Comprehensive Model Capability Evaluation

If you want to comprehensively evaluate a large model's capabilities, you can test on all three benchmarks:

```bash
# 1. Test Chinese retrieval reasoning capability
cd BrowseComp-ZH && bash run.sh

# 2. Test high-difficulty problem solving capability
cd ../HLE && python main.py --batch_file hle.json

# 3. Test long text understanding capability
cd ../LongBench && python simple_example.py
```

### Scenario 2: Specific Capability Evaluation

Choose specific benchmarks based on requirements:

- **Need to evaluate search capability?** → BrowseComp-ZH
- **Need to evaluate reasoning capability?** → HLE
- **Need to evaluate long text processing?** → LongBench

