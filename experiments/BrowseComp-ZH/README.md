# BrowseComp-ZH Chinese Web Browsing Capability Evaluation

## 📋 Introduction

BrowseComp-ZH is a benchmark specifically designed to evaluate the retrieval and reasoning capabilities of large language models in Chinese web environments. This benchmark constructs complex multi-hop retrieval and reasoning tasks for the Chinese information environment, requiring models to handle multiple challenges such as platform fragmentation, language characteristics, and content censorship.

## 🎯 Evaluation Objectives

- Evaluate large models' Chinese web retrieval capabilities
- Test multi-hop reasoning abilities
- Compare performance of open-source models, closed-source models, and AI search systems
- Analyze model calibration error

## 📊 Dataset

- **Number of Questions**: 289 multi-hop retrieval reasoning questions
- **Language**: Chinese
- **Data Format**: JSON
- **Encryption Protection**: Dataset is encrypted and requires password for decryption

### Data Decryption

```bash
python data/browsecomp-zh-decrypt.py \
    --input data/browsecomp-zh-encrypted.xlsx \
    --output data/browsecomp-zh-decrypted.xlsx \
    --json_output raw_data/browsecomp-zh-decrypted.json
```

The system will prompt for a password. Decrypted data is saved in `raw_data/browsecomp-zh-decrypted.json`.

## 🚀 Quick Start

### 1. Environment Configuration

```bash
cd BrowseComp-ZH
pip install -r requirements.txt
```

### 2. Configure API Keys

Configure corresponding model API keys in `run.py`, `rag.py`, `deepsearch.py` files.

### 3. Run Evaluation

#### Standard Evaluation Mode
```bash
bash run.sh
```

#### RAG Enhancement Mode - Query-focused summarization
```bash
bash rag.sh
```

#### DeepSearch Mode (LingoEDU): Structural decomposition
```bash
bash deepsearch.sh
```

### 4. Results Statistics

```bash
python run_acc_calibration_error.py
```

## 📁 Directory Structure

```
BrowseComp-ZH/
├── data/                           # Data files
│   ├── browsecomp-zh-encrypted.xlsx   # Encrypted dataset
│   └── browsecomp-zh-decrypt.py       # Decryption script
├── raw_data/                       # Decrypted JSON data
│   └── browsecomp-zh-decrypted.json
├── predict_data/                   # Model prediction results
├── eval_data/                      # Answer extraction results
├── output_data/                    # Final evaluation results
├── outcome_data/                   # Accuracy and calibration error statistics
├── run.py                          # Standard evaluation script
├── rag.py                          # Query-focused Summarization evaluation
├── deepsearch.py                   # LingoEDU Structural Decomposition evaluation
├── edu.py                          # LingoEDU parsing tool
├── prompt.py                       # Prompt templates
├── run.sh                          # Standard evaluation launch script
├── rag.sh                          # RAG evaluation launch script
├── deepsearch.sh                   # DeepSearch launch script
└── requirements.txt                # Python dependencies
```


## 📈 Evaluation Modes

### 1. Standard Mode (run.py)
Directly use large models to answer questions without providing additional context.

**Supported Models**:
- Open-source: DeepSeek-V3, DeepSeek-R1, Qwen3-235B, QwQ-32B
- Closed-source: GPT-4o, O3, O4-mini, Claude-3.5/3.7-Sonnet, Gemini series

### 2. RAG Mode (rag.py) - Query-focused Summarization
Use query-focused summarization techniques to provide compressed and focused context to the model.


**Technical Note**: Query-focused summarization dynamically extracts and compresses the most relevant information fragments based on questions.

### 3. DeepSearch Mode (deepsearch.py) - LingoEDU Structural Decomposition
Use LingoEDU for structural decomposition, parsing retrieved content into hierarchical structures.


**Technical Note**: LingoEDU Structural Decomposition breaks down webpage content into hierarchical structural units, facilitating model understanding and reasoning.


## ⚙️ Configuration Instructions

### Model Configuration

Modify the following parameters in the corresponding evaluation scripts:

```python
parser.add_argument('--model', type=str, default="GPT-4o")
parser.add_argument('--max_workers', type=int, default=10)  # Concurrency
```

### Concurrency Control

Different models have different concurrency limits:
- Qwen2.5-Max, QwQ-32B: 3
- O4-mini: 5
- Claude series: 3
- Others: 10



## 🛠️ Tool Scripts

### edu.py
LingoEDU structural decomposition tool for parsing hierarchical structure of webpage content.

**Functions**:
- Decompose webpage content into hierarchical structure (Level 1-N)
- Identify semantic units such as chapters, paragraphs, lists
- Preserve semantic hierarchical relationships of documents

### prompt.py
Contains system prompt and evaluation prompt templates:
- `SYSTEM_PROMPT_CN`: Chinese system prompt
- `JUDGE_PROMPT_CN`: Evaluation prompt template

