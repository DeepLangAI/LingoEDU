# HLE 

## 🏗️ Core Architecture

### Solver-Selector Pipeline

```
Input Question
    ↓
Solver (Generate 5 solutions in parallel)
    ├─ Solution 1
    ├─ Solution 2
    ├─ Solution 3
    ├─ Solution 4
    └─ Solution 5
    ↓
Selector (Select best solution)
    ↓
Final Answer
```

## ✨ Core Features

### 1. Basic Mode
Pure large model reasoning without external tools.

### 2. Search Enhancement Mode (--enable_search)
Integrate online search to provide real-time information to the model.

**Features**:
- Automatic search for relevant information
- Multi-source result aggregation
- Intelligent result ranking

### 3. LingoEDU Structural Decomposition Mode (--enable_edu)
On top of search enhancement, use LingoEDU to perform structural decomposition on retrieved webpages.

**Features**:
- LingoEDU structural decomposition technology
- Parse webpage hierarchical structure (titles, paragraphs, lists, etc.)
- Preserve semantic hierarchical relationships
- Extract key information, reduce noise interference

**Technical Note**: LingoEDU Structural Decomposition transforms unstructured webpage content into hierarchical structural units, each containing level information, facilitating subsequent information filtering and reasoning.

## 🚀 Quick Start


### 1. Configure API Keys

Configure in `main.py`:
```python
API_KEY = 'your_api_key_here'
BASE_URL = "your_api_endpoint"
```

### 2. Prepare Data

Data format (`hle.json`):
```json
[
  {
    "id": "question_1",
    "query": "Question description",
    "gt": "Standard answer",
    "category": "Math/Physics/Computer Science"
  }
]
```

### 3. Run Evaluation

#### Basic Mode
```bash
python main.py \
    --batch_file hle.json \
    --output_file results/hle_base.jsonl
```

#### Search Enhancement Mode
```bash
python main.py \
    --batch_file hle.json \
    --output_file results/hle_search.jsonl \
    --enable_search
```

#### Full Enhancement Mode (Search + LingoEDU Structural Decomposition)
```bash
python main.py \
    --batch_file hle.json \
    --output_file results/hle_full.jsonl \
    --enable_search \
    --enable_edu
```

### 4. Scoring

```bash
python hle_score.py
```

Modify configuration in the script:
```python
judge_model = "o3-mini"  # Scoring model
data_path = "results/hle_full.jsonl"  # File to score
```

## 📁 Directory Structure

```
HLE/
├── llm_agent/                      # Agent core module
│   ├── base_agent.py                  # Base Agent class
│   ├── context.py                     # Context management
│   ├── utils.py                       # Utility functions
│   └── tools/                         # Toolset
│       └── tool_manager.py               # Tool manager
├── prompt/                         # Prompt templates
│   ├── solver_user.txt                # Solver user prompt
│   ├── solver_prefix.txt              # Solver prefix
│   ├── select_user.txt                # Selector user prompt
│   └── select_prefix.txt              # Selector prefix
├── results/                        # Evaluation results
├── main.py                         # Main program
├── search_tool.py                  # Search tool
├── hle_score.py                    # Scoring script
├── hle.json                        # Test data
├── judge_config.yaml               # Scoring configuration
└── README.md                       # This document
```

## 🔧 Configuration Instructions

### Main Program Parameters

```bash
python main.py \
    --query "Single question"        # Single question mode
    --batch_file "data.json"        # Batch processing mode
    --output_file "result.jsonl"    # Output file
    --enable_search                 # Enable search
    --enable_edu                    # Enable LingoEDU structural decomposition
    --debug                         # Debug mode
```

### Model Configuration

In the `LLMAgent` class in `main.py`:
```python
model = "gemini-3-pro-preview"  # Default model
max_tokens = 16000               # Maximum tokens
temperature = 1.0                # Temperature parameter
```

Supported models:
- Gemini series
- GPT series
- Kimi series

### Concurrency Configuration

```python
# Solver concurrency (fixed)
max_workers = 5  # Generate 5 solutions

# Batch processing concurrency
num_workers = 20  # Can be specified via command line
```

## 📊 Scoring System

### Scoring Standards

Use large model (e.g., O3-mini) as judge:
- **2 points**: Correct answer
- **0 points**: Incorrect answer

### Scoring Configuration (judge_config.yaml)

```yaml
HLE_eval_prompt_template: |
  Question: {question}
  Model Response: {response}
  Correct Answer: {correct_answer}
  
  Please judge whether the model's answer is correct.
  If correct, reply: "correct: yes"
  If incorrect, reply: "correct: no"
```


## 🛠️ Search Tool

### SearchTool Class (search_tool.py)

```python
# Initialize
search_tool = SearchTool(SearchConfig())

# Execute search (basic)
results = search_tool.perform_enhanced_search(
    query="question",
    enable_edu=False
)

# Execute search (EDU enhancement)
results = search_tool.perform_enhanced_search(
    query="question",
    enable_edu=True
)

# Format results
formatted = search_tool.format_enhanced_search_results(
    results,
    enable_edu=True
)
```

