# LongBench Long Text Understanding Evaluation

## 📋 Introduction

LongBench is a benchmark for evaluating the long text understanding capabilities of large language models. The core innovation of this project lies in using **LingoEDU Structural Decomposition + Rerank Model** to intelligently compress long texts, significantly reducing token consumption while preserving key information, improving evaluation efficiency and model performance.


## 💡 Core Technology

### LingoEDU Structural Decomposition

Perform structural decomposition on long texts, parsing them into hierarchical semantic units:
- **Level 1**: Chapter titles
- **Level 2**: Section titles
- **Level 3**: Paragraphs
- **Level 4**: Sentences



### Rerank Model

Re-rank content at each level based on question relevance, selecting the Top-K most relevant content.

**Working Principle**:
- Calculate semantic relevance between each structural unit and the question
- Perform relevance ranking independently at each level
- Select the most relevant Top-K units

### Processing Flow

```
Original Long Text (10000 tokens)
    ↓
LingoEDU Structural Decomposition → Hierarchical Structure
    ├─ Level 1: [Chapter 1, Chapter 2, Chapter 3]

    ↓
Rerank Filtering (Top-K per level)
    ├─ Level 1: Keep Top-K chapters

    ↓
Recombine
    ↓
Compressed Text 
    ↓
LLM Reasoning → Answer
```

## 🚀 Quick Start

### 1. Environment Configuration

```bash
cd LongBench
pip install requests tiktoken
# If using EDU SDK
pip install deeplang_parse
```

### 2. Configure API Keys

Create `config.py` (refer to `config_example.py`):

```python
# Rerank API configuration
RERANK_API_KEY = "your_rerank_api_key"
RERANK_ENDPOINT = 'https://your-rerank-endpoint/rerank'
RERANK_MODEL = "qwen3"

# LingoEDU API configuration
class EduConfig:
    EduHost = 'http://your-lingoedu-endpoint'
    AccessKeyId = 'your_access_key_id'
    AccessKeySecret = 'your_access_key_secret'
```

### 3. Simple Example

```python
from edu_rerank_example import EduRerankProcessor, process_context_with_edu_rerank

# Initialize processor
processor = EduRerankProcessor()

# Prepare data
context = "Very long document content..."
query = "Your question"

# Execute compression
filtered_text, stats = process_context_with_edu_rerank(
    processor=processor,
    context=context,
    query=query,
    top_k_per_level=3  
)

print(f"Compression ratio: {stats['ratio']:.2%}")
print(f"Original Tokens: {stats['orig_tokens']}")
print(f"Compressed Tokens: {stats['new_tokens']}")
```

### 4. Run Evaluation

```bash
python simple_example.py
```

## 📁 Directory Structure

```
LongBench/
├── edu_rerank_example.py           # EDU+Rerank core implementation
├── simple_example.py               # Simple usage example
├── config_example.py               # Configuration file example
├── config.py                       # Actual configuration (create yourself)
└── README.md                       # This document
```

## 🔧 Core Components

### 1. EduRerankProcessor Class

```python
class EduRerankProcessor:
    def __init__(self):
        # Initialize LingoEDU client and Rerank configuration
        
    def parse_with_edu(self, context: str) -> dict:
        """Use LingoEDU for structural decomposition"""
        
    def rerank_nodes(self, nodes: list, query: str, top_k: int) -> list:
        """Use Rerank model to rank structural units"""
```

### 2. Main Functions

```python
def process_context_with_edu_rerank(
    processor: EduRerankProcessor,
    context: str,
    query: str,
    top_k_per_level: int = 3
) -> Tuple[str, dict]:
    """
    Perform LingoEDU Structural Decomposition + Rerank compression on long text
    
    Parameters:
        processor: EDU processor instance
        context: Original long text
        query: Question/query
        top_k_per_level: Number of nodes to keep per level
        
    Returns:
        filtered_text: Compressed text
        stats: Statistics {orig_tokens, new_tokens, ratio}
    """
```

## 📊 Evaluation Process

### Standard Evaluation Process

```python
# 1. Load data
with open('data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 2. Initialize processor
processor = EduRerankProcessor()

# 3. Process each sample
for item in data:
    context = item['context']
    query = item['query']
    
    # 4. LingoEDU Structural Decomposition + compression (only for long texts)
    if len(context) > 300:
        new_context, stats = process_context_with_edu_rerank(
            processor, context, query, top_k_per_level=3
        )
        item['context'] = new_context
        item['lingoedu_stats'] = stats
    
    # 5. LLM reasoning
    answer = call_llm(item)
    
    # 6. Save results
    item['prediction'] = answer
    save_result(item)
```

## ⚙️ Configuration Parameters

### LingoEDU Configuration

```python
class EduConfig:
    EduHost = 'http://api-lingoedu.example.com'  # LingoEDU API endpoint
    AccessKeyId = 'your_key_id'                    # Access key ID
    AccessKeySecret = 'your_key_secret'           # Access key secret
```

### Rerank Configuration

```python
RERANK_API_KEY = "your_api_key"
RERANK_ENDPOINT = 'https://api.example.com/rerank'
RERANK_MODEL = "qwen3"  # Or "bge-reranker-v2-m3"
```


## 🔍 API Documentation

### LingoEDU API

**Create structural decomposition task**:
```python
resp = client.create_task_by_file(file_path, url="")
task_id = resp.data.task_id
```

**Query task status**:
```python
resp = client.query_task_status(task_id=task_id)
status = resp.data.status  # 2=success, 3=failure
```

**Get decomposition results**:
```python
res = client.fetch_task_result(task_id=task_id)
parsed_data = json.loads(res.data.model_dump_json())
```

**Returned structured data**:
```json
{
  "sentences": [
    {
      "level": 1,        // Level (1=chapter, 2=paragraph, 3=sentence)
      "text": "content",    // Text content
      "idx": 0          // Index
    }
  ],
  "markdown": "original markdown"
}
```

### Rerank API

**Request format**:
```python
{
    "model": "qwen3",
    "query": "user question",
    "keys": [
        {"caption": "document 1"},
        {"caption": "document 2"}
    ],
    "top_n": 2
}
```

**Response format**:
```json
{
  "data": [
    {
      "index": 0,
      "relevance_score": 0.95
    },
    {
      "index": 1,
      "relevance_score": 0.82
    }
  ]
}
```

