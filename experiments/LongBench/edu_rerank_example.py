"""
LingoEDU Structural Decomposition + Rerank Filtering Example Code

This example demonstrates how to use LingoEDU structural decomposition and Rerank model to compress long text context in LongBench evaluation.

Core Process:
1. Context -> TXT file -> LingoEDU structural decomposition -> Get hierarchical structure nodes
2. Input(Question) + nodes -> Rerank -> Filter Top-K for each level
3. Filtered nodes -> LLM -> Answer
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Tuple

# ================= Configuration Section =================
# Note: Replace with actual API Key and Endpoint when using

# [Comment in English]
RERANK_API_KEY = ""
RERANK_ENDPOINT = ""

# [Comment in English]
class EduConfig:
    EduHost = ""
    AccessKeyId = ""
    AccessKeySecret = ""

# ================= Utility Functions =================

def count_tokens(text: str) -> int:
    """Calculate the number of tokens in text (using tiktoken cl100k_base encoding)"""
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text, disallowed_special=()))
    except:
        return len(text) // 4  # Fallback estimation


def call_rerank_api(
    api_key: str, 
    endpoint: str, 
    model: str, 
    query: str, 
    documents: List[Dict[str, str]]
) -> Optional[Dict]:
    """
    Call Rerank API to rerank documents
    
    Args:
        api_key: Rerank API key
        endpoint: Rerank API endpoint
        model: Rerank model name to use (e.g., "qwen3")
        query: Query text (question)
        documents: List of documents to be ranked, format: [{"caption": "document content"}, ...]
    
    Returns:
        JSON result returned by API, containing ranked documents and their relevance scores
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "query": query,
        "keys": documents,
        "top_n": len(documents)
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[Rerank Error] {e}")
        return None


# [Comment in English]

class EduRerankProcessor:
    """
    LingoEDU Structural Decomposition + Rerank Filtering Processor
    
    Functions:
    1. Use LingoEDU API to perform structural decomposition on documents and obtain hierarchical structure
    2. Use Rerank API to rank nodes at each level by relevance
    3. Filter out the most relevant nodes and concatenate them for return
    """
    
    def __init__(self):
        """Initialize LingoEDU client"""
        try:
            from deeplang_parse.parse_client import AsyncParseClient, ParseAuther
            from deeplang_parse.enuming import TaskStatusEnum
            
            self.TaskStatusEnum = TaskStatusEnum
            self.auther = ParseAuther(
                access_key_id=EduConfig.AccessKeyId,
                access_key_secret=EduConfig.AccessKeySecret,
                endpoint=EduConfig.EduHost
            )
            self.auther.auth_v1()
            self.client = AsyncParseClient(endpoint=EduConfig.EduHost)
            self.use_sdk = True
        except ImportError:
            print("Warning: deeplang_parse SDK not found, please ensure the corresponding SDK is installed in the environment.")
            self.client = None
            self.use_sdk = False
    
    def _wait_until_done(self, task_id: str) -> bool:
        """Poll task status until completion or failure"""
        if not self.client:
            return False
        
        try:
            try:
                success_status = self.TaskStatusEnum.Success.value
                failed_status = self.TaskStatusEnum.Failed.value
            except:
                success_status = 2
                failed_status = 3
            
            for _ in range(60):  # Wait up to 60 seconds
                resp = self.client.query_task_status(task_id=task_id)
                if not resp:
                    return False
                
                status = resp.data.status
                if status == success_status:
                    return True
                if status == failed_status:
                    return False
                
                time.sleep(1)
            return False
        except Exception as e:
            print(f"Wait Error: {e}")
            return False
    
    def parse_file(self, file_path: str) -> Optional[Dict]:
        """
        Perform structural decomposition on file and return hierarchical data
        
        Args:
            file_path: File path to decompose (supports .txt and other text formats)
        
        Returns:
            LingoEDU structural decomposition result, containing:
            - sentences: List, each element contains {"level": int, "text": str, ...}
            - markdown: Original markdown text
        """
        if not self.client:
            return None
        
        try:
            # Create parsing task
            resp = self.client.create_task_by_file(file_path, "")
            if not resp:
                return None
            
            task_id = resp.data.task_id
            
            # Wait for task to complete
            if self._wait_until_done(task_id):
                # Get parsing result
                res = self.client.fetch_task_result(task_id=task_id)
                return json.loads(res.data.model_dump_json())
            
            return None
        except Exception as e:
            print(f"[LingoEDU Parse Error] {e}")
            return None
    
    def semantic_filter(
        self, 
        parsed_data: Dict, 
        query: str, 
        top_k_per_level: int = 3
    ) -> str:
        """
        Use Rerank to perform semantic filtering on LingoEDU structural decomposition results
        
        Args:
            parsed_data: Data returned from LingoEDU structural decomposition
            query: Query text (question)
            top_k_per_level: Number of Top-K nodes to keep at each level
        
        Returns:
            Filtered text (concatenated string)
        """
        # Extract sentences
        sentences = parsed_data.get("sentences", [])
        
        # If no sentences, try using markdown
        if not sentences:
            md = parsed_data.get("markdown", "")
            if md:
                sentences = [
                    {"text": line, "level": 1, "idx": i} 
                    for i, line in enumerate(md.split('\n')) 
                    if line.strip()
                ]
            else:
                return ""
        
        # Group by level
        level_map = {}
        for s in sentences:
            lvl = int(s.get("level", 1) or 1)
            if lvl not in level_map:
                level_map[lvl] = []
            
            level_map[lvl].append({
                "caption": (s.get("text") or "").strip(),
                "original_node": s
            })
        
        # Perform Rerank filtering for each level
        selected_nodes = []
        for lvl, nodes in level_map.items():
            if not nodes:
                continue
            
            # If the number of nodes is less than top_k, no need for Rerank, keep all
            if len(nodes) <= top_k_per_level:
                for n in nodes:
                    selected_nodes.append(n["original_node"])
                continue
            
            # Prepare Rerank request data (limit caption length)
            docs_payload = [
                {"caption": n["caption"][:1000]} 
                for n in nodes
            ]
            
            if not docs_payload:
                continue
            
            # Call Rerank API
            rerank_res = call_rerank_api(
                api_key=RERANK_API_KEY,
                endpoint=RERANK_ENDPOINT,
                model="qwen3",
                query=query,
                documents=docs_payload
            )
            
            # Process Rerank result
            if rerank_res and 'data' in rerank_res:
                # Sort by relevance score
                ranked_items = sorted(
                    rerank_res['data'], 
                    key=lambda x: x.get('relevance_score', 0), 
                    reverse=True
                )
                # Get Top-K indices
                top_indices = [
                    item['index'] 
                    for item in ranked_items[:top_k_per_level]
                ]
                # Add selected nodes
                for idx in top_indices:
                    if idx < len(nodes):
                        selected_nodes.append(nodes[idx]["original_node"])
            else:
                # Rerank failed, use first top_k nodes
                for n in nodes[:top_k_per_level]:
                    selected_nodes.append(n["original_node"])
        
        # Concatenate filtered text
        result_text = "\n".join([
            (n.get("text") or "").strip() 
            for n in selected_nodes
        ])
        
        return result_text


# ================= Complete Processing Flow Example =================

def process_context_with_edu_rerank(
    processor: EduRerankProcessor,
    context: str,
    query: str,
    temp_dir: str = "temp",
    top_k_per_level: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """
    Complete LingoEDU Structural Decomposition + Rerank Processing Flow
    
    Args:
        processor: EduRerankProcessor instance
        context: Original context text
        query: Query question
        temp_dir: Temporary file directory
        top_k_per_level: Number of Top-K nodes to keep at each level
    
    Returns:
        (filtered_text, stats_dict)
        - filtered_text: Filtered text
        - stats_dict: Statistics {"orig_tokens", "new_tokens", "ratio"}
    """
    orig_tokens = count_tokens(context)
    
    if not context:
        return "", {"orig_tokens": 0, "new_tokens": 0, "ratio": 0.0}
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Write temporary file
    import uuid
    safe_id = str(uuid.uuid4())
    txt_path = os.path.join(temp_dir, f"{safe_id}.txt")
    
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(context)
    except Exception as e:
        print(f"Failed to write temp file: {e}")
        return context, {
            "orig_tokens": orig_tokens, 
            "new_tokens": orig_tokens, 
            "ratio": 1.0
        }
    
    # LingoEDU structural decomposition
    parsed_json = processor.parse_file(txt_path)
    
    # Clean up temporary file
    if os.path.exists(txt_path):
        os.remove(txt_path)
    
    if not parsed_json:
        return context, {
            "orig_tokens": orig_tokens, 
            "new_tokens": orig_tokens, 
            "ratio": 1.0
        }
    
    # Rerank filtering
    filtered_text = processor.semantic_filter(
        parsed_json, 
        query, 
        top_k_per_level=top_k_per_level
    )
    
    if not filtered_text:
        return context, {
            "orig_tokens": orig_tokens, 
            "new_tokens": orig_tokens, 
            "ratio": 1.0
        }
    
    # Calculate new length statistics
    new_tokens = count_tokens(filtered_text)
    ratio = new_tokens / orig_tokens if orig_tokens > 0 else 0.0
    
    return filtered_text, {
        "orig_tokens": orig_tokens,
        "new_tokens": new_tokens,
        "ratio": round(ratio, 4)
    }


# ================= Usage Example =================

if __name__ == "__main__":
    # Example: Process a long text context
    
    # 1. Initialize processor
    processor = EduRerankProcessor()
    
    # 2. Prepare data
    context = """
    [Translated]document content。[Translated]，[Translated] LongBench [Translated] context [Translated]。
    
    Chapter 1: Introduction
    This is the content of the first chapter, containing important information.
    
    Chapter 2: Methods
    This is the content of the second chapter, describing specific methods.
    
    Chapter 3: Results
    This is the content of the third chapter, showing experimental results.
    """
    
    query = "What methods are mentioned in the document?"
    
    # 3. Execute processing
    filtered_text, stats = process_context_with_edu_rerank(
        processor=processor,
        context=context,
        query=query,
        top_k_per_level=3
    )
    
    # 4. View results
    print("=" * 50)
    print("Original text tokens:", stats["orig_tokens"])
    print("Filtered tokens:", stats["new_tokens"])
    print("Compression ratio:", stats["ratio"])
    print("=" * 50)
    print("Filtered text:")
    print(filtered_text)
    print("=" * 50)

