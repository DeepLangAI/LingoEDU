import os
import re
import sys
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from prompt import *
from openai import OpenAI
import openai

from search_tool import SearchTool, SearchConfig
SEARCH_AVAILABLE = True
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))
sys.path.append(current_dir)

with open(os.path.join(current_dir, "prompt/solver_user.txt"), "r", encoding="utf8") as f:
    SolverPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(current_dir, "prompt/solver_prefix.txt"), "r", encoding="utf8") as f:
    SolverPrompt_Assistant_Template = "".join(f.readlines())

with open(os.path.join(current_dir, "prompt/select_user.txt"), "r", encoding="utf8") as f:
    SelectPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(current_dir, "prompt/select_prefix.txt"), "r", encoding="utf8") as f:
    SelectPrompt_Assistant_Template = "".join(f.readlines())


# --- Search related configuration and model ---
BASE_URL = ""
API_KEY = ''

claude_apiKey = ''
claude_url = ''

class BaseConfig:
    EduHost = ''
    AccessKeyId = ''
    AccessKeySecret = ''

KIMI_API_KEY = ''

# Import search related modules
# try:
#     from search_tool import SearchTool, SearchConfig
#     SEARCH_AVAILABLE = True
# except ImportError:
#     print("Warning: SearchTool not available. Search functionality will be disabled.")
#     SEARCH_AVAILABLE = False

Config = BaseConfig

def strip_think_and_exec(text: str) -> str:
    """Keep only the visible answer part by removing </think> ... and </execution_results> tails."""
    if text is None:
        return ""
    out = text
    if "</think>" in out:
        out = out.split("</think>")[-1]
    if "</execution_results>" in out:
        out = out.split("</execution_results>")[-1]
    return out.strip()


class LLMAgent:
    def __init__(
        self,
        model_url: str = None,
        model_key: str = None,
        max_tokens: int = 16000,
        temperature: float = 1,
        model: str = "gemini-3-pro-preview"
        # model: str = "gpt-5"
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.model.startswith("kimi-"):
            self.base_url = "https://api.moonshot.cn/v1"
            self.api_key = KIMI_API_KEY
            if not self.api_key:
                raise ValueError("Missing Kimi API Key. Please set environment variable MOONSHOT_API_KEY or pass model_key.")
            self.client = openai.Client(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        else:
            self.base_url = model_url or BASE_URL
            self.api_key = model_key or API_KEY
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

        self.llm_config: Dict[str, Any] = {
            'model': self.model,
            'base_url': self.base_url,
            'api_key': '***hidden***',
            'generation_config': {
                'max_tokens': max_tokens,
                'temperature': temperature,
            },
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def call_model(self, user_prompt: str, assistant_prefix: str = "", stream: bool = True, print_reasoning: bool = False) -> str:
        try:
            messages = [{"role": "user", "content": user_prompt}]
            if assistant_prefix:
                messages.append({"role": "assistant", "content": assistant_prefix})

            if self.model.startswith("kimi-") and stream:
                visible_chunks = []
                thinking = False

                stream_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True,
                )

                for chunk in stream_resp:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue

                    # [Comment translated to English]
                    rc = getattr(delta, "reasoning_content", None)
                    if rc and print_reasoning:
                        if not thinking:
                            thinking = True
                            print("=============Start Reasoning=============")
                        print(rc, end="", flush=True)

                    # [Comment translated to English]
                    c = getattr(delta, "content", None)
                    if c:
                        if thinking and print_reasoning:
                            thinking = False
                            print("\n=============End Reasoning=============")
                        visible_chunks.append(c)

                content = "".join(visible_chunks).strip()
                if not content:
                    raise ValueError("Model returned empty content")
                return content

            # [Comment translated to English]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned empty content")
            return content.strip()

        except Exception as e:
            print(f"API call failed: {type(e).__name__}: {e}")
            raise

class Agent:
    def __init__(self, debug: bool = False, enable_search: bool = False, enable_edu: bool = False):
        # [Comment translated to English]
        self.deepseek_api_url = BASE_URL
        self.deepseek_api_key = API_KEY
        
        self.chat_obj = LLMAgent(self.deepseek_api_url, self.deepseek_api_key)
        
        # [Comment translated to English]
        self.debug = debug
        self.enable_search = enable_search and SEARCH_AVAILABLE
        self.enable_edu = enable_edu and SEARCH_AVAILABLE
        
        # Initialize search tool
        if self.enable_search and SEARCH_AVAILABLE:
            self.search_tool = SearchTool(SearchConfig())
            print(f"Search tool initialized. EDU parsing: {'enabled' if self.enable_edu else 'disabled'}")
        else:
            self.search_tool = None
            if enable_search and not SEARCH_AVAILABLE:
                print("Warning: Search requested but SearchTool not available")

    def _enhance_query_with_search(self, query: str) -> str:
        """Enhance query with search results"""
        if not self.enable_search or not self.search_tool:
            return query
        
        try:
            print(f"Performing search for query: {query[:100]}...")
            
            # Execute enhanced search (may include EDU parsing)
            search_results = self.search_tool.perform_enhanced_search(
                query=query, 
                enable_edu=self.enable_edu
            )
            
            # Format search results
            formatted_results = self.search_tool.format_enhanced_search_results(
                search_results, 
                enable_edu=self.enable_edu
            )
            
            # If search successful, enhance query
            if search_results.get("success") and formatted_results:
                search_type_desc = "Search results[Translated]EDUwebpage parsing" if self.enable_edu else "Search results"
                enhanced_query = f"""
                    Original question: {query}

                    Relevant{search_type_desc}:
                    {formatted_results}

                    Please combine the above search information with your knowledge to answerOriginal question。If search results conflict with your knowledge, please prioritize the latest information in search results。
                """.strip()
                
                print(f"Search enhancement completed. Found {len(search_results.get('data', {}).get('data', {}).get('web', []))} results")
                return enhanced_query
            else:
                print("Search failed or returned no results, using original query")
                return query
                
        except Exception as e:
            print(f"Search enhancement failed: {e}")
            return query

    def _forward_solver(self, query: str) -> str:
        """Solver - Enhanced version, supports search"""
        # If search enabled, enhance query first
        enhanced_query = self._enhance_query_with_search(query)
        
        # Format prompt with enhanced query
        user_prompt = SolverPrompt_User_Template.format(query=enhanced_query)
        assistant_prefix = SolverPrompt_Assistant_Template

        return self.chat_obj.call_model(user_prompt, assistant_prefix)

    def _forward_selector(self, query: str, candidates: List[str]) -> str:
        """Selector"""
        assert len(candidates) == 5, "selector expects exactly 5 candidates"

        user_prompt = SelectPrompt_User_Template.format(
            query=query, 
            solution_1=strip_think_and_exec(candidates[0]),
            solution_2=strip_think_and_exec(candidates[1]),
            solution_3=strip_think_and_exec(candidates[2]),
            solution_4=strip_think_and_exec(candidates[3]),
            solution_5=strip_think_and_exec(candidates[4])
        )
        assistant_prefix = SelectPrompt_Assistant_Template

        selector_response = self.chat_obj.call_model(user_prompt, assistant_prefix)
        m = re.search(r'<select>Response (\d+)</select>', selector_response)
        if not m:
            print("Warning: Could not parse selector's decision. Defaulting to Response 1.")
            idx = 0
        else:
            idx = max(0, min(4, int(m.group(1)) - 1))
        return candidates[idx]

    def forward(self, query: str) -> str:
        """Run the pipeline: solver(5, parallel) -> selector."""
        print("=== Agent Pipeline: Solver -> Selector ===")
        
        print("Step 1 | Solver : Generating 5 solutions in parallel...")
        solutions: List[str] = ["" for _ in range(5)]
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut2idx = {ex.submit(self._forward_solver, query): i for i in range(5)}
            for fut in as_completed(fut2idx):
                solutions[fut2idx[fut]] = fut.result()

        print("Step 2 | Selector : Selecting the best solution...")
        final_solution: str = self._forward_selector(query, solutions)

        if self.debug:
            dump = {
                "query": query,
                "solutions": solutions,
                "final_solution": final_solution,
                "search_enabled": self.enable_search,
                "edu_enabled": self.enable_edu,
            }
            with open("pipeline_debug.json", "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)
            print("Debug info saved to: pipeline_debug.json")

        return final_solution

    def batch_process(self, queries: List[str], max_workers: int = 3) -> List[str]:
        """Batch process queries"""
        print(f"Batch processing {len(queries)} queries with {max_workers} workers...")
        
        results = ["" for _ in range(len(queries))]
        
        def process_single_query(idx_query):
            idx, query = idx_query
            try:
                result = self.forward(query)
                return idx, result
            except Exception as e:
                print(f"Error processing query {idx}: {e}")
                return idx, f"Error: {str(e)}"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indexed_queries = [(i, query) for i, query in enumerate(queries)]
            future_to_idx = {
                executor.submit(process_single_query, idx_query): idx_query[0] 
                for idx_query in indexed_queries
            }
            
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                results[idx] = result
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get Agent statistics"""
        stats = {
            "search_enabled": self.enable_search,
            "edu_enabled": self.enable_edu,
            "search_available": SEARCH_AVAILABLE,
            "model_config": {
                "model": self.chat_obj.llm_config['model'],
                "max_tokens": self.chat_obj.llm_config['generation_config']['max_tokens'],
                "temperature": self.chat_obj.llm_config['generation_config']['temperature']
            }
        }
        return stats



def process_batch_with_resume(agent, args):
    """Batch processing with resume support - Adapt to original data format"""
    import os
    import json
    from threading import Lock
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. Load data to process
    print(f"Loading data from: {args.batch_file}")
    with open(args.batch_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 2. Validate data format
    if not isinstance(dataset, list):
        raise ValueError("Input data must be a list of items")
    
    # Check required fields
    required_fields = ['id', 'query', 'gt', 'category']
    sample_item = dataset[0] if dataset else {}
    missing_fields = [field for field in required_fields if field not in sample_item]
    if missing_fields:
        print(f"Warning: Missing fields in data: {missing_fields}")
    
    total = len(dataset)
    print(f"Loaded dataset with {total} items.")
    print(f"Data structure preview: {list(sample_item.keys()) if sample_item else 'Empty dataset'}")
    
    # 3. Set output file
    output_file = args.output_file or 'batch_results.jsonl'
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 4. Read processed data
    processed_ids = set()
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if line:  # [Comment translated to English]
                        item = json.loads(line)
                        if "id" in item:
                            processed_ids.add(item["id"])
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
        print(f"Found {len(processed_ids)} already processed items.")
    else:
        print("No existing results found, starting fresh.")
    
    # 5. File write lock (thread-safe)
    file_lock = Lock()
    
    def save_result(result):
        """Thread-safe result saving"""
        with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    def process_one_item(item):
        """Process single query"""
        try:
            item_id = item["id"]
            query = item["query"]
            gt = item.get("gt", "")
            category = item.get("category", "")
            
            print(f"🔄 Processing {item_id}: {query[:50]}...")
            
            # Call agent to process query
            response = agent.forward(query)
            
            # [Comment translated to English]
            result = {
                "id": item_id,
                "query": query,
                "gt": gt,
                "category": category,
                "response": response,  # New response field
                "status": "success",
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Preserve other fields from original data
            for key, value in item.items():
                if key not in result:
                    result[f"original_{key}"] = value
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {item.get('id', 'unknown')}: {e}")
            
            # Preserve original data structure even on error
            error_result = {
                "id": item.get("id", "unknown"),
                "query": item.get("query", ""),
                "gt": item.get("gt", ""),
                "category": item.get("category", ""),
                "response": None,  # Response is empty
                "status": "error",
                "error": str(e),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Preserve other original fields
            for key, value in item.items():
                if key not in error_result:
                    error_result[f"original_{key}"] = value
            
            return error_result
    
    # 6. Resume processing loop
    import time
    start_time = time.time()
    
    while len(processed_ids) < total:
        # Filter items to process
        to_be_processed = [item for item in dataset if item["id"] not in processed_ids]
        
        if not to_be_processed:
            break
        
        remaining = len(to_be_processed)
        completed = len(processed_ids)
        progress_pct = (completed / total) * 100
        
        print(f"\n📊 Progress: {completed}/{total} ({progress_pct:.1f}%)")
        print(f"🔄 Processing {remaining} remaining items...")
        
        # Parallel processing configuration
        max_workers = getattr(args, 'num_workers', 20)
        print(f"🚀 Using {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_one_item, item): item 
                for item in to_be_processed
            }
            
            # Collect results
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                
                try:
                    result = future.result()
                    
                    # Save result immediately
                    save_result(result)
                    processed_ids.add(item["id"])
                    
                    # Show progress
                    status = "✅" if result["status"] == "success" else "❌"
                    current_progress = len(processed_ids)
                    progress_pct = (current_progress / total) * 100
                    
                    print(f"{status} [{current_progress}/{total}] {item['id']} - {item['category']} ({progress_pct:.1f}%)")
                    
                except Exception as e:
                    print(f"💥 Fatal error processing {item.get('id', 'unknown')}: {e}")
    
    # 7. Processing completion statistics
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📊 Total processed: {len(processed_ids)}/{total}")
    print(f"⏱️  Total time: {duration:.1f} seconds")
    print(f"⚡ Average time per item: {duration/total:.1f} seconds")
    print(f"📁 Results saved to: {output_file}")
    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Agent with a query.")
    parser.add_argument('--query', type=str, 
                       default="What is the largest order of a non-cyclic torsion subgroup of an elliptic curve over $\\mathbb{Q}(\\sqrt{-3})$?", 
                       help='The query to process.')
    parser.add_argument('--enable_search', action='store_true', default=True, help='Enable search functionality')
    parser.add_argument('--enable_edu', action='store_true', default=True, help='Enable EDU parsing functionality')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--batch_file', type=str,  default= "hle.json", help='Process queries from JSON file')
    parser.add_argument('--output_file', type=str, default= "results/hle_8_22_gemini3_edu.jsonl", help='Output file for batch processing')
    
    args = parser.parse_args()
    
    # [Comment translated to English]
    if args.enable_edu:
        import os
        file_path, file_ext = os.path.splitext(args.output_file)
        args.output_file = f"{file_path}_edu{file_ext}"
    
    # [Comment translated to English]
    agent = Agent(
        debug=args.debug,
        enable_search=args.enable_search,
        enable_edu=args.enable_edu
    )
    
    # [Comment translated to English]
    stats = agent.get_stats()
    print(f"Agent Configuration:")
    print(f"  Model: {stats['model_config']['model']}")
    print(f"  Search: {'enabled' if stats['search_enabled'] else 'disabled'}")
    print(f"  EDU: {'enabled' if stats['edu_enabled'] else 'disabled'}")
    print(f"  Debug: {'enabled' if args.debug else 'disabled'}")
    print()
    
    if args.batch_file:
        process_batch_with_resume(agent, args)
        
    else:
        # [Comment translated to English]
        print(f"Processing query: {args.query}")
        result = agent.forward(args.query)
        print("\n" + "="*50)
        print("Final Result:")
        print("="*50)
        print(result)
