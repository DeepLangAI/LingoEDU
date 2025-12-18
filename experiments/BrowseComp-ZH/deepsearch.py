import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from run import BrowsecampEval 
from prompt import SYSTEM_PROMPT_CN
import argparse
from deeplang_parse.parse_client import AsyncParseClient, ParseAuther
from deeplang_parse.logger import logger
import os
import asyncio
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from deeplang_parse.enuming import TaskStatusEnum
from prompt import SYSTEM_PROMPT, JUDGE_PROMPT, SYSTEM_PROMPT_CN, JUDGE_PROMPT_CN

ANTHROPIC_API_KEY = ''
BASE_URL = ""
class BaseConfig:
    EduHost = ''
    AccessKeyId = ''
    AccessKeySecret = ''

Config = BaseConfig


class SearchConfig:
    def __init__(self):
        self.SEARCH_API_URL = ""
        self.DEFAULT_QUERY_TYPE = 'cloudsway'
        self.DEFAULT_CONTENT_TYPE = "HTML"
        self.DEFAULT_DATE = ''
        self.DEFAULT_TOP_K = 10
        self.EDU = False
        self.EDU_ACCESS_KEY_ID = getattr(BaseConfig, 'AccessKeyId', '')
        self.EDU_ACCESS_KEY_SECRET = getattr(BaseConfig, 'AccessKeySecret', '')
        self.EDU_ENDPOINT = getattr(BaseConfig, 'EduHost', '')
        self.DEFAULT_ENTRY_TYPE = 7


class SearchResult(BaseModel):
    search_tool: Optional[str] = None
    type: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None
    content: Optional[str] = None
    score: Optional[float] = None
    summary: Optional[str] = None
    snippet: Optional[str] = None
    site_name: Optional[str] = Field(None, alias="siteName")
    logo: Optional[str] = None
    image_list: Optional[List[str]] = Field(default_factory=list)
class SearchResponseData(BaseModel):
    query: str
    web: List[SearchResult]
class ApiResponse(BaseModel):
    code: int
    msg: str
    data: Optional[SearchResponseData] = None


class SearchTool:
    def __init__(self, config: SearchConfig):
        self.config = config
        self._init_edu_client()

    def _init_edu_client(self):
        """Initialize EDU API"""
        try:
            self.edu_auther = ParseAuther(
                access_key_id=self.config.EDU_ACCESS_KEY_ID,
                access_key_secret=self.config.EDU_ACCESS_KEY_SECRET,
                endpoint=self.config.EDU_ENDPOINT
            )
            self.edu_auther.auth_v1()
            
            self.edu_client = AsyncParseClient(endpoint=self.config.EDU_ENDPOINT)
            print("EDU API client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize EDU API client: {e}")
            self.edu_client = None

    def perform_edu_parse(self, url: str, timeout: int = 60) -> Dict[str, Any]:
        """Parse URL content using EDU API (synchronous version with polling)"""
        if not self.edu_client:
            return {"success": False, "error": "EDU API client not initialized"}
        
        try:
            resp = self.edu_client.create_task(
                entry_type=self.config.DEFAULT_ENTRY_TYPE,
                entry_url=url
            )
            
            if not resp:
                return {"success": False, "error": "Failed to create EDU parsing task"}
            
            task_id = resp.data.task_id
            print(f'EDU task created successfully, task_id: {task_id}')
            
            # 2. Poll task status
            start_time = time.time()
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    return {
                        "success": False, 
                        "error": f"EDU parsing timeout after {timeout} seconds",
                        "task_id": task_id
                    }
                
                # Query task status
                status_resp = self.edu_client.query_task_status(task_id=task_id)
                if not status_resp:
                    print(f'task_id: {task_id}, query task status failed')
                    time.sleep(1)
                    continue
                
                status = status_resp.data.status
                
                if status == TaskStatusEnum.Failed.value:
                    return {
                        "success": False, 
                        "error": f"EDU parsing task failed",
                        "task_id": task_id
                    }
                elif status == TaskStatusEnum.Success.value:
                    print(f'task_id: {task_id}, task completed successfully')
                    break
                else:
                    print(f'task_id: {task_id}, task is running, status: {status}')
                    time.sleep(1)  # Wait 1 second before querying again
            
            # 3. Fetch parsing result
            result_resp = self.edu_client.fetch_task_result(
                task_id=task_id, 
                with_markdown=True
            )
            
            if not result_resp:
                return {
                    "success": False, 
                    "error": "Failed to fetch EDU parsing result",
                    "task_id": task_id
                }
            
            parsed_content = result_resp.data.model_dump_json(indent=4)
            return {
                "success": True, 
                "data": parsed_content, 
                "task_id": task_id
            }
            
        except Exception as e:
            error_msg = f"EDU API parsing error: {e}"
            return {"success": False, "error": error_msg}
    
           
    
    def perform_enhanced_search(self, query: str, enable_edu: bool = False) -> Dict[str, Any]:
        """Enhanced search method (synchronous version), combining regular search and EDU parsing"""
        # 1. Perform regular search first
        search_results = self.perform_search(query)
        
        if not enable_edu or not search_results.get("success"):
            return search_results
        
        # 2. If EDU enabled and search successful, perform EDU parsing on top URLs
        try:
            web_results = search_results["data"]["data"]["web"]  # Perform EDU parsing on results
            
            # Use thread pool to process EDU parsing in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit EDU parsing tasks
                future_to_result = {}
                for i, result in enumerate(web_results):
                    url = result.get("url", "")
                    if url:
                        future = executor.submit(self.perform_edu_parse, url, timeout=30)
                        future_to_result[future] = i
                
                # Collect results
                for future in as_completed(future_to_result, timeout=60):
                    result_index = future_to_result[future]
                    try:
                        edu_result = future.result()
                        result = web_results[result_index]
                        
                        if edu_result.get("success"):
                            result["edu_parsed_content"] = edu_result["data"]
                            result["edu_task_id"] = edu_result.get("task_id")
                            print(f"EDU parsing completed for: {result.get('url', '')}")
                        else:
                            result["edu_parse_error"] = edu_result.get("error")
                            print(f"EDU parsing failed for: {result.get('url', '')}, error: {edu_result.get('error')}")
                    except Exception as e:
                        result = web_results[result_index]
                        result["edu_parse_error"] = f"EDU parsing exception: {str(e)}"
                        print(f"EDU parsing exception for: {result.get('url', '')}, error: {str(e)}")
            
            # Update search results
            search_results["data"]["data"]["web"] = web_results
            search_results["edu_enhanced"] = True
            
        except Exception as e:
            print(f"EDU enhancement failed: {e}")
            search_results["edu_enhancement_error"] = str(e)
        
        return search_results
    
    def format_enhanced_search_results(self, search_results: Dict[str, Any], enable_edu: bool = False) -> str:
        """Format enhanced search results"""
        if not search_results.get("success"):
            return f"Search failed: {search_results.get('error', 'Unknown error')}"
        
        data = search_results.get("data", {})
        if not data.get("data") or not data["data"].get("web"):
            return "No relevant search results found"
        
        formatted_results = []
        for i, item in enumerate(data["data"]["web"][:5], 1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "")
            snippet = item.get("snippet", "")
            
            # Basic content
            description = snippet or content[:300]
            
            result_text = f"""
                        Result {i}:
                        Title: {title}
                        Link: {url}
                        Content summary: {description}"""
            # If EDU parsed content exists, add to results
            if enable_edu and item.get("edu_parsed_content"):
                edu_content = item["edu_parsed_content"]
                # Truncate EDU content to first 500 characters as preview
                edu_preview = edu_content[:500] + "..." if len(edu_content) > 500 else edu_content
                result_text += f"""EDU parsed content: {edu_preview}"""
            
            elif enable_edu and item.get("edu_parse_error"):
                result_text += f"""EDU parsing failed: {item['edu_parse_error']}"""
            
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
    


    def perform_search(self, query: str) -> Dict[str, Any]:
        """Execute search request"""
        payload = {
            "query": query,
            "query_type": self.config.DEFAULT_QUERY_TYPE,
            "content_type": self.config.DEFAULT_CONTENT_TYPE,
            "date": self.config.DEFAULT_DATE,
            "top_k": self.config.DEFAULT_TOP_K,
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                self.config.SEARCH_API_URL, 
                json=payload, 
                headers=headers, 
                timeout=60.0
            )
            response.raise_for_status()
            raw_response = response.json()
            
            # Filter out results with empty content
            if raw_response.get('data') and raw_response['data'].get('web'):
                filtered_results = [
                    result for result in raw_response['data']['web']
                    if result.get('content') and result['content'].strip()
                ]
                raw_response['data']['web'] = filtered_results
            
            return {"success": True, "data": raw_response}
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error in search: {e}"
            return {"success": False, "error": error_msg}
    
    def format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results as LLM readable text"""
        if not search_results.get("success"):
            return f"Search failed: {search_results.get('error', 'Unknown error')}"
        
        data = search_results.get("data", {})
        if not data.get("data") or not data["data"].get("web"):
            return "No relevant search results found"
        
        formatted_results = []
        for i, item in enumerate(data["data"]["web"][:5], 1):  # Only take top 5 results
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "")
            snippet = item.get("snippet", "")
            
            # Use first 300 characters of snippet or content
            description = snippet or content[:300]
            
            formatted_results.append(f"""
                Result {i}:
                Title: {title}
                Link: {url}
                Content summary: {description}
                """)
        
        return "\n".join(formatted_results)
    

class BrowsecampEvalWithSearch(BrowsecampEval):
    def __init__(self, args):
        super().__init__(args)
        self.search_tool = SearchTool(SearchConfig())
        self.enable_search = getattr(args, 'enable_search', False)
        self.enable_edu = getattr(args, 'enable_edu', False)
        self._setup_search_logging()
        # Add thread lock for logging
        self._search_lock = Lock()
        self._original_model = args.model
  
    def _setup_search_logging(self):
        """Setup search related logging"""
        import logging
      
        # Create search-specific logger
        self.search_logger = logging.getLogger('search_enhancer')
        self.search_logger.setLevel(logging.INFO)
      
        # Avoid duplicate handlers
        if not self.search_logger.handlers:
            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
          
            file_handler = logging.FileHandler(log_dir / "search_enhancement.log", encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
          
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
          
            self.search_logger.addHandler(file_handler)
            self.search_logger.addHandler(console_handler)
  
    def _create_search_enhanced_messages(self, question: str) -> List[Dict[str, str]]:
        """Create enhanced messages with search results (synchronous version, EDU support)"""
        if self.enable_search:
            with self._search_lock:
                self.search_logger.info(f"Performing search for question: {question}")
                self.search_logger.info(f"EDU parsing enabled: {self.enable_edu}")
          
            # Execute enhanced search (may include EDU parsing)
            search_results = self.search_tool.perform_enhanced_search(
                query=question, 
                enable_edu=self.enable_edu
            )
            formatted_results = self.search_tool.format_enhanced_search_results(
                search_results, 
                enable_edu=self.enable_edu
            )
          
            # Create enhanced system prompt
            search_type_desc = "search results and EDU webpage parsing" if self.enable_edu else "search results"
            enhanced_system_prompt = f"""{SYSTEM_PROMPT_CN}
                You now have access to the following latest {search_type_desc} to help answer questions:
                {formatted_results}
              
                {'Note: EDU parsed content contains structured information from webpages, please pay special attention to these details.' if self.enable_edu else ''}
                Please combine this information with your knowledge to answer the question. If the information in the search results conflicts with your knowledge, prioritize the latest information in the search results.
                Please ensure your answer format still follows these requirements:
                **Explanation**: [Detailed explanation process]
                **Exact Answer**: [Accurate answer]
                **Confidence**: [Confidence percentage]
            """
          
            search_result_count = len(search_results.get('data', {}).get('data', {}).get('web', []))
            edu_enhanced_count = sum(1 for item in search_results.get('data', {}).get('data', {}).get('web', []) 
                                   if item.get('edu_parsed_content'))
          
            with self._search_lock:
                self.search_logger.info(f"Search completed, found {search_result_count} results")
                if self.enable_edu:
                    self.search_logger.info(f"EDU enhanced {edu_enhanced_count} results")
          
            return [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": question}
            ]
        else:
            return [
                {"role": "system", "content": SYSTEM_PROMPT_CN},
                {"role": "user", "content": question}
            ]

    
    def get_multi_infer_response(self):
        """Synchronous version of inference method (supports search+EDU)"""
        import concurrent.futures
        from tqdm import tqdm
        import re
        
        # 1. Load data
        with open(self.args.input_file_path, 'r', encoding='utf-8') as f:
            querys = json.load(f)
        
        # 2. Create enhanced messages in parallel (including Search and EDU Parse)
        print("Creating enhanced messages with search and EDU parsing...")
        
        def create_single_message(query_data):
            """Message creation function for single query"""
            try:
                messages = self._create_search_enhanced_messages(query_data["Question"])
                return {
                    'messages': messages,
                    'question': query_data["Question"],
                    'answer': query_data["Answer"],
                    'success': True
                }
            except Exception as e:
                print(f"Error creating message for question: {query_data['Question'][:50]}..., error: {e}")
                return {
                    'messages': [
                        {"role": "system", "content": SYSTEM_PROMPT_CN},
                        {"role": "user", "content": query_data["Question"]}
                    ],
                    'question': query_data["Question"],
                    'answer': query_data["Answer"],
                    'success': False,
                    'error': str(e)
                }
        
        # Create messages in parallel (includes search and EDU parsing)
        chat_datas = []
        max_search_workers = 3 if self.enable_edu else 6  # EDU parsing is slower, reduce concurrency
        # max_search_workers = 1
        with ThreadPoolExecutor(max_workers=max_search_workers) as executor:
            message_futures = [
                executor.submit(create_single_message, query) 
                for query in querys
            ]
            
            for future in tqdm(as_completed(message_futures), 
                            desc="Creating enhanced messages", 
                            total=len(message_futures)):
                chat_data = future.result()
                chat_datas.append(chat_data)
        
        # Sort by original order (as_completed doesn't guarantee order)
        question_to_index = {q["Question"]: i for i, q in enumerate(querys)}
        chat_datas.sort(key=lambda x: question_to_index[x['question']])
        
        # 3. Generate directory name based on features
        model_suffix = self._get_model_suffix()
        predict_infer_file_dir = Path(self.args.predict_infer_file_dir) / model_suffix
        predict_infer_file_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Concurrent LLM inference - modify this part to avoid concurrent write issues
        print(f"Starting LLM inference with {self.args.max_workers} workers...")
        
        # Collect results in dictionary to ensure order
        results = {}
        results_lock = Lock()  # Add lock to protect results dictionary
        
        def process_inference(index_and_chat_data):
            """Process single inference task"""
            idx, chat_data = index_and_chat_data
            try:
                response, total_tokens = self.get_remote_response(chat_data['messages'])
            except Exception as e:
                print(f"Error in LLM inference for question {idx}: {e}")
                response, total_tokens = "Error in inference", 0
            
            if response is None:
                response = "None"
            if total_tokens is None:
                total_tokens = 0
            
            # Parse response
            pattern = r"""
                \*{0,2}Explanation\*{0,2}\s*?:\s*(.*?)\n
                \*{0,2}Exact\sAnswer\*{0,2}\s*?:\s*(.*?)\n
                \*{0,2}Confidence\*{0,2}\s*?:\s*(.*?)$
            """
            matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)
            if matches:
                explanation = matches.group(1).strip()
                exact_answer = matches.group(2).strip()
                confidence = matches.group(3).strip()
            else:
                explanation, exact_answer, confidence = "", "", ""
            
            result = {
                "question": chat_data['question'], 
                "answer": chat_data['answer'], 
                "response": response,
                "explanation": explanation, 
                "exact_answer": exact_answer, 
                "confidence": confidence, 
                "total_tokens": total_tokens,
                "used_search": self.enable_search,
                "used_edu": self.enable_edu,
                "message_creation_success": chat_data.get('success', True),
                "question_index": idx  # add index for subsequent analysis
            }
            
            # Record error if message creation fails
            if not chat_data.get('success', True):
                result["message_creation_error"] = chat_data.get('error')
            
            # Thread-safe result saving
            with results_lock:
                results[idx] = result
            
            return idx, result
        
        # Execute inference concurrently
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            indexed_chat_datas = [(i, chat_data) for i, chat_data in enumerate(chat_datas)]
            
            desc = self._get_progress_desc()
            
            # Submit all tasks
            future_to_index = {}
            for idx, chat_data in indexed_chat_datas:
                future = executor.submit(process_inference, (idx, chat_data))
                future_to_index[future] = idx
            
            # Wait for all tasks to complete
            for future in tqdm(as_completed(future_to_index.keys()), 
                            desc=desc, 
                            total=len(future_to_index)):
                try:
                    idx, result = future.result()
                    # Results already saved to results dictionary in process_inference
                except Exception as e:
                    original_index = future_to_index[future]
                    print(f"Error processing question {original_index}: {e}")
        
        # 5. Write all results at once (in order)
        with open(predict_infer_file_dir / 'infer.jsonl', 'w', encoding='utf-8') as f:
            for i in range(len(chat_datas)):
                if i in results:
                    f.write(json.dumps(results[i], ensure_ascii=False) + '\n')
                else:
                    print(f"Warning: Missing result for question {i}")
        
        # 6. Record completion status
        if self.enable_search:
            successful_searches = sum(1 for chat_data in chat_datas if chat_data.get('success', True))
            with self._search_lock:
                self.search_logger.info(f"Completed inference for {len(chat_datas)} questions")
                self.search_logger.info(f"Successful search enhancements: {successful_searches}/{len(chat_datas)}")
                if self.enable_edu:
                    self.search_logger.info("EDU parsing was enabled for this run")
        
        print(f"Inference completed! Results saved to: {predict_infer_file_dir / 'infer.jsonl'}")
        print(f"Total results written: {len(results)}")  # Add this line to verify result count


    def _get_model_suffix(self) -> str:
        """Generate model suffix based on enabled features"""
        suffix = self.args.model
        if self.enable_search:
            suffix += "_with_search"
        if self.enable_edu:
            suffix += "_with_edu"
        return suffix
  
    def _get_progress_desc(self) -> str:
        """Generate progress bar description"""
        desc = "Generating responses"
        if self.enable_search:
            desc += " with search"
        if self.enable_edu:
            desc += " and EDU"
        return desc
    
    def generate_infer_eval(self):
        """Generate inference evaluation results (completely rewritten to adapt to new directory structure and preserve original files)"""
        import concurrent.futures
        import re
        import copy
        from tqdm import tqdm
        
        # 1. Load original data
        with open(self.args.input_file_path, 'r', encoding='utf-8') as f:
            raw_datas = json.load(f)
        raw_datas_copy = copy.deepcopy(raw_datas)

        # 2. Read inference results
        model_suffix = self._get_model_suffix()
        predict_infer_file_dir = Path(self.args.predict_infer_file_dir) / model_suffix
        infer_file = predict_infer_file_dir / 'infer.jsonl'
        
        if not infer_file.exists():
            print(f"Inference file not found: {infer_file}")
            return
        
        # Read inference data (don't delete file)
        infer_datas = []
        with open(infer_file, 'r', encoding='utf-8') as f:
            for line in f:
                infer_datas.append(json.loads(line))
        
        print(f"Loaded {len(infer_datas)} inference results")
        
        # 3. Create evaluation messages
        messages = []
        for i, infer_data in enumerate(infer_datas):
            message = [
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": JUDGE_PROMPT_CN.format(
                    question=infer_data['question'], 
                    response=infer_data['response'], 
                    correct_answer=infer_data['answer']
                )},
            ]
            messages.append(message)
        
        # 4. Create output directory (in same directory)
        eval_file_path = predict_infer_file_dir / 'eval.jsonl'
        output_file_path = predict_infer_file_dir / 'final_output.jsonl'
        
        # Remove evaluation files if they exist
        if eval_file_path.exists():
            eval_file_path.unlink()
        if output_file_path.exists():
            output_file_path.unlink()
        
        print(f"Starting evaluation with {self.args.max_workers} workers...")
        
        # 5. Execute evaluation in parallel
        results = {}
        results_lock = Lock()
        
        def process_evaluation(index_and_message):
            """Process single evaluation task"""
            idx, message = index_and_message
            try:
                # Assume get_remote_response can handle evaluation requests
                response, _ = self.get_remote_response(message)
                
                # Parse evaluation response
                pattern = r"""
                    \*{0,2}extracted_final_answer\*{0,2}\s*?:\s*(.*?)\n
                    \*{0,2}reasoning\*{0,2}\s*:\s*?(.*?)\n
                    \*{0,2}correct\*{0,2}\s*:\s*?(.*?)\n
                    \*{0,2}confidence\*{0,2}\s*?:\s*(.*?)$
                """
                matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)

                if matches:
                    model_extracted_answer = matches.group(1).strip()
                    reasoning = matches.group(2).strip()
                    is_correct = matches.group(3).strip()
                    model_extracted_confidence = matches.group(4).strip()
                else:
                    model_extracted_answer, reasoning, is_correct, model_extracted_confidence = "", "", "", ""
                
                # Create evaluation result
                eval_result = {
                    "model_extracted_answer": model_extracted_answer,
                    "model_prediction": infer_datas[idx]["response"],
                    "is_correct": is_correct,
                    "model_extracted_confidence": model_extracted_confidence
                }
                
                chat_data = {
                    "question": raw_datas_copy[idx]['Question'],
                    "answer": raw_datas_copy[idx]['Answer'],
                    "response": response,
                    "model_extracted_answer": model_extracted_answer,
                    "reasoning": reasoning,
                    "is_correct": is_correct,
                    "model_extracted_confidence": model_extracted_confidence
                }
                
                # Update original data
                final_data = raw_datas_copy[idx].copy()
                final_data["eval_result"] = [eval_result]
                
                # Thread-safe result saving
                with results_lock:
                    results[idx] = {
                        'eval_data': chat_data,
                        'final_data': final_data
                    }
                
                return idx
                
            except Exception as e:
                print(f"Error in evaluation for question {idx}: {e}")
                with results_lock:
                    results[idx] = {
                        'eval_data': None,
                        'final_data': None,
                        'error': str(e)
                    }
                return idx
        
        # 6. Execute parallel evaluation
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            indexed_messages = [(i, message) for i, message in enumerate(messages)]
            
            future_to_index = {}
            for idx, message in indexed_messages:
                future = executor.submit(process_evaluation, (idx, message))
                future_to_index[future] = idx
            
            # Wait for all tasks to complete
            for future in tqdm(as_completed(future_to_index.keys()), 
                            desc="Generating infer eval responses", 
                            total=len(future_to_index)):
                try:
                    idx = future.result()
                except Exception as e:
                    original_index = future_to_index[future]
                    print(f"Error processing evaluation {original_index}: {e}")
        
        # 7. Write results in order
        with open(eval_file_path, 'w', encoding='utf-8') as eval_f, \
            open(output_file_path, 'w', encoding='utf-8') as final_f:
            
            for i in range(len(infer_datas)):
                if i in results and not results[i].get('error'):
                    result = results[i]
                    if result['eval_data']:
                        eval_f.write(json.dumps(result['eval_data'], ensure_ascii=False) + '\n')
                    if result['final_data']:
                        final_f.write(json.dumps(result['final_data'], ensure_ascii=False) + '\n')
                else:
                    print(f"Warning: Missing or failed evaluation for question {i}")
        
        # 8. Output result statistics
        successful_evals = sum(1 for r in results.values() if not r.get('error'))
        print(f"Evaluation completed!")
        print(f"Successful evaluations: {successful_evals}/{len(infer_datas)}")
        print(f"Results saved to:")
        print(f"  - Evaluation details: {eval_file_path}")
        print(f"  - Final output: {output_file_path}")
        print(f"  - Original infer file preserved: {infer_file}")
        
        # Check generated files
        files_in_dir = list(predict_infer_file_dir.glob("*"))
        print(f"Files in directory: {files_in_dir}")


    def _generate_search_report(self, infer_file: Path):
        """Generate search enhancement statistics report"""
        import json
        
        results = []
        try:
            with open(infer_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        except Exception as e:
            print(f"Error reading inference results: {e}")
            return
        
        if not results:
            print("No results found in inference file")
            return
        
        # Statistics
        total_questions = len(results)
        successful_searches = sum(1 for r in results if r.get('message_creation_success', True))
        search_errors = sum(1 for r in results if not r.get('message_creation_success', True))
        
        # Calculate token usage statistics
        total_tokens = sum(r.get('total_tokens', 0) for r in results)
        avg_tokens = total_tokens / total_questions if total_questions > 0 else 0
        
        # Prepare report
        report = {
            "search_enhancement_summary": {
                "total_questions": total_questions,
                "successful_search_enhancements": successful_searches,
                "search_enhancement_errors": search_errors,
                "search_success_rate": successful_searches / total_questions if total_questions > 0 else 0,
                "search_enabled": self.enable_search,
                "edu_enabled": self.enable_edu,
                "total_tokens_used": total_tokens,
                "average_tokens_per_question": round(avg_tokens, 2)
            }
        }
        
        if search_errors > 0:
            error_examples = []
            for r in results:
                if not r.get('message_creation_success', True):
                    question_preview = r["question"][:100] + "..." if len(r["question"]) > 100 else r["question"]
                    error_examples.append({
                        "question": question_preview,
                        "error": r.get("message_creation_error", "Unknown error")
                    })
                    if len(error_examples) >= 5:  # Only show first 5 error examples
                        break
            
            report["search_enhancement_summary"]["error_examples"] = error_examples
        
        # Add confidence statistics if confidence info available
        confidence_scores = []
        for r in results:
            confidence_str = r.get('confidence', '').strip()
            if confidence_str:
                # Try to extract numbers (support percentage format)
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', confidence_str)
                if match:
                    confidence_scores.append(float(match.group(1)))
        
        if confidence_scores:
            report["search_enhancement_summary"]["confidence_stats"] = {
                "average_confidence": round(sum(confidence_scores) / len(confidence_scores), 2),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores),
                "confidence_count": len(confidence_scores)
            }
        
        # Save report
        report_file = infer_file.parent / 'search_enhancement_report.json'
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"Search enhancement report saved to: {report_file}")
            print(f"Search Success Rate: {report['search_enhancement_summary']['search_success_rate']:.2%}")
            print(f"Total Questions: {total_questions}")
            print(f"Average Tokens per Question: {avg_tokens:.2f}")
            
            if confidence_scores:
                avg_conf = report["search_enhancement_summary"]["confidence_stats"]["average_confidence"]
                print(f"Average Confidence: {avg_conf}%")
            
            if self.enable_search:
                with self._search_lock:
                    self.search_logger.info(f"Generated search enhancement report: {report_file}")
                    self.search_logger.info(f"Search success rate: {report['search_enhancement_summary']['search_success_rate']:.2%}")
                    
        except Exception as e:
            print(f"Error saving search enhancement report: {e}")
    
    def run_full_evaluation(self):
        """Run full evaluation process (including inference and evaluation)"""
        print("Starting full evaluation with search enhancement...")
        
        # 1. Generate inference results
        print("Step 1: Generating inference responses...")
        self.get_multi_infer_response()
        
        # 2. Generate evaluation results
        print("Step 2: Generating evaluation metrics...")
        self.generate_infer_eval()
        
        print("Full evaluation completed!")
        
        # 3. Display result summary
        model_suffix = self._get_model_suffix() 
        results_dir = Path(self.args.predict_infer_file_dir) / model_suffix
        
        print(f"\nResults saved to: {results_dir}")
        print("Files generated:")
        print(f"- Inference results: {results_dir / 'infer.jsonl'}")
        print(f"- Evaluation scores: {results_dir / 'score.json'}")
        if self.enable_search:
            print(f"- Search enhancement report: {results_dir / 'search_enhancement_report.json'}")

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Claude4")
    parser.add_argument('--input_file_path', type=str, default="raw_data/browsecomp-zh-decrypted.json")
    parser.add_argument('--predict_infer_file_dir', type=str, default="predict_infer")
    parser.add_argument('--eval_infer_file_dir', type=str, default="eval_infer")
    parser.add_argument('--output_infer_file_dir', type=str, default="output_infer")
    parser.add_argument('--max_workers', type=int, default=5)
    
    parser.add_argument('--mode', type=str, default='all', choices=['infer', 'eval', 'all'])
    parser.add_argument('--enable_search', default=True, action='store_true', help='Enable online search functionality')

    parser.add_argument("--enable_edu", default=True, action="store_true", help="Enable EDU API for webpage parsing")
    parser.add_argument("--edu_entry_type", type=int, default=7, help="EDU API entry type")

    # Execution control parameters
    # parser.add_argument('--inference_only', action='store_true', help='Only run inference, skip evaluation')
    # parser.add_argument('--eval_only', action='store_true', help='Only run evaluation, skip inference')
   
    
    args = parser.parse_args()
    
    # Create evaluation instance with search capability
    evaluator = BrowsecampEvalWithSearch(args)
    
    print(f"Model: {args.model}")
    print(f"Search function: {'Enabled' if args.enable_search else 'Disabled'}")
    print(f"Run mode: {args.mode}")
    print(f"EDU mode: {args.enable_edu}")
    
    if args.mode == 'infer':
        print("Starting inference phase...")
        evaluator.get_multi_infer_response()
        print("Inference completed!")
        
    elif args.mode == 'eval':
        print("Starting evaluation phase...")
        evaluator.generate_infer_eval() 
        print("Evaluation completed!")
        
    elif args.mode == 'all':
        print("Starting inference phase...")
        evaluator.get_multi_infer_response()
        print("Inference completed!")
        
        print("Starting evaluation phase...")
        evaluator.generate_infer_eval()
        print("Evaluation completed!")
    
    if args.enable_search:
        print(f"Results saved to: {args.predict_infer_file_dir}/{args.model}_with_search/")
    else:
        print(f"Results saved to: {args.predict_infer_file_dir}/{args.model}/")
