import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from deeplang_parse.parse_client import AsyncParseClient, ParseAuther
from deeplang_parse.logger import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from deeplang_parse.enuming import TaskStatusEnum
import json
# from DeepLang.dl_pro.agents.model import LLM

# --- Search related configuration and model ---
OPENAI_ENDPOINT = ""
OPENAI_API_KEY = ''

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
        # EDU API configuration
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
    # New EDU parsing related fields
    edu_title: Optional[str] = None
    edu_content: Optional[str] = None
    edu_key_points: Optional[List[str]] = Field(default_factory=list)
    edu_structure: Optional[Dict[str, Any]] = None
    edu_markdown: Optional[str] = None
    llm_summary: Optional[str] = None  # New LLM-generated summary
class SearchResponseData(BaseModel):
    query: str
    web: List[SearchResult]
class ApiResponse(BaseModel):
    code: int
    msg: str
    data: Optional[SearchResponseData] = None
# --- Search tool class ---
class SearchTool:
    def __init__(self, config: SearchConfig, llm_model: str = "gemini-2.5-pro"):
        self.config = config
        self.llm_model = llm_model
        self.llm_client = None
        self._init_edu_client()
        self._init_llm_client()
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
            print("✅ EDU API client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize EDU API client: {e}")
            self.edu_client = None
    def _init_llm_client(self):
        """Initialize LLM client"""
        try:
            
            self.llm_client = LLM(call_target=f"gpt:{self.llm_model}")
            print("✅ LLM client initialized successfully")
        except Exception as e:
            print(f"⚠️ LLM client initialization failed: {e}")
            print("🔄 Will use fallback summary mode")
            self.llm_client = None
    def _extract_edu_content_elements(self, parsed_data_str: str) -> Dict[str, Any]:
        """
        Extract content elements from EDU parsed data (using hierarchical parsing method)
        """
        try:
            parsed_data = json.loads(parsed_data_str) if isinstance(parsed_data_str, str) else parsed_data_str
        except json.JSONDecodeError:
            return {
                'title': '',
                'main_content': '',
                'key_points': [],
                'markdown': '',
                'structure_info': {}
            }
        
        sentences = parsed_data.get('sentences', [])
        markdown = parsed_data.get('markdown', '')
        
        # [Comment translated to English]
        title = ""
        main_contents = []
        key_points = []
        
        # [Comment translated to English]
        for sentence in sentences:
            level = sentence.get('level', 1)
            text = sentence.get('text', '').strip()
            
            if not text:
                continue
                
            # [Comment translated to English]
            if level == 1 and text and not title:
                title = text
            
            # [Comment translated to English]
            if text and text != title:
                # [Comment translated to English]
                indent = "  " * (level - 1) if level > 1 else ""
                formatted_text = f"{indent}{text}"
                main_contents.append(formatted_text)
                
                # [Comment translated to English]
                if 2 <= level <= 3 and len(text) > 10:
                    key_points.append(text)
        
        # [Comment translated to English]
        main_content = "\n".join(main_contents) if main_contents else ""
        
        # [Comment translated to English]
        if not main_content and markdown:
            main_content = markdown
            # [Comment translated to English]
            if not title:
                lines = markdown.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('#'):
                        title = line.lstrip('#').strip()
                        break
        
        return {
            'title': title,
            'main_content': main_content,
            'key_points': key_points[:5],  # [Comment translated to English]
            'markdown': markdown,
            'structure_info': {
                'total_sentences': len(sentences),
                'has_hierarchy': any(s.get('level', 1) > 1 for s in sentences),
                'max_level': max((s.get('level', 1) for s in sentences), default=1)
            }
        }
    def _build_llm_summary_prompt(self, url: str, content_elements: Dict[str, Any], query: str = "") -> str:
        """
        Build prompt for LLM summary
        """
        title = content_elements.get('title', '')
        main_content = content_elements.get('main_content', '')
        key_points = content_elements.get('key_points', [])
        
        # [Comment translated to English]
        prompt = f"""Please generate a professional summary for search results based on the following hierarchical content structure of the webpage:
**Search query:** {query}
**Webpage URL:** {url}
**Webpage title:** {title or '[Translated]'}
**Webpage content hierarchical structure:**
{main_content if main_content else '[Translated]TextContent'}
**Extracted key points:**
{chr(10).join([f"• {point}" for point in key_points]) if key_points else '[Translated]'}
**Summary requirements:**
1. analyzeWebpageContentandSearch query'sRelevantrelevance
2. provide[Translated]concisebutcomprehensive'sContentsummary（100-200words）
3. highlightandSearch querymostRelevant'sinformation
4. Identify and list 3-5keypoints
5. Evaluate the value and credibility of information
**Output format:**
Please strictly follow the following JSONformatreturn：
{{
    "summary": "[Translated]，focusattentionandSearch query'sRelevantrelevance",
    "key_points": ["[Translated]1", "[Translated]2", "[Translated]3"],
    "relevance_score": "[Translated]RelevantRelevance score (1-10)",
    "content_quality": "[Translated]（High/Medium/Low）",
    "main_topics": ["[Translated]1", "[Translated]2"]
}}
please[Translated]summary[Translated]、[Translated]，focushighlightandSearch query"{query}"[Translated]Relevantrelevance。"""
        return prompt
    def _call_llm_for_summary(self, prompt: str) -> Dict[str, Any]:
        """
        [Translated]LLMgeneratesummary
        """
        try:
            print("🤖 [Translated]LLMgenerate[Translated]summary...")
            
            if not self.llm_client:
                print("⚠️ [Translated]LLM[Translated]，return[Translated]summary")
                return {"use_fallback": True}
            
            # [Comment translated to English]
            messages = [
                {
                    "role": "system",
                    "content": "[Translated]Search resultsanalyze[Translated]，[Translated]analyzeWebpageContentandgenerateandSearch queryhigh[Translated]Relevant'ssummary。please[Translated]returnhave[Translated]'sJSONformat。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # [Comment translated to English]
            llm_response = self.llm_client(messages)
            
            if not llm_response or llm_response.strip() == "":
                print("⚠️ LLM[Translated]，Use[Translated]summary")
                return {"use_fallback": True}
            
            # [Comment translated to English]
            try:
                # [Comment translated to English]
                cleaned_response = llm_response.strip()
                
                # [Comment translated to English]
                if '```json' in cleaned_response:
                    start_idx = cleaned_response.find('```json') + 7
                    end_idx = cleaned_response.find('```', start_idx)
                    if end_idx != -1:
                        cleaned_response = cleaned_response[start_idx:end_idx].strip()
                elif '```' in cleaned_response:
                    start_idx = cleaned_response.find('```') + 3
                    end_idx = cleaned_response.find('```', start_idx)
                    if end_idx != -1:
                        cleaned_response = cleaned_response[start_idx:end_idx].strip()
                
                # [Comment translated to English]
                if '{' in cleaned_response and '}' in cleaned_response:
                    start_idx = cleaned_response.find('{')
                    end_idx = cleaned_response.rfind('}') + 1
                    cleaned_response = cleaned_response[start_idx:end_idx]
                
                # [Comment translated to English]
                summary_data = json.loads(cleaned_response)
                print("✅ LLM[Translated]success")
                return summary_data
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON[Translated]failed: {e}，[Translated]extractkeyinformation")
                # [Comment translated to English]
                return {
                    "summary": llm_response[:250] + "..." if len(llm_response) > 250 else llm_response,
                    "key_points": ["[Translated]LLMText[Translated]extract'sinformation"],
                    "relevance_score": "[Translated]",
                    "content_quality": "[Translated]",
                    "main_topics": ["[Translated]"]
                }
                
        except Exception as e:
            print(f"❌ LLM[Translated]failed: {e}")
            return {"use_fallback": True}
    def _generate_smart_summary(self, content_elements: Dict[str, Any], original_snippet: str = "", query: str = "", url: str = "") -> Dict[str, Any]:
        """
        based onEDUparse/parsingContentgenerate[Translated]summary（[Translated]LLM）
        """
        # [Comment translated to English]
        if self.llm_client:
            prompt = self._build_llm_summary_prompt(url, content_elements, query)
            llm_result = self._call_llm_for_summary(prompt)
            
            if not llm_result.get("use_fallback", False):
                return {
                    "summary": llm_result.get("summary", ""),
                    "key_points": llm_result.get("key_points", []),
                    "source": "llm",
                    "relevance_score": llm_result.get("relevance_score", ""),
                    "content_quality": llm_result.get("content_quality", ""),
                    "main_topics": llm_result.get("main_topics", [])
                }
        
        # [Comment translated to English]
        print("📝 Use[Translated]")
        
        title = content_elements.get('title', '')
        main_content = content_elements.get('main_content', '')
        key_points = content_elements.get('key_points', [])
        
        # [Comment translated to English]
        summary_parts = []
        
        # [Comment translated to English]
        if title and title != original_snippet[:50]:
            summary_parts.append(f"[Translated]: {title}")
        
        # [Comment translated to English]
        if key_points:
            summary_parts.append("[Translated]:")
            for i, point in enumerate(key_points[:3], 1):
                summary_parts.append(f"{i}. {point}")
        
        # [Comment translated to English]
        elif main_content:
            content_preview = main_content.replace('\n', ' ').strip()
            if len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."
            summary_parts.append(f"[Translated]: {content_preview}")
        
        # [Comment translated to English]
        final_summary = "\n".join(summary_parts) if summary_parts else original_snippet
        
        return {
            "summary": final_summary or "[Translated]...",
            "key_points": key_points[:3],
            "source": "fallback",
            "relevance_score": "[Translated]",
            "content_quality": "[Translated]",
            "main_topics": [title] if title else []
        }
    def perform_edu_parse(self, url: str, timeout: int = 60) -> Dict[str, Any]:
        """UseEDU API[Translated]URLContent（[Translated]，[Translated]）"""
        if not self.edu_client:
            return {"success": False, "error": "EDU API client not initialized"}
        
        try:
            # [Comment translated to English]
            resp = self.edu_client.create_task(
                entry_type=self.config.DEFAULT_ENTRY_TYPE,
                entry_url=url
            )
            
            if not resp:
                return {"success": False, "error": "Failed to create EDU parsing task"}
            
            task_id = resp.data.task_id
            print(f'📋 EDU task created successfully, task_id: {task_id}')
            
            # [Comment translated to English]
            start_time = time.time()
            while True:
                # [Comment translated to English]
                if time.time() - start_time > timeout:
                    return {
                        "success": False, 
                        "error": f"EDU parsing timeout after {timeout} seconds",
                        "task_id": task_id
                    }
                
                # [Comment translated to English]
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
                    print(f'✅ task_id: {task_id}, task completed successfully')
                    break
                else:
                    print(f'🔄 task_id: {task_id}, task is running, status: {status}')
                    time.sleep(1)
            
            # [Comment translated to English]
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
            
            # [Comment translated to English]
            content_elements = self._extract_edu_content_elements(parsed_content)
            
            return {
                "success": True, 
                "data": parsed_content, 
                "content_elements": content_elements,
                "task_id": task_id
            }
            
        except Exception as e:
            error_msg = f"EDU API[Translated]error: {e}"
            return {"success": False, "error": error_msg}
    def perform_enhanced_search(self, query: str, enable_edu: bool = False) -> Dict[str, Any]:
        """[Translated]（[Translated]），[Translated]EDUparse/parsing"""
        # [Comment translated to English]
        search_results = self.perform_search(query)
        
        if not enable_edu or not search_results.get("success"):
            return search_results
        
        # [Comment translated to English]
        try:
            web_results = search_results["data"]["data"]["web"]
            
            # [Comment translated to English]
            with ThreadPoolExecutor(max_workers=3) as executor:
                # [Comment translated to English]
                future_to_result = {}
                for i, result in enumerate(web_results[:3]):  # [Comment translated to English]
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
                            content_elements = edu_result.get("content_elements", {})
                            
                            # [Comment translated to English]
                            result["edu_title"] = content_elements.get("title", "")
                            result["edu_content"] = content_elements.get("main_content", "")
                            result["edu_key_points"] = content_elements.get("key_points", [])
                            result["edu_structure"] = content_elements.get("structure_info", {})
                            result["edu_markdown"] = content_elements.get("markdown", "")
                            
                            # [Comment translated to English]
                            original_snippet = result.get("snippet", "") or result.get("content", "")
                            smart_summary_result = self._generate_smart_summary(
                                content_elements, 
                                original_snippet, 
                                query, 
                                result.get("url", "")
                            )
                            
                            result["summary"] = smart_summary_result.get("summary", "")
                            result["llm_summary"] = smart_summary_result.get("summary", "")
                            
                            # [Comment translated to English]
                            if not result.get("content") or len(result.get("content", "")) < 100:
                                if content_elements.get("main_content"):
                                    result["content"] = content_elements["main_content"][:500] + "..."
                            
                            result["edu_task_id"] = edu_result.get("task_id")
                            result["summary_source"] = smart_summary_result.get("source", "fallback")
                            print(f"✅ EDU+LLM processing completed for: {result.get('url', '')}")
                            
                        else:
                            result["edu_parse_error"] = edu_result.get("error")
                            print(f"❌ EDU parsing failed for: {result.get('url', '')}, error: {edu_result.get('error')}")
                            
                    except Exception as e:
                        result = web_results[result_index]
                        result["edu_parse_error"] = f"EDU parsing exception: {str(e)}"
                        print(f"❌ EDU parsing exception for: {result.get('url', '')}, error: {str(e)}")
            
            # [Comment translated to English]
            search_results["data"]["data"]["web"] = web_results
            search_results["edu_enhanced"] = True
            search_results["llm_enhanced"] = self.llm_client is not None
            
        except Exception as e:
            print(f"❌ EDU enhancement failed: {e}")
            search_results["edu_enhancement_error"] = str(e)
        
        return search_results
    
    def format_enhanced_search_results(self, search_results: Dict[str, Any], enable_edu: bool = False) -> str:
        """[Translated]Search results"""
        if not search_results.get("success"):
            return f"Search failed: {search_results.get('error', 'Unknown error')}"
        
        data = search_results.get("data", {})
        if not data.get("data") or not data["data"].get("web"):
            return "[Translated]FoundRelevantSearch results"
        
        formatted_results = []
        for i, item in enumerate(data["data"]["web"][:5], 1):
            title = item.get("title", "[Translated]")
            url = item.get("url", "")
            
            # [Comment translated to English]
            if enable_edu and item.get("llm_summary"):
                # [Comment translated to English]
                display_title = item.get("edu_title") or title
                
                # [Comment translated to English]
                description = item.get("llm_summary", "")
                
                # [Comment translated to English]
                key_points = item.get("edu_key_points", [])
                if key_points:
                    key_points_text = "\n".join([f"  • {point}" for point in key_points[:3]])
                    description += f"\n\n[Translated]:\n{key_points_text}"
                
                # [Comment translated to English]
                summary_source = item.get("summary_source", "unknown")
                status_text = "🤖 AI[Translated]" if summary_source == "llm" else "📝 [Translated]"
                
            elif enable_edu and item.get("edu_content"):
                # [Comment translated to English]
                display_title = item.get("edu_title") or title
                description = item.get("summary", "")
                status_text = "📋 EDU[Translated]"
                
            else:
                # [Comment translated to English]
                display_title = title
                content = item.get("content", "")
                snippet = item.get("snippet", "")
                description = item.get("summary") or snippet or content[:300]
                status_text = "📄 [Translated]Search results"
            
            result_text = f"""
[Translated] {i}: {status_text}
title: {display_title}
[Translated]: {url}
Contentsummary: {description}"""
            
            # [Comment translated to English]
            if enable_edu:
                if item.get("edu_parse_error"):
                    result_text += f"\n⚠️ [Translated]: failed ({item['edu_parse_error'][:50]}...)"
                elif item.get("edu_content"):
                    result_text += f"\n✅ [Translated]: success[Translated]"
            
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
            
            # [Comment translated to English]
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
            error_msg = f"[Translated]error: {e}"
            return {"success": False, "error": error_msg}
    
    def format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results[Translated]LLM[Translated]'sText"""
        if not search_results.get("success"):
            return f"Search failed: {search_results.get('error', 'Unknown error')}"
        
        data = search_results.get("data", {})
        if not data.get("data") or not data["data"].get("web"):
            return "[Translated]FoundRelevantSearch results"
        
        formatted_results = []
        for i, item in enumerate(data["data"]["web"][:5], 1):  # [Comment translated to English]
            title = item.get("title", "[Translated]")
            url = item.get("url", "")
            content = item.get("content", "")
            snippet = item.get("snippet", "")
            
            # [Comment translated to English]
            description = snippet or content[:300]
            
            formatted_results.append(f"""
[Translated] {i}:
title: {title}
[Translated]: {url}
Contentsummary: {description}
""")
        
        return "\n".join(formatted_results)