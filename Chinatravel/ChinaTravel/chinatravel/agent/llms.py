from abc import ABC, abstractmethod
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from dotenv import load_dotenv

# from modelscope import AutoModelForCausalLM, AutoTokenizer
try:
    import tiktoken
except Exception:
    tiktoken = None

"""LLM backends and API clients.
This file supports both API-based inference (OpenAI/DeepSeek/GLM4-Plus)
and local inference (Qwen/Mistral/Llama via vLLM). For environments that
do not need local inference, the vLLM import is optional and guarded.
"""

# vLLM is optional for API-only runs
try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
except Exception:
    LLM = None  # type: ignore
    _VLLM_AVAILABLE = False
    # Provide a lightweight placeholder so type checking doesn't fail
    class SamplingParams:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
import re
import sys
import os

project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

load_dotenv()


def _optional_transformers():
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception:
        return None, None
    return AutoConfig, AutoTokenizer


def _require_transformers():
    AutoConfig, AutoTokenizer = _optional_transformers()
    if AutoConfig is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required for local tokenizer/model initialization. "
            "API-backed models can run without it, but local ChinaTravel LLM backends cannot."
        )
    return AutoConfig, AutoTokenizer


def _safe_tiktoken_encoding():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _repair_json_if_available(text, *, ensure_ascii=False):
    try:
        from json_repair import repair_json
    except Exception:
        return text
    return repair_json(text, ensure_ascii=ensure_ascii)


def _env_first(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return value
    return default

def chat_template(messages):
    """
    将 messages 列表转成符合 Chat 模板格式的字符串
    用于 tiktoken.encode 计算 token 数。
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|{role}|>\n{content}\n"
    formatted += "<|assistant|>\n"  # 留空表示用户希望 assistant 继续回复
    return formatted

def merge_repeated_role(messages):
    ptr = len(messages) - 1
    last_role = ""
    while ptr >= 0:
        cur_role = messages[ptr]["role"]
        if cur_role == last_role:
            messages[ptr]["content"] += "\n" + messages[ptr + 1]["content"]
            del messages[ptr + 1]
        last_role = cur_role
        ptr -= 1
    return messages


class AbstractLLM(ABC):
    class ModeError(Exception):
        pass

    def __init__(self):
        self.input_token_count = 0
        self.output_token_count = 0
        self.input_token_maxx = 0
        pass

    def __call__(self, messages, one_line=True, json_mode=False):
        if one_line and json_mode:
            raise self.ModeError(
                "one_line and json_mode cannot be True at the same time"
            )
        return self._get_response(messages, one_line, json_mode)

    @abstractmethod
    def _get_response(self, messages, one_line, json_mode):
        pass


class Deepseek(AbstractLLM):
    def __init__(self):
        super().__init__()
        # Allow overriding base_url via environment for OpenAI-compatible gateways
        base_url = _env_first(
            "OPENAI_BASE_URL",
            "DMXAPI_BASE_URL",
            "SILICONCLOUD_BASE_URL",
            default="https://api.deepseek.com",
        )
        api_key = _env_first(
            "OPENAI_API_KEY",
            "DMXAPI_API_KEY",
            "SILICONCLOUD_API_KEY",
            "API_KEY",
        )
        self.llm = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", "deepseek_v3_tokenizer"
        )
        self.name = "DeepSeek-V3"
        self.tokenizer = None
        self.fallback_tokenizer = _safe_tiktoken_encoding()
        _, AutoTokenizer = _optional_transformers()
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            except Exception:
                self.tokenizer = None

    def _send_request(self, messages, kwargs):
        if self.tokenizer is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_tokens = self.tokenizer(text)["input_ids"]
        else:
            text = chat_template(messages)
            input_tokens = (
                self.fallback_tokenizer.encode(text)
                if self.fallback_tokenizer is not None
                else []
            )

        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
        
        res_str = (
            self.llm.chat.completions.create(messages=messages, **kwargs)
            .choices[0]
            .message.content
        )
        if self.tokenizer is not None:
            output_tokens = self.tokenizer(res_str)["input_ids"]
        else:
            output_tokens = (
                self.fallback_tokenizer.encode(res_str)
                if self.fallback_tokenizer is not None
                else []
            )
        self.output_token_count += len(output_tokens)
        
        res_str = res_str.strip()
        return res_str

    def _get_response(self, messages, one_line, json_mode):
        # Allow overriding model via environment for OpenAI-compatible gateways
        model_name = _env_first(
            "MODEL_NAME",
            "DMXAPI_MODEL",
            "SILICONCLOUD_MODEL",
            default="deepseek-chat",
        )
        kwargs = {
            "model": model_name,
            "max_tokens": 4096,
            "temperature": 0,
            "top_p": 0.00000001,
        }
        if one_line:
            kwargs["stop"] = ["\n"]
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
        except OpenAIRateLimitError as e:
            # Bubble up rate limit error for external orchestrator handling
            raise
        except Exception as e:
            print("Deepseek request error:", e)
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class GLM4Plus(AbstractLLM):
    def __init__(self):
        super().__init__()
        api_key = _env_first(
            "OPENAI_API_KEY",
            "DMXAPI_API_KEY",
            "SILICONCLOUD_API_KEY",
            "API_KEY",
        )
        self.llm = OpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=api_key,
        )
        self.name = "GLM4Plus"

    def _send_request(self, messages, kwargs):
        res_str = (
            self.llm.chat.completions.create(messages=messages, **kwargs)
            .choices[0]
            .message.content
        )
        res_str = res_str.strip()
        return res_str

    def _get_response(self, messages, one_line, json_mode):
        kwargs = {
            "model": "glm-4-plus",
            "max_tokens": 4095,
            "temperature": 0,
            "top_p": 0.01,
        }
        if one_line:
            kwargs["stop"] = ["<STOP>"]
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
        except Exception as e:
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class GPT4o(AbstractLLM):
    def __init__(self):
        super().__init__()
        base_url = _env_first("OPENAI_BASE_URL", "DMXAPI_BASE_URL", "SILICONCLOUD_BASE_URL")
        api_key = _env_first(
            "OPENAI_API_KEY",
            "DMXAPI_API_KEY",
            "SILICONCLOUD_API_KEY",
            "API_KEY",
        )
        self.llm = OpenAI(api_key=api_key, base_url=base_url)
        self.name = "GPT4o"
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o") if tiktoken is not None else None


    def _send_request(self, messages, kwargs):

        # print(messages)
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(chat_template(messages))
            self.input_token_count += len(tokens)
            self.input_token_maxx = max(self.input_token_maxx, len(tokens))

        # print(tokens)
        # print(self.input_token_count)
        # exit(0)

        res_str = (
            self.llm.chat.completions.create(messages=messages, **kwargs)
            .choices[0]
            .message.content
        )
        
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(res_str)
            self.output_token_count += len(tokens)

        res_str = res_str.strip()
        return res_str

    def _get_response(self, messages, one_line, json_mode):
        kwargs = {
            "model": "chatgpt-4o-latest",
            "max_tokens": 4095,
            "temperature": 0,
            "top_p": 0.01,
        }
        if one_line:
            kwargs["stop"] = ["\n"]
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
        except Exception as e:
            print(e)
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class GPT4oMini(GPT4o):
    """OpenAI-compatible GPT-4o-mini backend for LLMNeSy replanning."""

    def __init__(self):
        super().__init__()
        self.name = "GPT4oMini"
        try:
            self.tokenizer = (
                tiktoken.encoding_for_model("gpt-4o-mini")
                if tiktoken is not None
                else None
            )
        except Exception:
            self.tokenizer = _safe_tiktoken_encoding()

    def _get_response(self, messages, one_line, json_mode):
        model_name = _env_first(
            "GPT4O_MINI_MODEL",
            "DMXAPI_GPT4O_MINI_MODEL",
            "SILICONCLOUD_GPT4O_MINI_MODEL",
            default="gpt-4o-mini",
        )
        kwargs = {
            "model": model_name,
            "max_tokens": 4095,
            "temperature": 0,
            "top_p": 0.01,
        }
        if one_line:
            kwargs["stop"] = ["\n"]
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
        except Exception as e:
            print(e)
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class QwenAPI(AbstractLLM):
    """远程 Qwen（如 qwen3-8b）通过 OpenAI 兼容网关调用。
    基础地址优先级：QWEN_API_BASE_URL > DMXAPI_API_BASE_URL > OPENAI_BASE_URL > 默认 https://www.dmxapi.cn/v1
    模型名优先级：MODEL_NAME 环境变量 > 默认 qwen3-8b
    """
    def __init__(self):
        super().__init__()
        base_url = (
            os.environ.get("QWEN_API_BASE_URL")
            or os.environ.get("DMXAPI_API_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://www.dmxapi.cn/v1"
        )
        self.llm = OpenAI(base_url=base_url)
        self.name = os.environ.get("MODEL_NAME", "qwen3-8b")
        # 使用一个通用的 tiktoken 编码做粗略 token 统计（无严格精度需求）
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
        except Exception:
            self.tokenizer = None

    def _send_request(self, messages, kwargs):
        # 基于通用 chat 模板统计输入 token
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(chat_template(messages))
            self.input_token_count += len(tokens)
            self.input_token_maxx = max(self.input_token_maxx, len(tokens))
        res = self.llm.chat.completions.create(messages=messages, **kwargs)
        content = res.choices[0].message.content.strip()
        if self.tokenizer is not None:
            out_tokens = self.tokenizer.encode(content)
            self.output_token_count += len(out_tokens)
        return content

    def _get_response(self, messages, one_line, json_mode):
        kwargs = {
            "model": self.name,
            "max_tokens": 4096,
            "temperature": 0,
            "top_p": 0.01,
        }
        if one_line:
            kwargs["stop"] = ["\n"]
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
        except OpenAIRateLimitError:
            raise
        except Exception as e:
            print("QwenAPI request error:", e)
            res_str = '{"error": "Request failed, please try again."}'
        return res_str


class Qwen(AbstractLLM):
    def __init__(self, model_name, max_model_len=None):
        super().__init__()
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required for local Qwen inference. "
                "Please install local LLM dependencies as in requirements.txt (vllm, torch, etc.)."
            )
        AutoConfig, AutoTokenizer = _require_transformers()
        self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", model_name
        )
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1" 
        if "Qwen3" in model_name:    
            self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096)
            
        else:
            self.sampling_params = SamplingParams(temperature=0, top_p=0.001, max_tokens=4096)

        if max_model_len is not None and max_model_len > 32768:
            config = AutoConfig.from_pretrained(self.path)
            config.rope_scaling = {
                    "type": "yarn", 
                    "factor": max_model_len//32768, # 2.0,  # 原长 32,768 → 扩展到 32,768 * 2 = 65536
                    "original_max_position_embeddings": 32768
                }
            config.save_pretrained(self.path)
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        else:
            config = AutoConfig.from_pretrained(self.path)
            if "rope_scaling" in config.to_dict():
                del config.rope_scaling
            config.save_pretrained(self.path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

        if max_model_len is None:
            max_model_len = 32768
            
        self.llm = LLM(
            model=self.path,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,  # 强制上下文长度为 65536
            # max_num_seqs = 1,           # Limit batch size
            # tensor_parallel_size=2,     # GPUs=2
            enable_prefix_caching=(max_model_len>=32768),  # 可选：启用前缀缓存优化长文本
        )

        self.name = model_name
        self.max_model_len = max_model_len

        

    def _get_response(self, messages, one_line, json_mode):
        # print(messages)
        
        

        if "Qwen3" in self.name:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switch between thinking and non-thinking modes. Default is True.
            )

            input_tokens = self.tokenizer(text)["input_ids"]
            self.input_token_count += len(input_tokens)       
            self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
            
            if len(input_tokens) >= self.max_model_len:
                return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})
            # conduct text completion
            outputs = self.llm.generate([text], self.sampling_params)


            generated_text = outputs[0].outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            # print(generated_text)

            output_token_ids = outputs[0].outputs[0].token_ids
            self.output_token_count += len(output_token_ids)

            try:
                m = re.match(r"<think>\n(.+)</think>\n\n", generated_text, flags=re.DOTALL)
                content = generated_text[len(m.group(0)):]
                thinking_content = m.group(1).strip()

            except Exception as e:
                thinking_content = ""
                content = generated_text.strip()
            
            # print("think content: ", thinking_content)
            # print("content: ", content)
            res_str = content
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            input_tokens = self.tokenizer(text)["input_ids"]
            self.input_token_count += len(input_tokens)        
            self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
            
            if len(input_tokens) >= self.max_model_len:
                return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

            outputs = self.llm.generate([text], self.sampling_params)
            res_str = outputs[0].outputs[0].text

            output_token_ids = outputs[0].outputs[0].token_ids
            self.output_token_count += len(output_token_ids)
        try:
            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
            elif one_line:
                res_str = res_str.split("\n")[0]
        except Exception as e:
            res_str = '{"error": "Request with specific format failed, please try again."}'
        # print("---qwen_output---")
        # print(res_str)
        # print("---qwen_output_end---")
        return res_str


class Mistral(AbstractLLM):
    def __init__(self, max_model_len=None):
        super().__init__()
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required for local Mistral inference. "
                "Please install local LLM dependencies as in requirements.txt (vllm, torch, etc.)."
            )
        AutoConfig, AutoTokenizer = _require_transformers()
        self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", "Mistral-7B-Instruct-v0.3",
        )
        self.sampling_params = SamplingParams(
            temperature=0, top_p=0.001, max_tokens=4096
        )

        if max_model_len is not None and max_model_len > 32768:
            config = AutoConfig.from_pretrained(self.path)
            config.rope_scaling = {
                "type": "yarn", 
                "factor": max_model_len // 32768,
                "original_max_position_embeddings": 32768
            }
            config.save_pretrained(self.path)
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        else:
            config = AutoConfig.from_pretrained(self.path)
            if "rope_scaling" in config.to_dict():
                del config.rope_scaling
            config.save_pretrained(self.path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

        if max_model_len is None:
            max_model_len = 32768

        self.llm = LLM(
            model=self.path,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
            # max_num_seqs = 1,           # Limit batch size
            # tensor_parallel_size=2,     # GPUs=2
            enable_prefix_caching=(max_model_len>=32768),  # 可选：启用前缀缓存优化长文本
        )
        self.name = "Mistral-7B-Instruct-v0.3"
        self.max_model_len = max_model_len

    def _get_response(self, messages, one_line, json_mode):
        messages = merge_repeated_role(messages)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        input_tokens = self.tokenizer(text)["input_ids"]
        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

        if len(input_tokens) >= self.max_model_len:
            return str({"error": f"Input prompt is longer than {self.max_model_len} tokens."})

        # try:
        outputs = self.llm.generate([text], self.sampling_params)
        res_str = outputs[0].outputs[0].text
        
        output_token_ids = outputs[0].outputs[0].token_ids
        self.output_token_count += len(output_token_ids)
        
        if json_mode:
            res_str = _repair_json_if_available(res_str, ensure_ascii=False)
        elif one_line:
            res_str = res_str.split("\n")[0]
        # except Exception as e:
        #     print("error: ", e)
        #     res_str = '{"error": "Request failed, please try again."}'
        return res_str


class Llama(AbstractLLM):
    def __init__(self, model_name):
        super().__init__()
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required for local Llama inference. "
                "Please install local LLM dependencies as in requirements.txt (vllm, torch, etc.)."
            )
        _, AutoTokenizer = _require_transformers()


        Llama_supported = ["Llama3-3B", "Llama3-8B"]
        if model_name not in Llama_supported:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models: {Llama_supported}")
        
        if model_name == "Llama3-3B":
            self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", "Llama-3.2-3B-Instruct"
            )
        elif model_name == "Llama3-8B":
            self.path = os.path.join(
            project_root_path, "chinatravel", "local_llm", "Meta-Llama-3.1-8B-Instruct"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.sampling_params = SamplingParams(
            temperature=0, top_p=0.001, max_tokens=4096
        )
        self.llm = LLM(model=self.path) #, local_files_only=True)
        self.name = model_name

    def _get_response(self, messages, one_line, json_mode):
        # print(messages)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
        input_tokens = self.tokenizer(text)["input_ids"]
        self.input_token_count += len(input_tokens)
        self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))
        
        if len(input_tokens) >= 131072:
            return '{"error": "Input prompt is longer than 131072 tokens."}'
        
        
        try:
            outputs = self.llm.generate([text], self.sampling_params)
            res_str = outputs[0].outputs[0].text
            
            output_token_ids = outputs[0].outputs[0].token_ids
            self.output_token_count += len(output_token_ids)

            if json_mode:
                res_str = _repair_json_if_available(res_str, ensure_ascii=False)
            elif one_line:
                res_str = res_str.split("\n")[0]
        except Exception as e:
            res_str = '{"error": "Request failed, please try again."}'
        # print("---mistral_output---")
        # print(res_str)
        # print("---mistral_output_end---")
        print(res_str)
        return res_str

class EmptyLLM(AbstractLLM):
    def __init__(self):
        super().__init__()
        self.name = "EmptyLLM"

    def _get_response(self, messages, one_line, json_mode):
        return "Empty LLM response"

if __name__ == "__main__":
    # model = Mistral()
    model = GPT4o()
    print(model([{"role": "user", "content": "hello!"}], one_line=False))
