import os
import time
import random
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 120


class LLMCall:
    def __init__(self, API_key: Optional[str], model_name: str, max_retries: int = 3,
                 backoff_base: float = 0.8, timeout: int = DEFAULT_REQUEST_TIMEOUT):
        if not model_name:
            raise ValueError("model_name is required")

        self.API_key = API_key or os.environ.get("OPENAI_API_KEY")
        if not self.API_key:
            raise ValueError("API key not provided. Pass API_key or set OPENAI_API_KEY environment variable.")

        self.model_name = model_name.strip()
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.default_timeout = int(timeout)

        api_base = self._resolve_base_url(self.model_name)

        client_kwargs = {"api_key": self.API_key, "timeout": float(self.default_timeout)}
        if api_base:
            client_kwargs["base_url"] = api_base
            logger.info("Using custom base_url=%s for model %s", api_base, self.model_name)

        self.client = OpenAI(**client_kwargs)

        logger.info("Resolved model '%s' (timeout=%ds)", self.model_name, self.default_timeout)

    @staticmethod
    def _resolve_base_url(name: str) -> Optional[str]:
        if name == 'qwen':
            return ""
        elif name.startswith("qwen"):
            return os.environ.get("OPENAI_BASE_URL")
        elif name.startswith("llama-3"):
            return os.environ.get("OPENAI_BASE_URL")
        elif name.startswith("deepseek"):
            return os.environ.get("OPENAI_BASE_URL")
        elif name.startswith("o3-mini"):
            return os.environ.get("OPENAI_BASE_URL")
        elif name.startswith("gemini"):
            return os.environ.get("OPENAI_BASE_URL")
        elif name.startswith("gpt"):
            return os.environ.get("OPENAI_BASE_URL")
        return None

    def _parse_response_content(self, response: Any) -> tuple:
        
        token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            usage = getattr(response, "usage", None) or response.get("usage", None)
            if usage:
                token_usage['prompt_tokens'] = getattr(usage, "prompt_tokens", 0) or usage.get("prompt_tokens", 0)
                token_usage['completion_tokens'] = getattr(usage, "completion_tokens", 0) or usage.get("completion_tokens", 0)
                token_usage['total_tokens'] = getattr(usage, "total_tokens", 0) or usage.get("total_tokens", 0)
        except Exception:
            pass
        
        try:
            return response.choices[0].message.content, token_usage
        except Exception:
            pass
        try:
            return response.choices[0].message["content"], token_usage
        except Exception:
            pass
        try:
            return response.choices[0].text, token_usage
        except Exception:
            pass
        try:
            choices = getattr(response, "choices", None) or response.get("choices", None)
            if choices and len(choices) > 0:
                c0 = choices[0]
                if isinstance(c0, dict):
                    if "message" in c0:
                        msg = c0["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            return msg["content"], token_usage
                    if "text" in c0:
                        return c0["text"], token_usage
                else:
                    return str(c0), token_usage
        except Exception:
            pass

        s = str(response)
        return s if len(s) < 2000 else s[:2000] + "...", token_usage

    def call(self, messages: List[Dict[str, str]], seed: int = 0, timeout: Optional[int] = None,
             max_tokens: Optional[int] = None, return_tokens: bool = False):

        effective_timeout = timeout if timeout is not None else self.default_timeout
        attempt = 0
        last_exc = None

        while attempt <= self.max_retries:
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "timeout": float(effective_timeout),
                    "temperature": 0,
                }
                if seed is not None:
                    kwargs["seed"] = seed
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                logger.debug("Calling API: attempt=%d model=%s timeout=%ds",
                             attempt + 1, self.model_name, effective_timeout)

                if self.model_name.startswith("qwen"):
                    kwargs["extra_body"] = {"enable_thinking": False}

                response = self.client.chat.completions.create(**kwargs)
                content, token_usage = self._parse_response_content(response)

                if return_tokens:
                    return {
                        'content': content,
                        'token_usage': token_usage
                    }
                return content
            except Exception as e:
                last_exc = e
                attempt += 1
                if attempt > self.max_retries:
                    break
                base_wait = (2 ** attempt) * self.backoff_base
                jitter = random.uniform(-0.3, 0.3) * base_wait
                wait = max(0.5, base_wait + jitter)
                logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.2fs",
                               attempt, self.max_retries, e, wait)
                time.sleep(wait)

        logger.error("LLM call failed after %d attempts. Last error: %s", self.max_retries, last_exc)
        error_msg = f"__LLM_CALL_FAILED__ {last_exc}"

        if return_tokens:
            return {
                'content': error_msg,
                'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
        return error_msg
