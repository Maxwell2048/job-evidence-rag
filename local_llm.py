"""Local generation-model adaptation layer.

Business modules call one uniform interface and never depend on a specific
inference service protocol:

    llm = build_llm(config)
    data = llm.generate_json(task, system_prompt, payload, schema,
                             timeout_seconds, validate)

The adapter owns the wire protocol, output parsing, timeouts and retries, and
is the single place that counts attempts: program-side checks are passed in as
`validate` so they share the one three-attempt budget per request.
"""
import json
import re
import socket
import sys
import urllib.error
import urllib.request

ADAPTERS = {}

# Prompt length is estimated in CHARACTERS; the service counts TOKENS. One CJK
# character is roughly one token, so 1 char/token is the conservative default:
# it never under-counts a Chinese JD. Raise it only for ASCII-only material.
DEFAULT_CHARS_PER_TOKEN = 1


def prompt_char_budget(config):
    """Characters left for the prompt after reserving the output window.

    context_budget and max_output_tokens are token counts (what the local
    service reports); the difference is converted to a character budget so the
    planners in jd_parser / evidence_matcher can slice text without tokenizing.
    """
    tokens = config["context_budget"] - config["max_output_tokens"]
    return max(1, int(tokens * config.get("chars_per_token",
                                          DEFAULT_CHARS_PER_TOKEN)))


class LLMError(Exception):
    """Base error; error_type is recorded in run metadata."""
    error_type = "llm_error"

    def __init__(self, message, error_type=None):
        super().__init__(message)
        if error_type is not None:
            self.error_type = error_type


class ConfigError(LLMError):
    error_type = "config"


class ServiceError(LLMError):
    error_type = "service_unavailable"


class RequestTimeoutError(LLMError):
    error_type = "timeout"


class StructureError(LLMError):
    error_type = "structure"


class ContextBudgetError(LLMError):
    error_type = "context_budget"


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_ok(value, expected):
    if isinstance(expected, list):
        return any(_type_ok(value, item) for item in expected)
    return {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "null": lambda v: v is None,
    }.get(expected, lambda v: False)(value)


def validate_against_schema(data, schema, path="$"):
    """Minimal JSON-schema validation covering the keywords this project uses."""
    errors = []
    expected = schema.get("type")
    if expected is not None and not _type_ok(data, expected):
        errors.append(f"{path}：期望类型 {expected}，实际为 {_type_name(data)}")
        return errors
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}：值 {data!r} 不在枚举 {schema['enum']} 中")
    if isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(f"{path}：长度 {len(data)} 小于 minLength {schema['minLength']}")
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(f"{path}：长度 {len(data)} 大于 maxLength {schema['maxLength']}")
    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}：缺少必需字段 '{key}'")
        properties = schema.get("properties", {})
        for key, sub in properties.items():
            if key in data:
                errors.extend(validate_against_schema(data[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in properties:
                    errors.append(f"{path}：出现未声明字段 '{key}'")
    if isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"{path}：元素数 {len(data)} 少于 minItems {schema['minItems']}")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path}：元素数 {len(data)} 多于 maxItems {schema['maxItems']}")
        items = schema.get("items")
        if items:
            for index, item in enumerate(data):
                errors.extend(validate_against_schema(item, items, f"{path}[{index}]"))
    return errors


def extract_json_object(text):
    """Parse a JSON object from model output, tolerating fences and prose."""
    if not isinstance(text, str):
        raise StructureError(f"模型输出不是文本（{_type_name(text)}）")
    stripped = text.strip()
    for candidate in (stripped,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise StructureError("无法从模型输出中解析出 JSON 对象")


def _require(config, key, kind, message):
    value = config.get(key)
    if kind is str:
        ok = isinstance(value, str) and value.strip() != ""
    else:
        ok = isinstance(value, kind) and not isinstance(value, bool)
    if not ok:
        raise ConfigError(message)
    return value


def load_config(path):
    """Read and validate the local model config file."""
    try:
        with open(path, encoding="utf-8-sig") as handle:
            config = json.loads(handle.read())
    except OSError as exc:
        raise ConfigError(f"配置文件无法读取：{path}（{exc}）")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件不是有效 JSON：{path}（{exc}）")
    if not isinstance(config, dict):
        raise ConfigError("配置文件顶层必须是 JSON 对象")
    adapter_name = _require(config, "adapter", str, "配置缺少非空字符串字段 adapter（适配器类型）")
    if adapter_name not in ADAPTERS:
        raise ConfigError(f"未知适配器类型 {adapter_name!r}；可用：{', '.join(sorted(ADAPTERS))}")
    _require(config, "model", str, "配置缺少非空字符串字段 model（生成模型名称）")
    _require(config, "context_budget", int,
             "配置缺少正整数 context_budget（模型上下文窗口，单位 token）")
    if config["context_budget"] < 1:
        raise ConfigError("context_budget 必须大于 0")
    _require(config, "timeout_seconds", (int, float), "配置缺少正数 timeout_seconds（请求超时秒数）")
    if config["timeout_seconds"] <= 0:
        raise ConfigError("timeout_seconds 必须大于 0")
    _require(config, "max_output_tokens", int,
             "配置缺少正整数 max_output_tokens（为输出预留的 token 数）")
    if config["max_output_tokens"] < 1:
        raise ConfigError("max_output_tokens 必须大于 0")
    if config["max_output_tokens"] >= config["context_budget"]:
        raise ConfigError(
            f"max_output_tokens（{config['max_output_tokens']}）必须小于 "
            f"context_budget（{config['context_budget']}）；"
            "上下文窗口要同时容纳提示词和输出")
    config.setdefault("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
    chars_per_token = config["chars_per_token"]
    if (not isinstance(chars_per_token, (int, float))
            or isinstance(chars_per_token, bool) or chars_per_token <= 0):
        raise ConfigError("chars_per_token 必须是大于 0 的数字")
    config.setdefault("temperature", 0)
    temperature = config["temperature"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ConfigError("temperature 必须是数字")
    config.setdefault("api_key", "local")
    config.setdefault("response_format", "json_object")
    if config["response_format"] not in {"json_object", "json_schema", "none"}:
        raise ConfigError("response_format 必须是 json_object、json_schema 或 none")
    config.setdefault("extra_body", {})
    if not isinstance(config["extra_body"], dict):
        raise ConfigError("extra_body 必须是 JSON 对象（原样并入请求体的额外字段，"
                          "例如 {\"chat_template_kwargs\": {\"enable_thinking\": false}}）")
    return config


class BaseAdapter:
    """Holds validated config and implements retry/correction around _call_service."""
    name = None
    max_attempts = 3

    def __init__(self, config):
        self.config = config
        self.model = config["model"]
        self.context_budget = config["context_budget"]
        self.timeout_seconds = config["timeout_seconds"]
        self.max_output_tokens = config["max_output_tokens"]
        self.temperature = config.get("temperature", 0)
        self.chars_per_token = config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        # Planners slice text against this, never against the raw window.
        self.prompt_char_budget = prompt_char_budget(config)

    def describe_parameters(self):
        """Generation settings recorded in run metadata so a run is reproducible."""
        return {
            "adapter": self.name,
            "temperature": self.temperature,
            "context_budget_tokens": self.context_budget,
            "max_output_tokens": self.max_output_tokens,
            "chars_per_token": self.chars_per_token,
            "prompt_char_budget": self.prompt_char_budget,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "response_format": self.config.get("response_format", "json_object"),
            "extra_body": self.config.get("extra_body", {}),
        }

    @staticmethod
    def _build_user_content(task, payload, schema):
        parts = [f"任务：{task}"]
        if schema is not None:
            parts.append("输出必须是符合以下 JSON Schema 的 JSON 对象（字段名、类型与枚举必须一致）：\n"
                         + json.dumps(schema, ensure_ascii=False))
        parts.append("任务数据（JSON）：\n" + json.dumps(payload, ensure_ascii=False))
        return "\n\n".join(parts)

    def estimate_context(self, task, system_prompt, payload, schema):
        """Conservative character estimate of the prompt this would send."""
        return len(system_prompt) + len(self._build_user_content(task, payload, schema))

    def _check_context(self, messages):
        total = sum(len(message["content"]) for message in messages)
        if total > self.prompt_char_budget:
            raise ContextBudgetError(
                f"提示词长度 {total} 字符超出提示词预算 {self.prompt_char_budget} 字符"
                f"（上下文窗口 {self.context_budget} token 减去为输出预留的 "
                f"{self.max_output_tokens} token，按 {self.chars_per_token} 字符/token 换算）；"
                "请拆批输入")

    def describe_model(self):
        """Best-effort model identity for run metadata; never raises."""
        return {"name": self.model, "version": None}

    def _correction(self, messages, raw, error):
        return messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content":
                f"上一次输出未通过程序校验，错误详情：{error}\n"
                "请根据上述 schema 与任务数据修正，只返回修正后的 JSON。"},
        ]

    def generate_json(self, task, system_prompt, payload, schema=None,
                      timeout_seconds=None, validate=None):
        """Return a dict that parsed and validated against schema.

        Up to three attempts TOTAL for this request: the first plus two
        retries. Structural errors are retried with the explicit validation
        message; service/timeout errors retry the same request. Raises
        LLMError subclasses with error_type.

        `validate(data) -> [error strings]` adds program-side checks (quote
        verification, ID ownership) to the same attempt budget, so a caller
        never gets a second full set of retries by calling this twice.
        """
        if not isinstance(task, str) or not task.strip():
            raise ConfigError("task 必须是非空字符串")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ConfigError("system_prompt 必须是非空字符串")
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ConfigError("timeout_seconds 必须大于 0")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_user_content(task, payload, schema)},
        ]
        last = None
        raw = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._check_context(messages)
                raw = self._call_service(messages, timeout, schema)
                data = extract_json_object(raw)
            except (ServiceError, RequestTimeoutError) as exc:
                last = exc
                if attempt < self.max_attempts:
                    print(f"[local-llm] {task} 第 {attempt} 次尝试失败（{exc.error_type}），重试",
                          file=sys.stderr)
                continue
            except StructureError as exc:
                last = exc
                messages = self._correction(messages, raw, exc)
                if attempt < self.max_attempts:
                    print(f"[local-llm] {task} 第 {attempt} 次尝试输出结构错误，带校验信息重试",
                          file=sys.stderr)
                continue
            errors = validate_against_schema(data, schema) if schema is not None else []
            if not errors and validate is not None:
                errors = list(validate(data))
            if not errors:
                return data
            last = StructureError("；".join(errors[:8]))
            messages = self._correction(messages, raw, last)
            if attempt < self.max_attempts:
                print(f"[local-llm] {task} 第 {attempt} 次尝试未通过校验，带校验信息重试："
                      f"{errors[0][:200]}" + ("" if len(errors) == 1 else f"（另 {len(errors) - 1} 条）"),
                      file=sys.stderr)
        raise LLMError(f"请求 {task!r} 在 {self.max_attempts} 次尝试后仍失败：{last}",
                       error_type=last.error_type)

    def _call_service(self, messages, timeout, schema):
        raise NotImplementedError


def register_adapter(cls):
    ADAPTERS[cls.name] = cls
    return cls


@register_adapter
class OpenAIHttpAdapter(BaseAdapter):
    """Any OpenAI-compatible local endpoint (LM Studio, Ollama, Open WebUI, vLLM)."""
    name = "openai_http"

    def __init__(self, config):
        super().__init__(config)
        base_url = _require(config, "base_url", str,
                            "openai_http 适配器需要非空字符串 base_url（本机服务地址）")
        self.base_url = base_url.rstrip("/")
        self.api_key = config.get("api_key", "local")
        self.response_format = config.get("response_format", "json_object")
        # Service-specific fields merged into every request body, e.g.
        # llama.cpp/vLLM chat_template_kwargs to disable Qwen3 thinking so the
        # output budget is not consumed by reasoning tokens.
        self.extra_body = dict(config.get("extra_body") or {})

    def _endpoint(self):
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return self.base_url + "/chat/completions"
        return self.base_url + "/v1/chat/completions"

    def describe_model(self):
        info = {"name": self.model, "version": None}
        endpoint = self._endpoint()
        models_url = endpoint[: -len("/chat/completions")] + "/models"
        request = urllib.request.Request(
            models_url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            return info
        try:
            for entry in data.get("data", []):
                if entry.get("id") == self.model:
                    version = entry.get("version")
                    info["version"] = str(version) if version is not None else None
                    break
        except (AttributeError, TypeError):
            pass
        return info

    def _call_service(self, messages, timeout, schema):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": False, "schema": schema or {}},
            }
        elif self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise ServiceError(f"本机服务返回 HTTP {exc.code}：{exc.reason}",
                               error_type="http_error") from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            if isinstance(exc.reason, (TimeoutError, socket.timeout)) or "timed out" in reason:
                raise RequestTimeoutError(f"本机服务请求超时（{timeout} 秒）：{reason}",
                                          error_type="timeout") from exc
            raise ServiceError(f"本机服务不可用：{reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RequestTimeoutError(f"本机服务请求超时（{timeout} 秒）", error_type="timeout") from exc
        try:
            parsed = json.loads(raw_body)
            choice = parsed["choices"][0]
            content = choice["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise StructureError(f"本机服务响应结构不符合预期：{exc}") from exc
        if choice.get("finish_reason") == "length":
            # The JSON is cut off; say so instead of a vague parse failure, so the
            # correction message asks for a shorter, compact answer.
            raise StructureError(
                f"模型输出在 {self.max_output_tokens} 个输出 token 处被截断（finish_reason=length）；"
                "请输出紧凑 JSON（不缩进、不换行）并精简内容，或调大 max_output_tokens")
        return content


def build_llm(config):
    """Create the adapter named by config['adapter']."""
    if not isinstance(config, dict):
        raise ConfigError("config 必须是 JSON 对象解析后的 dict")
    adapter_name = config.get("adapter")
    if adapter_name not in ADAPTERS:
        raise ConfigError(f"缺少或未知适配器类型 {adapter_name!r}；可用：{', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[adapter_name](config)


def generate_json(config, task, system_prompt, payload, schema=None,
                  timeout_seconds=None, validate=None):
    """One-shot convenience wrapper around build_llm(...).generate_json(...)."""
    return build_llm(config).generate_json(task, system_prompt, payload, schema,
                                           timeout_seconds, validate)
