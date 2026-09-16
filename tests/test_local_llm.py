import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_llm import (ADAPTERS, ConfigError, ContextBudgetError, LLMError,
                       OpenAIHttpAdapter, RequestTimeoutError, ServiceError,
                       StructureError, build_llm, extract_json_object, load_config,
                       prompt_char_budget, validate_against_schema)


SCHEMA = {
    "type": "object",
    "required": ["a", "b"],
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "string", "enum": ["x", "y"]},
    },
    "additionalProperties": False,
}


def make_config(**overrides):
    config = {
        "adapter": "openai_http",
        "model": "test-model",
        "base_url": "http://127.0.0.1:9/v1",
        "context_budget": 100000,
        "timeout_seconds": 5,
        "max_output_tokens": 100,
        "temperature": 0,
    }
    config.update(overrides)
    return config


def make_adapter(**overrides):
    return OpenAIHttpAdapter(make_config(**overrides))


class FakeTransport:
    """Scripted _call_service replacement recording attempts."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, timeout, schema):
        self.calls.append({"messages": [dict(m) for m in messages], "timeout": timeout,
                           "schema": schema})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ExtractJsonTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_and_prose(self):
        self.assertEqual(extract_json_object('好的，结果如下：\n```json\n{"a": 1}\n```\n以上。'),
                         {"a": 1})
        self.assertEqual(extract_json_object('前缀 {"a": 1} 后缀'), {"a": 1})

    def test_invalid_raises_structure_error(self):
        with self.assertRaises(StructureError):
            extract_json_object("没有 JSON 的文本")
        with self.assertRaises(StructureError):
            extract_json_object(42)


class SchemaValidatorTests(unittest.TestCase):
    def test_valid_data_has_no_errors(self):
        self.assertEqual(validate_against_schema({"a": 1, "b": "x"}, SCHEMA), [])

    def test_missing_required_and_type_and_enum(self):
        errors = validate_against_schema({"a": "1", "b": "z"}, SCHEMA)
        self.assertTrue(any("a" in e and "integer" in e for e in errors))
        self.assertTrue(any("b" in e and "枚举" in e for e in errors))
        self.assertTrue(any("缺少必需字段" in e for e in validate_against_schema({}, SCHEMA)))

    def test_unknown_field_rejected(self):
        errors = validate_against_schema({"a": 1, "b": "x", "c": 3}, SCHEMA)
        self.assertTrue(any("未声明字段 'c'" in e for e in errors))

    def test_nested_and_array_paths(self):
        schema = {"type": "object", "required": ["items"],
                  "properties": {"items": {"type": "array", "minItems": 1,
                                           "items": {"type": "object",
                                                     "required": ["v"],
                                                     "properties": {"v": {"type": "boolean"}}}}}}
        self.assertEqual(validate_against_schema({"items": [{"v": True}]}, schema), [])
        errors = validate_against_schema({"items": [{"v": "yes"}]}, schema)
        self.assertEqual(len(errors), 1)
        self.assertIn("items[0].v", errors[0])


class ConfigTests(unittest.TestCase):
    def write_config(self, tmp, config):
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_valid_config_loads_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, {"adapter": "openai_http", "model": "m",
                                           "base_url": "http://x/v1",
                                           "context_budget": 100,
                                           "timeout_seconds": 5,
                                           "max_output_tokens": 10})
            config = load_config(path)
            self.assertEqual(config["temperature"], 0)
            self.assertEqual(config["response_format"], "json_object")
            self.assertEqual(config["extra_body"], {})

    def test_missing_or_bad_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            for patch in ({}, {"model": ""}, {"context_budget": "100"},
                          {"timeout_seconds": 0}, {"max_output_tokens": -1},
                          {"adapter": "nope"}, {"temperature": "hot"},
                          {"response_format": "yaml"}, {"extra_body": "x"}):
                config = make_config()
                config.pop("model", None)
                config.update(patch)
                path = self.write_config(tmp, config)
                with self.assertRaises(ConfigError, msg=str(patch)):
                    load_config(path)

    def test_missing_file_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_config(Path(tmp) / "absent.json")
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


class AdapterTests(unittest.TestCase):
    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)

    def test_success_on_first_attempt(self):
        adapter = make_adapter()
        transport = FakeTransport(['{"a": 1, "b": "x"}'])
        adapter._call_service = transport
        self.assertEqual(self.quiet(adapter.generate_json, "t", "sys", {"jd": "文本"}, SCHEMA),
                         {"a": 1, "b": "x"})
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0]["messages"][0]["role"], "system")
        self.assertIn("jd", transport.calls[0]["messages"][1]["content"])

    def test_structure_error_retried_with_validation_message(self):
        adapter = make_adapter()
        transport = FakeTransport(['{"a": "不是整数", "b": "x"}', '{"a": 1, "b": "y"}'])
        adapter._call_service = transport
        result = self.quiet(adapter.generate_json, "t", "sys", {}, SCHEMA)
        self.assertEqual(result, {"a": 1, "b": "y"})
        self.assertEqual(len(transport.calls), 2)
        second_user = transport.calls[1]["messages"][-1]["content"]
        self.assertIn("未通过程序校验", second_user)
        self.assertIn("integer", second_user)

    def test_service_errors_exhaust_attempts(self):
        adapter = make_adapter()
        transport = FakeTransport([ServiceError("down"), ServiceError("down"),
                                   ServiceError("down")])
        adapter._call_service = transport
        with self.assertRaises(LLMError) as caught:
            self.quiet(adapter.generate_json, "t", "sys", {}, SCHEMA)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(caught.exception.error_type, "service_unavailable")

    def test_timeout_error_type(self):
        adapter = make_adapter()
        transport = FakeTransport([RequestTimeoutError("slow"), RequestTimeoutError("slow"),
                                   RequestTimeoutError("slow")])
        adapter._call_service = transport
        with self.assertRaises(LLMError) as caught:
            self.quiet(adapter.generate_json, "t", "sys", {}, SCHEMA)
        self.assertEqual(caught.exception.error_type, "timeout")

    def test_context_budget_blocks_before_service_call(self):
        adapter = make_adapter(context_budget=10, max_output_tokens=1)
        transport = FakeTransport(['{"a": 1, "b": "x"}'])
        adapter._call_service = transport
        with self.assertRaises(ContextBudgetError):
            self.quiet(adapter.generate_json, "t", "sys", {"long": "字" * 50}, SCHEMA)
        self.assertEqual(len(transport.calls), 0)

    def test_endpoint_url_variants(self):
        for base, expected in [
            ("http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1/chat/completions"),
            ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1/chat/completions"),
            ("http://127.0.0.1:1234/v1/chat/completions",
             "http://127.0.0.1:1234/v1/chat/completions"),
        ]:
            adapter = make_adapter(base_url=base)
            self.assertEqual(adapter._endpoint(), expected)

    def test_build_llm_dispatch_and_unknown_adapter(self):
        self.assertEqual(type(build_llm(make_config())), OpenAIHttpAdapter)
        with self.assertRaises(ConfigError):
            build_llm({"adapter": "missing"})
        self.assertEqual(set(ADAPTERS), {"openai_http"})

    def test_missing_base_url_is_config_error(self):
        config = make_config()
        del config["base_url"]
        with self.assertRaises(ConfigError):
            OpenAIHttpAdapter(config)


class PromptBudgetTests(unittest.TestCase):
    """context_budget is a TOKEN window; the prompt only gets what is left."""

    def test_output_window_is_reserved(self):
        config = make_config(context_budget=1000, max_output_tokens=400)
        self.assertEqual(prompt_char_budget(config), 600)
        self.assertEqual(make_adapter(context_budget=1000,
                                      max_output_tokens=400).prompt_char_budget, 600)

    def test_chars_per_token_scales_the_character_budget(self):
        config = make_config(context_budget=1000, max_output_tokens=400,
                             chars_per_token=4)
        self.assertEqual(prompt_char_budget(config), 2400)

    def test_default_ratio_is_one_char_per_token(self):
        adapter = make_adapter(context_budget=1000, max_output_tokens=400)
        self.assertEqual(adapter.chars_per_token, 1)

    def test_prompt_over_reserved_budget_is_blocked_before_the_call(self):
        adapter = make_adapter(context_budget=1000, max_output_tokens=400)
        transport = FakeTransport(['{"a": 1, "b": "x"}'])
        adapter._call_service = transport
        with self.assertRaises(ContextBudgetError) as caught:
            with contextlib.redirect_stderr(io.StringIO()):
                adapter.generate_json("t", "sys", {"long": "字" * 700}, SCHEMA)
        self.assertIn("600", str(caught.exception))
        self.assertEqual(len(transport.calls), 0)

    def test_output_window_must_fit_inside_the_context_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(make_config(context_budget=1000,
                                                   max_output_tokens=1000)),
                            encoding="utf-8")
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
            self.assertIn("max_output_tokens", str(caught.exception))

    def test_bad_chars_per_token_is_config_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (0, -1, "four"):
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(make_config(chars_per_token=value)),
                                encoding="utf-8")
                with self.assertRaises(ConfigError, msg=repr(value)):
                    load_config(path)

    def test_describe_parameters_records_the_generation_settings(self):
        adapter = make_adapter(context_budget=1000, max_output_tokens=400,
                               temperature=0.2)
        parameters = adapter.describe_parameters()
        self.assertEqual(parameters["adapter"], "openai_http")
        self.assertEqual(parameters["temperature"], 0.2)
        self.assertEqual(parameters["context_budget_tokens"], 1000)
        self.assertEqual(parameters["max_output_tokens"], 400)
        self.assertEqual(parameters["prompt_char_budget"], 600)
        self.assertEqual(parameters["max_attempts"], 3)


class ValidateCallbackTests(unittest.TestCase):
    """Program-side checks share the ONE three-attempt budget per request."""

    GOOD = '{"a": 1, "b": "x"}'

    def run_with(self, responses, validate):
        adapter = make_adapter()
        transport = FakeTransport(responses)
        adapter._call_service = transport
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                data = adapter.generate_json("t", "sys", {}, SCHEMA,
                                             validate=validate)
            except LLMError as exc:
                return transport, None, exc
        return transport, data, None

    def test_always_failing_validation_stops_at_three_attempts(self):
        transport, data, error = self.run_with(
            [self.GOOD] * 5, lambda payload: ["引用不是原文"])
        self.assertIsNone(data)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(error.error_type, "structure")
        self.assertIn("引用不是原文", str(error))

    def test_repair_on_second_attempt_returns_data(self):
        seen = []

        def validate(payload):
            seen.append(payload["a"])
            return [] if payload["a"] == 2 else ["a 必须是 2"]

        transport, data, error = self.run_with(
            ['{"a": 1, "b": "x"}', '{"a": 2, "b": "x"}'], validate)
        self.assertIsNone(error)
        self.assertEqual(data, {"a": 2, "b": "x"})
        self.assertEqual(seen, [1, 2])
        self.assertEqual(len(transport.calls), 2)
        correction = transport.calls[1]["messages"][-1]["content"]
        self.assertIn("a 必须是 2", correction)

    def test_schema_failure_and_validation_failure_share_the_budget(self):
        transport, data, error = self.run_with(
            ['{"a": "not int", "b": "x"}', self.GOOD, self.GOOD],
            lambda payload: ["候选 ID 不属于本次请求"])
        self.assertIsNone(data)
        self.assertEqual(len(transport.calls), 3)
        self.assertIn("候选 ID 不属于本次请求", str(error))

    def test_no_validate_callback_keeps_previous_behaviour(self):
        transport, data, error = self.run_with([self.GOOD], None)
        self.assertIsNone(error)
        self.assertEqual(data, {"a": 1, "b": "x"})
        self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main()


class TruncationTests(unittest.TestCase):
    def test_finish_reason_length_is_a_structure_error(self):
        import io
        from unittest import mock
        import local_llm

        adapter = local_llm.build_llm(make_config())

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout):
            return FakeResponse(json.dumps({"choices": [{"finish_reason": "length",
                                                         "message": {"content": '{"a": [1, 2'}}]}).encode("utf-8"))

        with mock.patch.object(local_llm.urllib.request, "urlopen", fake_urlopen):
            with self.assertRaises(local_llm.StructureError) as ctx:
                adapter._call_service([{"role": "user", "content": "hi"}], 5, None)
        self.assertIn("被截断", str(ctx.exception))


class ExtraBodyTests(unittest.TestCase):
    """extra_body fields are merged into every openai_http request body."""

    def test_extra_body_merged_into_request(self):
        import io
        from unittest import mock
        import local_llm

        config = make_config()
        config["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        adapter = local_llm.build_llm(config)
        sent = {}

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout):
            sent["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(json.dumps(
                {"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8"))

        with mock.patch.object(local_llm.urllib.request, "urlopen", fake_urlopen):
            raw = adapter._call_service([{"role": "user", "content": "hi"}], 5, None)
        self.assertEqual(raw, "{\"ok\": true}")
        self.assertEqual(sent["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(adapter.describe_parameters()["extra_body"], config["extra_body"])
