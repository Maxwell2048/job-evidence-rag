import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import match_job
from jd_parser import (EXTRACTION_SCHEMA, EXTRACTION_SYSTEM_PROMPT,
                       NOTE_TEMPLATE, plan_batches)
from local_llm import LLMError

VALID_CONFIG = {
    "adapter": "openai_http",
    "model": "fake-model",
    "base_url": "http://127.0.0.1:9/v1",
    "context_budget": 100000,
    "timeout_seconds": 5,
    "max_output_tokens": 100,
}

JD_TEXT = ("Senior ML Engineer\r\n"
           "\r\n"
           "Requirements\r\n"
           "\r\n"
           "- Python experience is required.\r\n"
           "- AWS experience is desirable.\r\n")


FAKE_GENERATION_PARAMETERS = {
    "adapter": "openai_http",
    "temperature": 0,
    "context_budget_tokens": 100000,
    "max_output_tokens": 100,
    "chars_per_token": 1,
    "prompt_char_budget": 99900,
    "timeout_seconds": 5,
    "max_attempts": 3,
    "response_format": "json_object",
}


class FakeLLM:
    def __init__(self, response=None, error=None, prompt_char_budget=100000):
        self.response = response
        self.error = error
        self.prompt_char_budget = prompt_char_budget
        self.calls = []

    def estimate_context(self, task, system_prompt, payload, schema):
        return len(system_prompt) + len(json.dumps(payload, ensure_ascii=False)) + 200

    def generate_json(self, task, system_prompt, payload, schema=None,
                      timeout_seconds=None, validate=None):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response

    def describe_model(self):
        return {"name": "fake-model", "version": "9.9"}

    def describe_parameters(self):
        return dict(FAKE_GENERATION_PARAMETERS)


SECTION_TEXT = ("我在 Synthetic Project 项目中负责用 Python 实现数据标注流水线，"
                "使用 Python 编写标注工具并处理合成测试图片集。")


def section_result(score):
    return {
        "project": "Synthetic Project",
        "section": "我的贡献",
        "source": "synthetic_project.md",
        "line_start": 22,
        "line_end": 27,
        "text": SECTION_TEXT,
        "matched_text": SECTION_TEXT,
        "match_char_start": 0,
        "match_char_end": len(SECTION_TEXT),
        "score": score,
        "section_id": "ab12cd34ef56ab12",
        "chunk_id": "cd34ef56ab12cd34",
        "evidence": ["- 来源：项目笔记 2024-03"],
    }


class FakeRetriever:
    def __init__(self, data_dir, offline=False, all_sections=False):
        self.data_dir = data_dir

    def build(self):
        return self

    def search(self, query, top_k=5):
        return [section_result(0.8), section_result(0.6)][:top_k]


class SequenceFakeLLM:
    """Serves extraction responses (one per batch), then judgments."""

    def __init__(self, extraction, judgments, error_by_id=(),
                 prompt_char_budget=100000):
        if isinstance(extraction, (list, tuple)):
            self.extractions = list(extraction)
        else:
            self.extractions = [extraction]
        self.judgments = judgments
        self.error_by_id = dict(error_by_id)
        self.prompt_char_budget = prompt_char_budget
        self.calls = []

    def estimate_context(self, task, system_prompt, payload, schema):
        return len(system_prompt) + len(json.dumps(payload, ensure_ascii=False)) + 200

    def generate_json(self, task, system_prompt, payload, schema=None,
                      timeout_seconds=None, validate=None):
        if task == "job_readiness_summary":
            return {"overview": "合成测试总结", "advice": [
                {"requirement_id": m["requirement_id"], "analysis": "按已给定证据准备。",
                 "next_step": "核实资料。", "knowledge_points": [], "exercise": "",
                 "acceptance_check": "", "urgency": "confirm_first"}
                for m in payload["matches"]]}
        if task == "extract_requirements":
            self.calls.append("extract")
            response = self.extractions.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        requirement_id = payload["requirement"]["requirement_id"]
        self.calls.append("judge:" + requirement_id)
        if requirement_id in self.error_by_id:
            raise self.error_by_id[requirement_id]
        response = self.judgments[requirement_id]
        errors = list(validate(response)) if validate is not None else []
        if errors:
            raise LLMError("；".join(errors), error_type="structure")
        return response

    def describe_model(self):
        return {"name": "fake-model", "version": "9.9"}

    def describe_parameters(self):
        return dict(FAKE_GENERATION_PARAMETERS)


def judgment(requirement_id, verdict, quote, reason,
             missing=("生产环境经验",)):
    return {
        "requirement_id": requirement_id,
        "verdict": verdict,
        "reason": reason,
        "evidence": [{"candidate_id": "E001", "quote": quote}] if quote else [],
        "missing_aspects": list(missing),
        "questions_for_user": ["相关经验是否用于生产环境？"],
    }


def extraction_response():
    return {
        "requirements": [
            {
                "text": "Python experience",
                "category": "technical_skill",
                "importance": "required",
                "source_quote": "Python experience is required.",
                "importance_source_quote": "Python experience is required.",
                "qualifiers": [],
                "search_queries": ["python development"],
                "needs_review": False,
            },
            {
                "text": "AWS experience",
                "category": "technical_skill",
                "importance": "preferred",
                "source_quote": "AWS experience is desirable.",
                "importance_source_quote": "AWS experience is desirable.",
                "qualifiers": [],
                "search_queries": ["aws cloud"],
                "needs_review": False,
            },
        ],
        "requirement_groups": [
            {"operator": "OR",
             "requirement_texts": ["Python experience", "AWS experience"],
             "source_quote": "Python or AWS experience is required."},
        ],
    }


class RetrieverThatCrashes:
    def __init__(self, data_dir, offline=False, all_sections=False):
        pass

    def build(self):
        raise RuntimeError("sentence-transformers is not installed")


class EmptyCorpusRetriever:
    def __init__(self, data_dir, offline=False, all_sections=False):
        pass

    def build(self):
        raise ValueError("no experience files found")


def run_cli(args, fake_llm, retriever_cls=None):
    with mock.patch.object(match_job, "build_llm", return_value=fake_llm), \
            mock.patch.object(match_job, "ExperienceRetriever",
                              retriever_cls or mock.DEFAULT):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = match_job.main(args)
            except SystemExit as exc:
                code = exc.code
    return code, stdout.getvalue(), stderr.getvalue()


class MatchJobCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.jd = self.tmp / "jd.txt"
        self.jd.write_bytes(("\ufeff" + JD_TEXT).encode("utf-8"))
        self.config = self.tmp / "config.json"
        self.config.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")
        self.out = self.tmp / "outputs"

    def tearDown(self):
        self._tmp.cleanup()

    def base_args(self, *extra):
        return ["--jd", str(self.jd), "--config", str(self.config),
                "--out", str(self.out), *extra]

    def test_extract_only_json_output(self):
        fake = FakeLLM(response=extraction_response())
        code, out, err = run_cli(self.base_args("--extract-only", "--json"), fake)
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(len(doc["requirements"]), 2)
        self.assertEqual(doc["review_count"], 0)
        for requirement in doc["requirements"]:
            self.assertIsNotNone(requirement["source_span"])
            self.assertEqual(requirement["source_span"]["line_start"],
                             requirement["source_span"]["line_end"])
        run_dirs = list(self.out.iterdir())
        self.assertEqual(len(run_dirs), 1)
        files = {p.name for p in run_dirs[0].iterdir()}
        self.assertEqual(files, {"jd.txt", "requirements.json", "run_meta.json"})
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["run_status"], "complete")
        self.assertEqual(meta["scope"], "extraction")
        self.assertEqual(meta["models"]["generator"]["name"], "fake-model")
        self.assertEqual(meta["models"]["generator"]["version"], "9.9")
        self.assertEqual(meta["input"]["jd_sha256"],
                         hashlib.sha256(self.jd.read_bytes()).hexdigest())
        normalized = (run_dirs[0] / "jd.txt").read_text(encoding="utf-8")
        self.assertEqual(normalized, JD_TEXT.replace("\r\n", "\n"))
        self.assertEqual(len(fake.calls), 1)

    def test_rerun_does_not_overwrite_previous_run(self):
        fake = FakeLLM(response=extraction_response())
        code1, _, _ = run_cli(self.base_args("--extract-only", "--json"), fake)
        code2, _, _ = run_cli(self.base_args("--extract-only", "--json"), fake)
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(len(list(self.out.iterdir())), 2)

    def test_human_summary_when_not_json(self):
        fake = FakeLLM(response=extraction_response())
        code, out, err = run_cli(self.base_args("--extract-only"), fake)
        self.assertEqual(code, 0)
        self.assertIn("状态：complete", out)
        self.assertIn("要求：2 条", out)
        self.assertIn("正在提取 JD 要求", err)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(out)

    def test_cli_argument_errors_exit_2(self):
        cases = [
            ["--jd", str(self.tmp / "missing.txt"), "--config", str(self.config),
             "--out", str(self.out), "--extract-only"],
            ["--jd", str(self.jd), "--config", str(self.tmp / "missing.json"),
             "--out", str(self.out), "--extract-only"],
            ["--jd", str(self.jd), "--config", str(self.config),
             "--out", str(self.out), "--top-k", "0", "--extract-only"],
        ]
        for args in cases:
            with self.subTest(args=args):
                code, _, _ = run_cli(args, FakeLLM(response=extraction_response()))
                self.assertEqual(code, 2)
        wrong_ext = self.tmp / "jd.docx"
        wrong_ext.write_bytes(b"x")
        code, _, _ = run_cli(["--jd", str(wrong_ext), "--config", str(self.config),
                              "--out", str(self.out), "--extract-only"],
                             FakeLLM(response=extraction_response()))
        self.assertEqual(code, 2)

    def test_invalid_config_content_exits_1_not_2(self):
        """The command was well formed; the run started and then failed."""
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"adapter": "openai_http"}), encoding="utf-8")
        code, _, _ = run_cli(["--jd", str(self.jd), "--config", str(bad),
                              "--out", str(self.out), "--extract-only"],
                             FakeLLM(response=extraction_response()))
        self.assertEqual(code, 1)
        run_dirs = list(self.out.iterdir())
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["run_status"], "failed")
        self.assertEqual(meta["errors"][0]["error_type"], "config")

    def test_model_unavailable_exits_1_and_records_error(self):
        fake = FakeLLM(error=LLMError("本机服务不可用：refused",
                                       error_type="service_unavailable"))
        code, out, _ = run_cli(self.base_args("--extract-only", "--json"), fake)
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "failed")
        self.assertEqual(doc["errors"][0]["error_type"], "service_unavailable")
        run_dirs = list(self.out.iterdir())
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["run_status"], "failed")

    def test_extract_only_partial_when_a_batch_fails(self):
        long_jd = "\n\n".join(
            f"Paragraph {i} text with token TK{i} required here.\n"
            for i in range(8))
        jd = self.tmp / "long_jd.txt"
        jd.write_text(long_jd, encoding="utf-8")
        probe = SequenceFakeLLM(extraction={}, judgments={})
        max_note = NOTE_TEMPLATE.format(index=999, count=999)
        budget = probe.estimate_context(
            "extract_requirements", EXTRACTION_SYSTEM_PROMPT,
            {"jd_text": long_jd + " " * 64, "note": max_note},
            EXTRACTION_SCHEMA) - 1
        batch_count = len(plan_batches(long_jd, SequenceFakeLLM(
            extraction={}, judgments={}, prompt_char_budget=budget)))
        self.assertGreater(batch_count, 1)
        requirement_item = {
            "text": "TK0 requirement",
            "category": "technical_skill",
            "importance": "required",
            "source_quote": "token TK0 required here.",
            "importance_source_quote": "token TK0 required here.",
            "qualifiers": [],
            "search_queries": ["tk0"],
            "needs_review": False,
        }
        empty = {"requirements": [], "requirement_groups": []}
        fake = SequenceFakeLLM(
            extraction=[{"requirements": [requirement_item],
                         "requirement_groups": []}]
            + [empty] * (batch_count - 2)
            + [LLMError("超时", error_type="timeout")],
            judgments={},
            prompt_char_budget=budget)
        code, out, _ = run_cli(
            ["--jd", str(jd), "--config", str(self.config),
             "--out", str(self.out), "--extract-only", "--json"], fake)
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "partial")
        self.assertEqual(len(doc["requirements"]), 1)
        self.assertEqual(doc["requirements"][0]["requirement_id"], "R001")
        self.assertEqual(len(doc["failed_batches"]), 1)
        failed = doc["failed_batches"][0]
        self.assertEqual(failed["index"], batch_count)
        self.assertGreater(failed["line_start"], 1)
        run_dirs = list(self.out.iterdir())
        self.assertEqual(len(run_dirs), 1)
        files = {p.name for p in run_dirs[0].iterdir()}
        self.assertEqual(files, {"jd.txt", "requirements.json", "run_meta.json"})
        saved = json.loads(
            (run_dirs[0] / "requirements.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved["requirements"]), 1)
        self.assertEqual(len(saved["failed_batches"]), 1)
        meta = json.loads(
            (run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["run_status"], "partial")
        self.assertTrue(any("第 %d/%d 批提取失败" % (batch_count, batch_count)
                            in error["message"]
                            for error in meta["errors"]))

    def test_blank_jd_exits_1_with_status_json(self):
        empty = self.tmp / "empty.md"
        empty.write_bytes(b"   \n")
        code, out, _ = run_cli(["--jd", str(empty), "--config", str(self.config),
                                "--out", str(self.out), "--extract-only", "--json"],
                               FakeLLM(response=extraction_response()))
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "failed")
        self.assertEqual(doc["scope"], "extraction")
        self.assertEqual(doc["errors"][0]["error_type"], "input")

    def make_llm(self, error_by_id=()):
        quote = "使用 Python 编写标注工具并处理合成测试图片集。"
        return SequenceFakeLLM(
            extraction=extraction_response(),
            judgments={
                "R001": judgment("R001", "direct", quote,
                                 "候选人使用 Python 实现了数据标注流水线。",
                                 missing=()),
                "R002": judgment("R002", "related",
                                 "负责用 Python 实现数据标注流水线",
                                 "标注流水线与 AWS 云服务仅部分相关。"),
            },
            error_by_id=error_by_id)

    def make_data_dir(self):
        data_dir = self.tmp / "data"
        data_dir.mkdir()
        (data_dir / "synthetic_project.md").write_text(
            "# Synthetic Project\n\n## 我的贡献\n" + SECTION_TEXT + "\n", encoding="utf-8")
        return data_dir

    def test_full_run_complete_writes_all_outputs(self):
        data_dir = self.make_data_dir()
        fake = self.make_llm()
        code, out, err = run_cli(
            self.base_args("--data", str(data_dir), "--json"),
            fake, retriever_cls=FakeRetriever)
        self.assertEqual(code, 0, out)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "complete")
        self.assertEqual(len(doc["matches"]), 2)
        first = doc["matches"][0]
        self.assertEqual(first["requirement_id"], "R001")
        self.assertEqual(first["processing_status"], "ok")
        self.assertEqual(first["verdict"], "direct")
        self.assertEqual(first["evidence"][0]["quote"],
                         "使用 Python 编写标注工具并处理合成测试图片集。")
        self.assertEqual(first["evidence"][0]["source"], "synthetic_project.md")
        self.assertEqual((first["evidence"][0]["line_start"],
                          first["evidence"][0]["line_end"]), (22, 27))
        self.assertEqual(first["evidence"][0]["candidate_id"], "E001")
        self.assertEqual(first["evidence"][0]["project"], "Synthetic Project")
        self.assertEqual(first["candidates"][0]["candidate_id"], "E001")
        self.assertEqual(first["candidates"][0]["score"], 0.8)
        second = doc["matches"][1]
        self.assertEqual(second["verdict"], "related")
        self.assertEqual(second["evidence"][0]["quote"],
                         "负责用 Python 实现数据标注流水线")
        run_dirs = list(self.out.iterdir())
        self.assertEqual(len(run_dirs), 1)
        files = {p.name for p in run_dirs[0].iterdir()}
        self.assertEqual(files, {"jd.txt", "requirements.json", "matches.json",
                                 "report.md", "run_meta.json", "job_summary.json"})
        self.assertEqual(doc["job_summary"]["status"], "complete")
        matches_file = json.loads(
            (run_dirs[0] / "matches.json").read_text(encoding="utf-8"))
        self.assertEqual(matches_file["matches"], doc["matches"])
        requirements_file = json.loads(
            (run_dirs[0] / "requirements.json").read_text(encoding="utf-8"))
        self.assertEqual(len(requirements_file["requirements"]), 2)
        self.assertNotIn("matches", requirements_file)
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["run_status"], "complete")
        self.assertEqual(meta["scope"], "full")
        self.assertEqual(meta["prompts"]["judgment"], "2")
        self.assertEqual(meta["models"]["generator"]["parameters"],
                         FAKE_GENERATION_PARAMETERS)
        self.assertEqual(meta["models"]["embedding"]["name"],
                         "intfloat/multilingual-e5-small")
        self.assertEqual(len(meta["experience_files"]), 1)
        recorded = meta["experience_files"][0]
        self.assertEqual(recorded["path"], "synthetic_project.md")
        self.assertEqual(recorded["sha256"], hashlib.sha256(
            (data_dir / "synthetic_project.md").read_bytes()).hexdigest())
        report = (run_dirs[0] / "report.md").read_text(encoding="utf-8")
        self.assertIn("要求概览", report)
        self.assertIn("直接支持", report)
        self.assertIn("相关但不足", report)
        self.assertIn("仅用于检索排序", report)
        self.assertIn("使用 Python 编写标注工具并处理合成测试图片集。", report)
        self.assertIn("待确认：相关经验是否用于生产环境？", report)
        self.assertIn("## AND/OR 关系", report)
        self.assertIn("G001 二选一", report)
        self.assertIn("组关系", report)

    def test_full_run_partial_when_judgment_fails(self):
        data_dir = self.make_data_dir()
        fake = self.make_llm(error_by_id={
            "R002": LLMError("模型超时", error_type="timeout")})
        code, out, _ = run_cli(
            self.base_args("--data", str(data_dir), "--json"),
            fake, retriever_cls=FakeRetriever)
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "partial")
        by_id = {m["requirement_id"]: m for m in doc["matches"]}
        self.assertEqual(by_id["R001"]["processing_status"], "ok")
        self.assertEqual(by_id["R002"]["processing_status"], "error")
        self.assertIsNone(by_id["R002"]["verdict"])
        run_dirs = list(self.out.iterdir())
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["run_status"], "partial")
        report = (run_dirs[0] / "report.md").read_text(encoding="utf-8")
        self.assertIn("处理错误", report)
        self.assertIn("未完成/出错 1 条", report)

    def test_run_meta_summarises_per_requirement_failures(self):
        data_dir = self.make_data_dir()
        fake = self.make_llm(error_by_id={
            "R002": LLMError("模型超时", error_type="timeout")})
        code, _, _ = run_cli(self.base_args("--data", str(data_dir), "--json"),
                             fake, retriever_cls=FakeRetriever)
        self.assertEqual(code, 1)
        run_dirs = list(self.out.iterdir())
        meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
        failures = [e for e in meta["errors"]
                    if e["error_type"].startswith("judgment_")]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error_type"], "judgment_failed")
        self.assertIn("R002", failures[0]["message"])
        self.assertIn("模型超时", failures[0]["message"])
        self.assertNotIn("R001", failures[0]["message"])

    def test_retriever_runtime_error_exits_1_with_status_json(self):
        data_dir = self.make_data_dir()
        fake = self.make_llm()
        code, out, _ = run_cli(
            self.base_args("--data", str(data_dir), "--json"),
            fake, retriever_cls=RetrieverThatCrashes)
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "failed")
        self.assertEqual(doc["scope"], "full")
        self.assertEqual(doc["errors"][0]["error_type"], "runtime")
        self.assertIn("sentence-transformers", doc["errors"][0]["message"])
        self.assertEqual(len(doc["requirements"]), 2)
        self.assertNotIn("matches", doc)

    def test_retriever_value_error_exits_1_with_status_json(self):
        data_dir = self.make_data_dir()
        fake = self.make_llm()
        code, out, _ = run_cli(
            self.base_args("--data", str(data_dir), "--json"),
            fake, retriever_cls=EmptyCorpusRetriever)
        self.assertEqual(code, 1)
        doc = json.loads(out)
        self.assertEqual(doc["run_status"], "failed")
        self.assertEqual(doc["scope"], "full")
        self.assertEqual(doc["errors"][0]["error_type"], "input")
        self.assertEqual(len(doc["requirements"]), 2)


if __name__ == "__main__":
    unittest.main()
