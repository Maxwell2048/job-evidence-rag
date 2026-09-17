import contextlib
import io
import json
import unittest
import unittest.mock
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence_matcher
from evidence_matcher import (JUDGMENT_SCHEMA, JUDGMENT_SYSTEM_PROMPT,
                              EvidenceMatcher, _requirement_payload, _split_text,
                              merge_candidates, validate_judgment)
from local_llm import LLMError

SECTION_A = {
    "project": "Synthetic Project",
    "section": "我的贡献 标注",
    "source": "synthetic_project.md",
    "line_start": 22,
    "line_end": 27,
    "text": ("我负责示例地图图片标注。完成并提交一批 12 幅示例地图的标注，"
             "以 LabelMe 格式记录制图元素。\n"
             "相关能力：Image Annotation。\n"
             "证据：[标注提交](https://github.com/example/commit/aaa)。"),
    "section_id": "sA",
    "chunk_id": "cA",
    "score": 0.8,
    "evidence": ["证据：[标注提交](https://github.com/example/commit/aaa)。"],
    "search_keywords": "相关能力：Image Annotation。",
}

SECTION_B = {
    "project": "CITS5508",
    "section": "我的工作 调参",
    "source": "cits5508_ridge_svm.md",
    "line_start": 26,
    "line_end": 31,
    "text": "我通过 GridSearchCV 对 C 和 gamma 执行参数搜索。",
    "section_id": "sB",
    "chunk_id": "cB",
    "score": 0.7,
    "evidence": ["证据：[原始 Notebook](<C:/Users/x/a.ipynb>)，Cell 30。"],
    "search_keywords": "相关能力：Hyperparameter Search。",
}


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.searches = []

    def search(self, query, top_k=5):
        self.searches.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.results.get(query, [])[:top_k]


class FakeLLM:
    """Mimics BaseAdapter.generate_json: the caller's validate() callback shares
    the one three-attempt budget, so a repair never buys extra retries."""
    max_attempts = 3

    def __init__(self, responses, prompt_char_budget=100000):
        self.responses = list(responses)
        self.prompt_char_budget = prompt_char_budget
        self.calls = []  # one entry per attempt; repairs carry fix_notes

    def estimate_context(self, task, system_prompt, payload, schema):
        return len(system_prompt) + len(json.dumps(payload, ensure_ascii=False)) + 200

    def generate_json(self, task, system_prompt, payload, schema=None,
                      timeout_seconds=None, validate=None):
        errors = []
        for _ in range(self.max_attempts):
            self.calls.append({**payload, "fix_notes": list(errors)}
                              if errors else dict(payload))
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            errors = list(validate(response)) if validate is not None else []
            if not errors:
                return response
            if not self.responses:
                break
        raise LLMError("；".join(errors), error_type="structure")


def make_requirement(requirement_id="R001", queries=("标注", "图片标注")):
    return {
        "requirement_id": requirement_id,
        "text": "Experience annotating images for object detection",
        "category": "technical_skill",
        "importance": "required",
        "source_quote": "Image annotation experience is required.",
        "source_span": {"start": 0, "end": 4, "line_start": 1, "line_end": 1},
        "importance_source_quote": "Image annotation experience is required.",
        "importance_source_span": None,
        "qualifiers": [],
        "search_queries": list(queries),
        "needs_review": False,
        "review_notes": [],
    }


def judgment(requirement_id, verdict, quotes=(), reason="证据与要求直接相关。",
             missing=(), questions=()):
    return {
        "requirement_id": requirement_id,
        "verdict": verdict,
        "reason": reason,
        "evidence": [{"candidate_id": cid, "quote": quote} for cid, quote in quotes],
        "missing_aspects": list(missing),
        "questions_for_user": list(questions),
    }


def make_matcher(llm, retriever, top_k=5):
    return EvidenceMatcher(llm, retriever, top_k=top_k)


class MergeCandidatesTests(unittest.TestCase):
    def test_keeps_best_score_and_producing_query(self):
        weak_a = {**SECTION_A, "score": 0.6}
        strong_a = {**SECTION_A, "score": 0.9}
        merged = merge_candidates([("q1", [weak_a, SECTION_B]), ("q2", [strong_a])])
        self.assertEqual([r["section_id"] for r in merged], ["sA", "sB"])
        self.assertEqual(merged[0]["score"], 0.9)
        self.assertEqual(merged[0]["matched_by_query"], "q2")
        self.assertEqual(merged[1]["matched_by_query"], "q1")
        self.assertEqual(len({r["section_id"] for r in merged}), 2)


class ValidateJudgmentTests(unittest.TestCase):
    def setUp(self):
        self.text_by_id = {"E001": SECTION_A["text"], "E002": SECTION_B["text"]}

    def test_requirement_id_mismatch(self):
        errors, _ = validate_judgment(
            judgment("R002", "insufficient"), "R001", self.text_by_id)
        self.assertTrue(any("requirement_id" in e for e in errors))

    def test_unknown_candidate_id_and_bad_quote(self):
        errors, valid = validate_judgment(
            judgment("R001", "direct",
                     quotes=[("E999", "某句"), ("E001", "不存在的句子")]),
            "R001", self.text_by_id)
        self.assertTrue(any("E999" in e for e in errors))
        self.assertTrue(any("连续子串" in e for e in errors))
        self.assertEqual(valid, [])

    def test_quote_from_wrong_candidate_is_rejected(self):
        # "我通过 GridSearchCV..." exists only in E002's text; citing it under
        # E001 must fail even though it is a real sentence.
        errors, valid = validate_judgment(
            judgment("R001", "direct",
                     quotes=[("E001", "我通过 GridSearchCV 对 C 和 gamma 执行参数搜索。")]),
            "R001", self.text_by_id)
        self.assertTrue(any("连续子串" in e for e in errors))
        self.assertEqual(valid, [])

    def test_direct_requires_valid_quote(self):
        errors, _ = validate_judgment(judgment("R001", "direct"), "R001",
                                      self.text_by_id)
        self.assertTrue(any("至少需要一条有效引用" in e for e in errors))

    def test_insufficient_needs_no_quote(self):
        errors, valid = validate_judgment(judgment("R001", "insufficient"), "R001",
                                          self.text_by_id)
        self.assertEqual(errors, [])
        self.assertEqual(valid, [])

    def test_direct_with_missing_aspects_is_contradiction(self):
        errors, _ = validate_judgment(
            judgment("R001", "direct",
                     quotes=[("E001", "我负责示例地图图片标注。")],
                     missing=["生产环境部署"]),
            "R001", self.text_by_id)
        self.assertTrue(any("direct 与未覆盖条件矛盾" in e for e in errors))

    def test_related_with_missing_aspects_is_allowed(self):
        errors, valid = validate_judgment(
            judgment("R001", "related",
                     quotes=[("E001", "我负责示例地图图片标注。")],
                     missing=["生产环境部署"]),
            "R001", self.text_by_id)
        self.assertEqual(errors, [])
        self.assertEqual(len(valid), 1)


class EvidenceMatcherTests(unittest.TestCase):
    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)

    def test_valid_direct_judgment_fills_program_metadata(self):
        retriever = FakeRetriever({"标注": [SECTION_A], "图片标注": [SECTION_A]})
        llm = FakeLLM([judgment("R001", "direct",
                                quotes=[("E001", "我负责示例地图图片标注。")])])
        matcher = make_matcher(llm, retriever)
        result = matcher.match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "direct")
        self.assertEqual(len(result["evidence"]), 1)
        evidence = result["evidence"][0]
        self.assertEqual(evidence["quote"], "我负责示例地图图片标注。")
        self.assertEqual(evidence["source"], "synthetic_project.md")
        self.assertEqual((evidence["line_start"], evidence["line_end"]), (22, 27))
        self.assertEqual(evidence["section_id"], "sA")
        self.assertEqual(evidence["chunk_id"], "cA")
        self.assertEqual(evidence["evidence_lines"], SECTION_A["evidence"])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["candidate_id"], "E001")
        self.assertEqual(result["candidates"][0]["matched_by_query"], "标注")
        # The model saw the full section text but no paths or line numbers.
        sent = llm.calls[0]["candidates"][0]
        self.assertEqual(sent["text"], SECTION_A["text"])
        self.assertNotIn("source", sent)
        self.assertNotIn("line_start", sent)
        self.assertEqual(len(retriever.searches), 2)

    def test_forged_candidate_id_fixed_on_retry(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        good = judgment("R001", "direct", quotes=[("E001", "我负责示例地图图片标注。")])
        llm = FakeLLM([judgment("R001", "direct", quotes=[("E999", "伪造")]), good])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "direct")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("fix_notes", llm.calls[1])

    def test_forged_candidate_id_still_fails_is_error_not_insufficient(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        llm = FakeLLM([judgment("R001", "direct", quotes=[("E999", "伪造")])] * 2)
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertIsNone(result["verdict"])
        self.assertTrue(result["errors"])

    def test_spacing_only_difference_is_repaired_to_the_verbatim_text(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        loose_quote = "完成并提交一批12幅示例地图的标注"  # original has spaces
        llm = FakeLLM([judgment("R001", "direct", quotes=[("E001", loose_quote)])])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["evidence"][0]["quote"], "完成并提交一批 12 幅示例地图的标注")

    def test_quote_not_in_full_text_fails(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        bad_quote = "完成并提交一批 120 幅示例地图的标注"  # the number was changed
        llm = FakeLLM([judgment("R001", "direct", quotes=[("E001", bad_quote)])] * 2)
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertTrue(any("连续子串" in e for e in result["errors"]))

    def test_direct_without_valid_evidence_is_error(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        llm = FakeLLM([judgment("R001", "direct")] * 2)
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertIsNone(result["verdict"])

    def test_related_and_insufficient_accepted(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        llm = FakeLLM([judgment("R001", "related",
                                quotes=[("E001", "我负责示例地图图片标注。")],
                                missing=["未覆盖生产环境"])])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "related")
        self.assertEqual(result["missing_aspects"], ["未覆盖生产环境"])
        llm2 = FakeLLM([judgment("R001", "insufficient",
                                 missing=["当前检索资料未提供证据"])])
        result2 = make_matcher(llm2, FakeRetriever({"标注": [SECTION_A]})) \
            .match_requirement(make_requirement())
        self.assertEqual(result2["processing_status"], "ok")
        self.assertEqual(result2["verdict"], "insufficient")
        self.assertEqual(result2["evidence"], [])

    def test_model_error_is_error_status_not_insufficient(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        llm = FakeLLM([LLMError("本机服务不可用", error_type="service_unavailable")])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertIsNone(result["verdict"])

    def test_retrieval_failure_is_error_status(self):
        retriever = FakeRetriever(error=ValueError("没有可检索的内容"))
        llm = FakeLLM([])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertTrue(any("检索失败" in e for e in result["errors"]))

    def test_no_candidates_is_error_status(self):
        llm = FakeLLM([])
        result = make_matcher(llm, FakeRetriever({})).match_requirement(
            make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertTrue(any("没有可检索的候选章节" in e for e in result["errors"]))

    def test_top_k_limit_and_multi_query(self):
        sections = [{**SECTION_B, "section_id": f"s{i}", "score": 0.9 - i * 0.1}
                    for i in range(4)]
        retriever = FakeRetriever({"q1": sections, "q2": [SECTION_A]})
        llm = FakeLLM([judgment("R001", "insufficient")] * 3)
        matcher = make_matcher(llm, retriever, top_k=3)
        result = matcher.match_requirement(
            make_requirement(queries=("q1", "q2")))
        self.assertEqual(len(result["candidates"]), 3)
        self.assertLessEqual(len(llm.calls), 1)

    def judge_fits(self, llm, budget, requirement, entries):
        return llm.estimate_context("judge_evidence", JUDGMENT_SYSTEM_PROMPT,
                                    {"requirement": _requirement_payload(requirement),
                                     "candidates": entries},
                                    JUDGMENT_SCHEMA) <= budget

    def test_long_section_split_into_numbered_sub_candidates(self):
        long_text = "训练与评估记录。\n" * 100  # 900 chars
        parent = {**SECTION_A, "text": long_text}
        retriever = FakeRetriever({"标注": [parent]})
        requirement = make_requirement()
        budget = len(JUDGMENT_SYSTEM_PROMPT) + 1400
        llm = FakeLLM([], prompt_char_budget=budget)
        base = llm.estimate_context("judge_evidence", JUDGMENT_SYSTEM_PROMPT,
                                    {"requirement": _requirement_payload(requirement),
                                     "candidates": []},
                                    JUDGMENT_SCHEMA)
        limit = max(200, budget - base) * 4 // 5
        pieces = _split_text(long_text, limit)
        self.assertGreater(len(pieces), 1)
        self.assertEqual("".join(pieces), long_text)
        entries = [{"candidate_id": f"E{i:03d}", "project": "p", "section": "s",
                    "text": piece, "evidence_lines": []} for i, piece
                   in enumerate(pieces, 1)]
        # Preconditions: unsplit does not fit, each piece does.
        unsplit = {"candidate_id": "E001", "project": "p", "section": "s",
                   "text": long_text, "evidence_lines": []}
        self.assertFalse(self.judge_fits(llm, budget, requirement, [unsplit]))
        self.assertTrue(self.judge_fits(llm, budget, requirement, [entries[0]]))
        if len(entries) == 2:
            self.assertFalse(self.judge_fits(llm, budget, requirement, entries))
        responses = [judgment("R001", "insufficient")] * (len(pieces) - 1)
        responses.append(judgment(
            "R001", "direct",
            quotes=[(f"E{len(pieces):03d}", pieces[-1].splitlines()[0])]))
        llm = FakeLLM(responses, prompt_char_budget=budget)
        result = make_matcher(llm, retriever).match_requirement(requirement)
        self.assertEqual(len(llm.calls), len(pieces))
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "direct")
        self.assertEqual(len(result["candidates"]), len(pieces))
        for index, candidate in enumerate(result["candidates"], 1):
            self.assertEqual(candidate["candidate_id"], f"E{index:03d}")
            self.assertEqual(candidate["source"], "synthetic_project.md")
            self.assertEqual(candidate["note"],
                             f"长章节摘录 {index}/{len(pieces)}")
        evidence = result["evidence"][0]
        self.assertEqual(evidence["candidate_id"], f"E{len(pieces):03d}")
        self.assertEqual(evidence["source"], "synthetic_project.md")
        self.assertEqual(evidence["line_start"], parent["line_start"])

    def test_multibatch_consolidation_takes_strongest_verdict(self):
        big = "参数搜索结果。\n" * 80
        s1 = {**SECTION_A, "text": big, "section_id": "s1"}
        s2 = {**SECTION_B, "text": big, "section_id": "s2", "score": 0.5}
        retriever = FakeRetriever({"标注": [s1, s2]})
        requirement = make_requirement()
        budget = len(JUDGMENT_SYSTEM_PROMPT) + 1400
        llm = FakeLLM([], prompt_char_budget=budget)
        entry = {"candidate_id": "E001", "project": "p", "section": "s",
                 "text": big, "evidence_lines": []}
        self.assertTrue(self.judge_fits(llm, budget, requirement, [entry]))
        self.assertFalse(self.judge_fits(llm, budget, requirement, [entry, entry]))
        llm = FakeLLM([
            judgment("R001", "related", quotes=[("E001", "参数搜索结果。")]),
            judgment("R001", "direct", quotes=[("E002", "参数搜索结果。")]),
        ], prompt_char_budget=budget)
        result = make_matcher(llm, retriever).match_requirement(requirement)
        self.assertGreaterEqual(len(llm.calls), 2)
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "direct")
        self.assertEqual({e["candidate_id"] for e in result["evidence"]},
                         {"E001", "E002"})

    def test_failed_batch_marks_incomplete_not_insufficient(self):
        big = "参数搜索结果。\n" * 80
        s1 = {**SECTION_A, "text": big, "section_id": "s1"}
        s2 = {**SECTION_B, "text": big, "section_id": "s2", "score": 0.5}
        retriever = FakeRetriever({"标注": [s1, s2]})
        requirement = make_requirement()
        budget = len(JUDGMENT_SYSTEM_PROMPT) + 1400
        llm = FakeLLM([
            judgment("R001", "related", quotes=[("E001", "参数搜索结果。")]),
            LLMError("超时", error_type="timeout"),
        ], prompt_char_budget=budget)
        result = make_matcher(llm, retriever).match_requirement(requirement)
        self.assertEqual(result["processing_status"], "incomplete")
        self.assertEqual(result["verdict"], "related")
        self.assertTrue(result["errors"])

    def test_single_batch_direct_missing_repaired_to_related(self):
        retriever = FakeRetriever({"标注": [SECTION_A], "图片标注": [SECTION_A]})
        quote = "我负责示例地图图片标注。"
        llm = FakeLLM([
            judgment("R001", "direct", quotes=[("E001", quote)],
                     missing=["生产环境部署"]),
            judgment("R001", "related", quotes=[("E001", quote)],
                     missing=["生产环境部署"]),
        ])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("direct 与未覆盖条件矛盾",
                      json.dumps(llm.calls[1], ensure_ascii=False))
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "related")

    def test_persistent_direct_missing_contradiction_is_error(self):
        retriever = FakeRetriever({"标注": [SECTION_A], "图片标注": [SECTION_A]})
        quote = "我负责示例地图图片标注。"
        llm = FakeLLM([
            judgment("R001", "direct", quotes=[("E001", quote)],
                     missing=["生产环境部署"]),
            judgment("R001", "direct", quotes=[("E001", quote)],
                     missing=["生产环境部署"]),
        ])
        result = make_matcher(llm, retriever).match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertIsNone(result["verdict"])
        self.assertTrue(any("矛盾" in e for e in result["errors"]))

    def test_multibatch_direct_with_missing_flagged_for_review(self):
        big = "参数搜索结果。\n" * 80
        s1 = {**SECTION_A, "text": big, "section_id": "s1"}
        s2 = {**SECTION_B, "text": big, "section_id": "s2", "score": 0.5}
        retriever = FakeRetriever({"标注": [s1, s2]})
        requirement = make_requirement()
        budget = len(JUDGMENT_SYSTEM_PROMPT) + 1400
        llm = FakeLLM([
            judgment("R001", "direct", quotes=[("E001", "参数搜索结果。")]),
            judgment("R001", "related", quotes=[("E002", "参数搜索结果。")],
                     missing=["生产环境部署"]),
        ], prompt_char_budget=budget)
        result = make_matcher(llm, retriever).match_requirement(requirement)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result["processing_status"], "ok")
        self.assertEqual(result["verdict"], "direct")
        self.assertTrue(result["needs_review"])
        self.assertTrue(any("多批判断合并" in n for n in result["review_notes"]))
        self.assertEqual(result["missing_aspects"], ["生产环境部署"])

    def test_candidate_from_another_batch_is_not_quotable(self):
        """Batch 2 may only quote what batch 2 was shown."""
        big = "参数搜索结果。\n" * 80
        s1 = {**SECTION_A, "text": big, "section_id": "s1"}
        s2 = {**SECTION_B, "text": big, "section_id": "s2", "score": 0.5}
        retriever = FakeRetriever({"标注": [s1, s2]})
        requirement = make_requirement()
        budget = len(JUDGMENT_SYSTEM_PROMPT) + 1400
        llm = FakeLLM([
            judgment("R001", "insufficient"),
            # E001 belongs to the first batch, so this request must not use it.
            judgment("R001", "direct", quotes=[("E001", "参数搜索结果。")]),
            judgment("R001", "direct", quotes=[("E001", "参数搜索结果。")]),
            judgment("R001", "direct", quotes=[("E001", "参数搜索结果。")]),
        ], prompt_char_budget=budget)
        result = make_matcher(llm, retriever).match_requirement(requirement)
        self.assertEqual(result["processing_status"], "incomplete")
        self.assertEqual(result["verdict"], "insufficient")
        self.assertTrue(any("不属于当前候选集" in e for e in result["errors"]))
        self.assertEqual(result["evidence"], [])

    def test_quote_is_rechecked_against_the_full_parent_section(self):
        """Last gate: the batch slice is not the authority, the section is."""
        retriever = FakeRetriever({"标注": [SECTION_A]})
        quote = "本句不在章节原文中。"
        llm = FakeLLM([judgment("R001", "direct", quotes=[("E001", quote)])] * 3)
        matcher = make_matcher(llm, retriever)
        # Show the model a slice that is NOT part of the original section.
        with unittest.mock.patch.object(evidence_matcher, "_split_text",
                                        return_value=[quote]):
            result = matcher.match_requirement(make_requirement())
        self.assertEqual(result["processing_status"], "error")
        self.assertIsNone(result["verdict"])
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("已丢弃该引用" in e for e in result["errors"]))

    def test_match_all_covers_every_requirement(self):
        retriever = FakeRetriever({"标注": [SECTION_A]})
        llm = FakeLLM([judgment("R001", "direct",
                                quotes=[("E001", "我负责示例地图图片标注。")]),
                       judgment("R002", "insufficient")])
        requirements = [make_requirement("R001"), make_requirement("R002")]
        results = make_matcher(llm, retriever).match_all(requirements)
        self.assertEqual([r["requirement_id"] for r in results], ["R001", "R002"])
        self.assertEqual([r["processing_status"] for r in results], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()


class ProfileCapTests(unittest.TestCase):
    """profile.md sections are capped so project evidence stays in the judgment."""

    @staticmethod
    def profile_hit(index, score):
        return {**SECTION_A, "project": "个人基本信息（自述事实）", "section": f"事实 {index}",
                "source": "profile.md", "section_id": f"p{index}", "chunk_id": f"cp{index}",
                "score": score, "text": f"自述事实 {index}", "evidence": [], "search_keywords": ""}

    def test_select_candidates_caps_profile_sections(self):
        from evidence_matcher import select_candidates
        ranked = [self.profile_hit(i, 0.9 - i * 0.01) for i in range(4)] + [
            {**SECTION_A, "score": 0.5}, {**SECTION_B, "score": 0.4}]
        chosen = select_candidates(ranked, top_k=5, max_profile=2)
        self.assertEqual([c["section_id"] for c in chosen], ["p0", "p1", "sA", "sB"])
        self.assertEqual(select_candidates(ranked, top_k=1, max_profile=2)[0]["section_id"], "p0")

    def test_windows_path_source_is_recognised(self):
        from evidence_matcher import is_profile_source
        self.assertTrue(is_profile_source({"source": "sub\\profile.md"}))
        self.assertFalse(is_profile_source({"source": "geomind.md"}))

    def test_matcher_overfetches_and_keeps_project_sections(self):
        hits = [self.profile_hit(i, 0.9 - i * 0.01) for i in range(5)] + [
            {**SECTION_A, "score": 0.6}]
        retriever = FakeRetriever({"标注": hits})
        llm = FakeLLM([{"judgments": [{"requirement_id": "R001", "verdict": "insufficient",
                                        "reason": "无", "evidence": [], "missing_aspects": [],
                                        "questions_for_user": []}]}])
        matcher = make_matcher(llm, retriever, top_k=5)
        matcher.match_requirement(make_requirement(queries=("标注",)))
        self.assertGreater(retriever.searches[0][1], 5)
        sent = llm.calls[0]["candidates"]
        sources = [c["project"] for c in sent]
        self.assertEqual(sources.count("个人基本信息（自述事实）"), 2)
        self.assertIn("Synthetic Project", sources)
