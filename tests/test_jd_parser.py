import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jd_parser import (AmbiguousQuoteError, EXTRACTION_SCHEMA,
                        EXTRACTION_SYSTEM_PROMPT, NOTE_TEMPLATE, batch_note,
                        extract_requirements, line_of_offset, locate_quote,
                        normalize_jd, plan_batches, read_jd, span_to_lines)
from local_llm import LLMError, validate_against_schema


def max_note():
    return NOTE_TEMPLATE.format(index=999, count=999)


def budget_just_below(llm, jd_text):
    """A budget where the whole JD (with note + heading reserve) does not fit."""
    return llm.estimate_context("extract_requirements", EXTRACTION_SYSTEM_PROMPT,
                                {"jd_text": jd_text + " " * 64, "note": max_note()},
                                EXTRACTION_SCHEMA) - 1

JD = (
    "Senior ML Engineer\n"
    "\n"
    "Requirements\n"
    "\n"
    "- Object detection experience is required.\n"
    "- Python or Java experience is required.\n"
    "- 3 years of production Python experience is required.\n"
    "- AWS experience is desirable.\n"
    "\n"
    "Nice to have: PostgreSQL backend.\n"
    "\n"
    "Object detection experience is required.\n"
)


class FakeLLM:
    def __init__(self, responses, prompt_char_budget=100000):
        if isinstance(responses, dict):
            responses = [responses]
        self.responses = list(responses)
        self.prompt_char_budget = prompt_char_budget
        self.calls = []

    def estimate_context(self, task, system_prompt, payload, schema):
        return len(system_prompt) + len(json.dumps(payload, ensure_ascii=False)) + 200

    def generate_json(self, task, system_prompt, payload, schema=None,
                      timeout_seconds=None, validate=None):
        """Mimics BaseAdapter: validate() shares the three-attempt budget."""
        last = None
        for _attempt in range(3):
            self.calls.append(payload)
            if not self.responses:
                break
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            errors = list(validate(response)) if validate is not None else []
            if not errors:
                return response
            last = errors
            self.fix_notes = getattr(self, "fix_notes", []) + [errors]
        raise LLMError("；".join(last or ["no response"]), error_type="structure")

    def describe_model(self):
        return {"name": "fake", "version": None}


class NormalizeTests(unittest.TestCase):
    def test_normalize_bom_and_crlf(self):
        self.assertEqual(normalize_jd("\ufeffTitle\r\nLine one\r\nLine two"),
                         "Title\nLine one\nLine two")
        self.assertEqual(normalize_jd("plain\ntext"), "plain\ntext")

    def test_read_jd_hash_and_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jd.txt"
            path.write_bytes("第一行\r\n第二行".encode("utf-8"))
            jd = read_jd(path)
            self.assertEqual(jd["text"], "第一行\n第二行")
            self.assertEqual(jd["raw_sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())

    def test_read_jd_error_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.txt"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "为空"):
                read_jd(empty)
            binary = Path(tmp) / "bin.txt"
            binary.write_bytes(b"\xff\xfe\x00bad")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                read_jd(binary)
            whitespace = Path(tmp) / "ws.txt"
            whitespace.write_text("  \n\r\n ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "空白"):
                read_jd(whitespace)


class LineOffsetTests(unittest.TestCase):
    def test_line_offsets_and_span_lines(self):
        text = "a\nbb\nccc\n"
        self.assertEqual(line_of_offset(text, 0), 1)
        self.assertEqual(line_of_offset(text, 2), 2)
        self.assertEqual(line_of_offset(text, len(text)), 4)
        self.assertEqual(span_to_lines(text, 0, 2), (1, 1))
        self.assertEqual(span_to_lines(text, 0, 5), (1, 2))


class LocateQuoteTests(unittest.TestCase):
    def test_unique_quote_located_exactly(self):
        start, end, mode = locate_quote(JD, "AWS experience is desirable.")
        self.assertEqual(mode, "exact")
        self.assertEqual(JD[start:end], "AWS experience is desirable.")

    def test_missing_quote_returns_none(self):
        self.assertIsNone(locate_quote(JD, "Kubernetes experience is required."))

    def test_duplicate_quote_resolved_by_context_not_first(self):
        quote = "Object detection experience is required."
        context = "PostgreSQL backend.\n\n" + quote
        start, end, mode = locate_quote(JD, quote, context)
        self.assertEqual(mode, "exact")
        self.assertGreater(start, JD.index(quote))
        self.assertEqual(JD[start:end], quote)

    def test_duplicate_quote_without_context_is_ambiguous(self):
        with self.assertRaises(AmbiguousQuoteError):
            locate_quote(JD, "Object detection experience is required.")

    def test_whitespace_fallback_maps_back_to_original(self):
        text = "We use Python  or Java here.\n"
        start, end, mode = locate_quote(text, "Python or Java here")
        self.assertEqual(mode, "whitespace")
        self.assertEqual(text[start:end], "Python  or Java here")


class PlanBatchesTests(unittest.TestCase):
    def test_single_batch_when_it_fits(self):
        llm = FakeLLM({}, prompt_char_budget=100000)
        batches = plan_batches(JD, llm)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["text"], JD)
        self.assertEqual((batches[0]["start"], batches[0]["end"]), (0, len(JD)))

    def test_long_text_splits_at_boundaries_without_dropping_chars(self):
        long_text = "\n\n".join(f"Paragraph {i} " + "x" * 40 for i in range(30))
        budget = len(EXTRACTION_SYSTEM_PROMPT) + 800
        llm = FakeLLM({}, prompt_char_budget=budget)
        batches = plan_batches(long_text, llm)
        self.assertGreater(len(batches), 1)
        covered = []
        for batch in batches:
            self.assertEqual(batch["text"], long_text[batch["start"]:batch["end"]])
            covered.append((batch["start"], batch["end"]))
            self.assertLessEqual(
                llm.estimate_context("extract_requirements", EXTRACTION_SYSTEM_PROMPT,
                                     {"jd_text": batch["text"]}, EXTRACTION_SCHEMA),
                budget)
        for (s1, e1), (s2, e2) in zip(covered, covered[1:]):
            self.assertLessEqual(e1, s2)
        for index, char in enumerate(long_text):
            if not char.isspace():
                self.assertTrue(any(s <= index < e for s, e in covered),
                                f"char {index} dropped by batching")

    def test_oversized_sentence_is_hard_cut_without_dropping_chars(self):
        # Review repro: 10000 chars without punctuation must not stay a single
        # over-budget batch.
        long_text = "x" * 10000
        probe = FakeLLM({}, prompt_char_budget=100000)
        llm = FakeLLM({}, prompt_char_budget=budget_just_below(probe, long_text))
        batches = plan_batches(long_text, llm)
        self.assertGreater(len(batches), 1)
        self.assertTrue(any(batch["hard_cut"] for batch in batches))
        covered = []
        for batch in batches:
            self.assertEqual(batch["text"], long_text[batch["start"]:batch["end"]])
            covered.append((batch["start"], batch["end"]))
            payload = {"jd_text": batch["text"],
                       "note": batch_note(batch["index"], batch["count"])}
            self.assertLessEqual(
                llm.estimate_context("extract_requirements",
                                     EXTRACTION_SYSTEM_PROMPT, payload,
                                     EXTRACTION_SCHEMA), llm.prompt_char_budget)
        self.assertEqual(covered[0][0], 0)
        self.assertEqual(covered[-1][1], len(long_text))
        for index in range(len(long_text)):
            self.assertTrue(any(s <= index < e for s, e in covered),
                            f"char {index} dropped by batching")

    def test_batches_fit_including_the_multi_batch_note(self):
        long_text = "\n\n".join(f"Paragraph {i} " + "x" * 40 for i in range(30))
        probe = FakeLLM({}, prompt_char_budget=100000)
        llm = FakeLLM({}, prompt_char_budget=budget_just_below(probe, long_text))
        batches = plan_batches(long_text, llm)
        self.assertGreater(len(batches), 1)
        for batch in batches:
            payload = {"jd_text": batch["text"],
                       "note": batch_note(batch["index"], batch["count"])}
            self.assertLessEqual(
                llm.estimate_context("extract_requirements",
                                     EXTRACTION_SYSTEM_PROMPT, payload,
                                     EXTRACTION_SCHEMA), llm.prompt_char_budget)

    def test_mid_section_batches_keep_heading_context(self):
        jd = ("Requirements\n\n" +
              "\n\n".join(f"Item {i} " + "x" * 60 for i in range(6)))
        probe = FakeLLM({}, prompt_char_budget=100000)
        llm = FakeLLM({}, prompt_char_budget=budget_just_below(probe, jd))
        batches = plan_batches(jd, llm)
        self.assertGreater(len(batches), 1)
        for batch in batches[1:]:
            self.assertEqual(batch["heading_context"], "Requirements")
            self.assertEqual(
                batch["text"],
                "Requirements\n" + jd[batch["start"]:batch["end"]])

    def test_template_over_budget_raises_value_error(self):
        probe = FakeLLM({}, prompt_char_budget=100000)
        tiny = probe.estimate_context(
            "extract_requirements", EXTRACTION_SYSTEM_PROMPT,
            {"jd_text": " " * 64, "note": max_note()}, EXTRACTION_SCHEMA)
        llm = FakeLLM({}, prompt_char_budget=tiny - 1)
        with self.assertRaisesRegex(ValueError, "context_budget"):
            plan_batches("x" * 100, llm)

    def test_schema_allows_empty_requirements(self):
        errors = validate_against_schema(
            {"requirements": [], "requirement_groups": []}, EXTRACTION_SCHEMA)
        self.assertEqual(errors, [])


class ExtractRequirementsTests(unittest.TestCase):
    def response(self):
        return {
            "requirements": [
                {
                    "text": "Object detection experience",
                    "category": "technical_skill",
                    "importance": "required",
                    "source_quote": "Object detection experience is required.",
                    "source_context":
                        "PostgreSQL backend.\n\nObject detection experience is required.",
                    "importance_source_quote":
                        "Object detection experience is required.",
                    "qualifiers": [],
                    "search_queries": ["object detection model training"],
                    "needs_review": False,
                },
                {
                    "text": "Python experience",
                    "category": "technical_skill",
                    "importance": "required",
                    "source_quote": "Python or Java experience is required.",
                    "importance_source_quote":
                        "Python or Java experience is required.",
                    "qualifiers": [],
                    "search_queries": ["python development"],
                    "needs_review": False,
                },
                {
                    "text": "Java experience",
                    "category": "technical_skill",
                    "importance": "required",
                    "source_quote": "Python or Java experience is required.",
                    "importance_source_quote":
                        "Python or Java experience is required.",
                    "qualifiers": [],
                    "search_queries": ["java development"],
                    "needs_review": False,
                },
                {
                    "text": "Three years of production Python",
                    "category": "experience",
                    "importance": "required",
                    "source_quote":
                        "3 years of production Python experience is required.",
                    "importance_source_quote":
                        "3 years of production Python experience is required.",
                    "qualifiers": [
                        {"type": "years", "value": "3", "source_quote": "3 years"},
                        {"type": "environment", "value": "production",
                         "source_quote": "production Python"},
                    ],
                    "search_queries": ["production python three years"],
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
                {
                    "text": "Some invented requirement",
                    "category": "other",
                    "importance": "unspecified",
                    "source_quote": "This sentence never appears verbatim.",
                    "importance_source_quote": "This sentence never appears verbatim.",
                    "qualifiers": [],
                    "search_queries": ["invented"],
                    "needs_review": False,
                },
            ],
            "requirement_groups": [
                {"operator": "OR",
                 "requirement_texts": ["Python experience", "Java experience"],
                 "source_quote": "Python or Java experience is required."},
                {"operator": "AND",
                 "requirement_texts": ["Python experience", "Ghost requirement"],
                 "source_quote": "not in the jd"},
            ],
        }

    def test_ids_spans_and_groups(self):
        llm = FakeLLM(self.response())
        doc = extract_requirements(JD, llm)
        requirements = doc["requirements"]
        self.assertEqual([r["requirement_id"] for r in requirements],
                         [f"R{i:03d}" for i in range(1, 7)])
        first = requirements[0]
        self.assertFalse(first["needs_review"])
        quote = "Object detection experience is required."
        span = first["source_span"]
        self.assertGreater(span["start"], JD.index(quote))
        self.assertEqual(JD[span["start"]:span["end"]], quote)
        expected_line = JD[:span["start"]].count("\n") + 1
        self.assertEqual(span["line_start"], expected_line)
        self.assertEqual(span["line_end"], expected_line)
        years = requirements[3]["qualifiers"][0]
        self.assertEqual(
            JD[years["source_span"]["start"]:years["source_span"]["end"]], "3 years")
        invented = requirements[5]
        self.assertTrue(invented["needs_review"])
        self.assertIsNone(invented["source_span"])
        self.assertTrue(invented["review_notes"])
        self.assertEqual(len(doc["requirement_groups"]), 2)
        group = doc["requirement_groups"][0]
        self.assertEqual(group["group_id"], "G001")
        self.assertEqual(group["operator"], "OR")
        self.assertEqual(group["requirement_ids"], ["R002", "R003"])
        self.assertEqual(group["missing_texts"], [])
        self.assertTrue(group["complete"])
        self.assertIsNotNone(group["source_span"])
        incomplete = doc["requirement_groups"][1]
        self.assertEqual(incomplete["group_id"], "G002")
        self.assertEqual(incomplete["operator"], "AND")
        self.assertEqual(incomplete["requirement_ids"], ["R002"])
        self.assertEqual(incomplete["missing_texts"], ["Ghost requirement"])
        self.assertFalse(incomplete["complete"])
        self.assertTrue(any("引用了未解析的要求 text" in w
                            for w in doc["warnings"]))
        self.assertTrue(any(w.startswith("组 G002") for w in doc["warnings"]))
        self.assertTrue(any("逐字原文" in w for w in doc["warnings"]))
        self.assertEqual(doc["batches"], 1)

    def test_duplicate_requirement_merged(self):
        data = self.response()
        data["requirements"].append(dict(data["requirements"][4]))
        llm = FakeLLM(data)
        doc = extract_requirements(JD, llm)
        aws = [r for r in doc["requirements"] if r["text"] == "AWS experience"]
        self.assertEqual(len(aws), 1)
        self.assertTrue(any("重复要求已合并" in w for w in doc["warnings"]))

    def test_multibatch_merge_keeps_ids_sequential(self):
        long_jd = "\n\n".join(f"Batch line {i} with unique quote Q{i}."
                              for i in range(20))
        budget = len(EXTRACTION_SYSTEM_PROMPT) + 500
        responses = [{
            "requirements": [{
                "text": f"Requirement {i}",
                "category": "other",
                "importance": "unspecified",
                "source_quote": f"unique quote Q{i}.",
                "importance_source_quote": f"unique quote Q{i}.",
                "qualifiers": [],
                "search_queries": [f"q{i}"],
                "needs_review": False,
            }],
            "requirement_groups": [],
        } for i in range(20)]
        llm = FakeLLM(responses, prompt_char_budget=budget)
        doc = extract_requirements(long_jd, llm)
        self.assertGreater(doc["batches"], 1)
        self.assertEqual(len(llm.calls), doc["batches"])
        self.assertEqual([r["requirement_id"] for r in doc["requirements"]],
                         [f"R{i:03d}" for i in range(1, len(doc["requirements"]) + 1)])
        for requirement in doc["requirements"]:
            span = requirement["source_span"]
            self.assertEqual(
                long_jd[span["start"]:span["end"]], requirement["source_quote"])

    def test_zero_requirement_batch_allowed_and_failure_keeps_earlier(self):
        long_jd = "\n\n".join(
            f"Paragraph {i} text with token TK{i} required here.\n"
            for i in range(8))
        probe = FakeLLM({}, prompt_char_budget=100000)
        budget = budget_just_below(probe, long_jd)
        batch_count = len(plan_batches(long_jd, FakeLLM({}, prompt_char_budget=budget)))
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
        llm = FakeLLM(
            [{"requirements": [requirement_item], "requirement_groups": []}]
            + [empty] * (batch_count - 2)
            + [LLMError("超时", error_type="timeout")],
            prompt_char_budget=budget)
        doc = extract_requirements(long_jd, llm)
        self.assertEqual(len(llm.calls), batch_count)
        self.assertEqual(len(doc["requirements"]), 1)
        first = doc["requirements"][0]
        self.assertEqual(first["requirement_id"], "R001")
        span = first["source_span"]
        self.assertEqual(long_jd[span["start"]:span["end"]],
                         "token TK0 required here.")
        self.assertEqual(len(doc["failed_batches"]), 1)
        failed = doc["failed_batches"][0]
        self.assertEqual(failed["index"], batch_count)
        self.assertEqual(failed["error_type"], "timeout")
        self.assertGreater(failed["line_start"], 1)
        self.assertLessEqual(failed["line_start"], failed["line_end"])
        self.assertLess(failed["line_end"],
                        long_jd.count("\n") + 1)


if __name__ == "__main__":
    unittest.main()


class ResponsibilityCoverageTests(unittest.TestCase):
    JD_WITH_DUTIES = (
        "IT Coordinator\n\nKey Responsibilities\n\n"
        "Assist with data cleanup and data entry.\n"
        "Maintain IT Asset Management system.\n\n"
        "About You\n\nCertificate or Diploma in IT\n")

    @staticmethod
    def item(text, quote, category):
        return {"text": text, "category": category, "importance": "unspecified",
                "source_quote": quote, "importance_source_quote": quote,
                "qualifiers": [], "search_queries": [text], "needs_review": False}

    def test_heading_detection(self):
        from jd_parser import responsibility_headings, make_batch_validator
        self.assertEqual(responsibility_headings(self.JD_WITH_DUTIES), ["Key Responsibilities"])
        self.assertEqual(responsibility_headings("What you'll do\n- x"), ["What you'll do"])
        self.assertEqual(responsibility_headings("岗位职责：\n- x"), ["岗位职责："])
        self.assertEqual(responsibility_headings(JD), [])
        self.assertIsNone(make_batch_validator(JD))

    def test_missing_responsibilities_triggers_retry_then_succeeds(self):
        skipped = {"requirements": [self.item("Certificate or Diploma in IT",
                                              "Certificate or Diploma in IT", "education")],
                   "requirement_groups": []}
        fixed = {"requirements": skipped["requirements"] + [
            self.item("Assist with data cleanup and data entry",
                      "Assist with data cleanup and data entry.", "responsibility")],
            "requirement_groups": []}
        llm = FakeLLM([skipped, fixed])
        doc = extract_requirements(self.JD_WITH_DUTIES, llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("responsibility", llm.fix_notes[0][0])
        self.assertEqual([r["category"] for r in doc["requirements"]],
                         ["education", "responsibility"])
        self.assertEqual(doc["failed_batches"], [])

    def test_no_duties_heading_means_no_extra_check(self):
        llm = FakeLLM([{"requirements": [self.item("Object detection experience",
                                                   "Object detection experience is required.",
                                                   "experience")],
                        "requirement_groups": []}])
        doc = extract_requirements(JD, llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(doc["requirements"]), 1)
