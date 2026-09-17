import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_matcher import repair_quote, validate_judgment
from interview_prep import validate_answer
from jd_parser import _finalize_requirement
from tailor_cv import jd_name_number, validate_basis

SECTION = ("## 我的贡献\n\n- 我使用 **Docker Compose** 部署了 `FastAPI` 后端，并编写了 problem-solving 文档。\n"
           "- I built a Retrieval-Augmented Generation pipeline, tested on 1,288 samples.\n")


class RepairQuoteTests(unittest.TestCase):
    def test_exact_quote_is_returned_unchanged(self):
        self.assertEqual(repair_quote("部署了 `FastAPI` 后端", SECTION), "部署了 `FastAPI` 后端")

    def test_markdown_case_and_punctuation_differences_map_back_to_source(self):
        repaired = repair_quote("我使用 docker compose 部署了 FastAPI 后端", SECTION)
        self.assertEqual(repaired, "我使用 **Docker Compose** 部署了 `FastAPI` 后端")
        self.assertIn(repaired, SECTION)
        self.assertEqual(repair_quote("retrieval augmented generation pipeline", SECTION),
                         "Retrieval-Augmented Generation pipeline")

    def test_changed_words_and_trivial_quotes_are_refused(self):
        self.assertIsNone(repair_quote("我使用 Kubernetes 部署了 FastAPI 后端", SECTION))
        self.assertIsNone(repair_quote("i BUILT", SECTION))  # too short for a loose match
        self.assertIsNone(repair_quote("   ", SECTION))

    def test_validate_judgment_stores_the_verbatim_text(self):
        result = {"requirement_id": "R001", "verdict": "direct", "reason": "x", "missing_aspects": [],
                  "evidence": [{"candidate_id": "E001", "quote": "built a retrieval augmented generation pipeline"}]}
        errors, valid = validate_judgment(result, "R001", {"E001": SECTION})
        self.assertEqual(errors, [])
        self.assertEqual(valid, [{"candidate_id": "E001",
                                  "quote": "built a Retrieval-Augmented Generation pipeline"}])


JD = ("About you\n"
      "- Basic understanding of networking concepts.\n"
      "- Basic knowledge of SQL is desirable.\n"
      "- Strong communication skills.\n")


def requirement(quote, qualifier_quote):
    return {"text": "x", "category": "skill", "importance": "required", "source_quote": quote,
            "source_context": "", "importance_source_quote": "About you",
            "qualifiers": [{"type": "proficiency", "value": "basic", "source_quote": qualifier_quote}],
            "search_queries": ["x"]}


class QualifierSpanTests(unittest.TestCase):
    def test_repeated_qualifier_word_is_located_inside_its_own_requirement(self):
        notes = []
        item = _finalize_requirement(requirement("Basic knowledge of SQL is desirable.", "Basic"), 2, JD, notes)
        span = item["qualifiers"][0]["source_span"]
        self.assertIsNotNone(span)
        self.assertEqual(JD[span["start"]:span["end"]], "Basic")
        self.assertEqual(span["start"], JD.index("Basic knowledge"))
        self.assertFalse(item["needs_review"], notes)

    def test_qualifier_missing_from_the_jd_still_needs_review(self):
        notes = []
        item = _finalize_requirement(requirement("Strong communication skills.", "Excellent"), 3, JD, notes)
        self.assertIsNone(item["qualifiers"][0]["source_span"])
        self.assertTrue(item["needs_review"])


MATERIALS = {"M001": {"text": "我熟练使用 Excel 和 Office 办公软件。", "scope": "personal"}}
JD_365 = "Support Microsoft 365 users. Minimum 5 years of experience. Knowledge of ISO 27001."


class JdNameNumberTests(unittest.TestCase):
    def test_only_names_from_the_jd_qualify(self):
        self.assertTrue(jd_name_number("365", "The role supports Microsoft 365 users.", JD_365))
        self.assertTrue(jd_name_number("27001", "I would study iso 27001 first.", JD_365))
        self.assertFalse(jd_name_number("365", "I resolved 365 tickets.", JD_365))
        self.assertFalse(jd_name_number("5", "I have Minimum 5 years of experience.", JD_365))
        self.assertFalse(jd_name_number("365", "Microsoft 365", None))

    def test_interview_answers_may_name_the_product_but_resume_checks_stay_strict(self):
        point = {"part": "Situation", "text": "The role supports Microsoft 365 users.",
                 "basis": [{"material_id": "M001", "quote": "我熟练使用 Excel 和 Office 办公软件"}]}
        self.assertEqual(validate_answer([point], MATERIALS, ["M001"], JD_365), [])
        honest = {"part": "Honest", "text": "I have not administered Microsoft 365 tenants yet.", "basis": []}
        self.assertEqual(validate_answer([honest], MATERIALS, ["M001"], JD_365), [])
        invented = {"part": "Honest", "text": "I closed 365 tickets.", "basis": []}
        self.assertTrue(validate_answer([invented], MATERIALS, ["M001"], JD_365))
        strict = validate_basis([point], MATERIALS, ["M001"], jd_text=JD_365)
        self.assertTrue(any("365" in e for e in strict))


if __name__ == "__main__":
    unittest.main()
