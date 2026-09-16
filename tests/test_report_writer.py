import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report_writer


def make_match(requirement_id="R001", status="ok", verdict="direct",
               needs_review=False, review_notes=None, evidence=None,
               missing=None, questions=None, errors=None, reason="理由"):
    return {
        "requirement_id": requirement_id,
        "text": "Python experience",
        "category": "technical_skill",
        "importance": "required",
        "source_quote": "Python experience is required.",
        "source_span": {"line_start": 5, "line_end": 5},
        "importance_source_quote": "Python experience is required.",
        "importance_source_span": {"line_start": 5, "line_end": 5},
        "qualifiers": [],
        "search_queries": ["python"],
        "needs_review": needs_review,
        "review_notes": review_notes or [],
        "candidates": [{
            "candidate_id": "E001", "project": "Synthetic Project", "section": "我的贡献",
            "source": "synthetic_project.md", "line_start": 22, "line_end": 27,
            "section_id": "ab12", "chunk_id": "cd34", "score": 0.8,
            "matched_by_query": "python", "evidence": [], "note": None,
        }],
        "processing_status": status,
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence or [],
        "missing_aspects": missing or [],
        "questions_for_user": questions or [],
        "errors": errors or [],
    }


def make_meta(run_status="complete", scope="full", warnings=None):
    return {
        "input": {"jd_path": "jobs/sample.txt",
                  "jd_sha256": "a" * 64},
        "scope": scope,
        "run_status": run_status,
        "warnings": warnings or [],
    }


class OverallStatusTests(unittest.TestCase):
    def test_all_ok_is_complete(self):
        matches = [make_match("R001"), make_match("R002", verdict="insufficient")]
        self.assertEqual(report_writer.overall_status(matches), "complete")

    def test_mixed_is_partial(self):
        matches = [make_match("R001"),
                   make_match("R002", status="error", verdict=None)]
        self.assertEqual(report_writer.overall_status(matches), "partial")

    def test_all_failed_is_failed(self):
        matches = [make_match("R001", status="error", verdict=None)]
        self.assertEqual(report_writer.overall_status(matches), "failed")

    def test_empty_is_failed(self):
        self.assertEqual(report_writer.overall_status([]), "failed")


class VerdictLabelTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(report_writer.verdict_label(make_match(verdict="direct")),
                         "直接支持")
        self.assertEqual(report_writer.verdict_label(make_match(verdict="related")),
                         "相关但不足")
        self.assertEqual(
            report_writer.verdict_label(make_match(verdict="insufficient")),
            "当前资料证据不足")

    def test_error_status_label(self):
        match = make_match(status="error", verdict=None)
        self.assertEqual(report_writer.verdict_label(match), "处理错误（无判断）")

    def test_incomplete_status_label(self):
        match = make_match(status="incomplete", verdict="related")
        self.assertEqual(report_writer.verdict_label(match), "未完成（相关但不足）")


class WriteReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_report_contains_required_parts(self):
        evidence = [{
            "candidate_id": "E001",
            "quote": "使用 Python 编写标注工具",
            "project": "Synthetic Project",
            "section": "我的贡献",
            "source": "synthetic_project.md",
            "line_start": 22,
            "line_end": 27,
            "section_id": "ab12",
            "chunk_id": "cd34",
            "evidence_lines": ["- 来源：项目笔记 2024-03"],
        }]
        matches = [
            make_match("R001", verdict="direct", evidence=evidence,
                       missing=["生产环境经验"],
                       questions=["相关经验是否用于生产环境？"]),
            make_match("R002", verdict="insufficient",
                       needs_review=True,
                       review_notes=["优先级由措辞推断"]),
            make_match("R003", status="error", verdict=None,
                       errors=["模型超时"]),
        ]
        meta = make_meta(run_status="partial")
        report_writer.write_report(self.run_dir, meta, matches)
        report = (self.run_dir / "report.md").read_text(encoding="utf-8")
        for part in ("# JD 匹配报告", "## 要求概览", "## 逐项详情",
                     "## 待补充资料 / 待确认问题"):
            self.assertIn(part, report)
        self.assertIn("待复核 1 条", report)
        self.assertIn("未完成/出错 1 条", report)
        self.assertIn("R001", report)
        self.assertIn("使用 Python 编写标注工具", report)
        self.assertIn("synthetic_project.md 章节行 22-27", report)
        self.assertIn("已有证据行（既有来源记录，本次未重新核查）", report)
        self.assertIn("处理错误（无判断）", report)
        self.assertIn("待复核**：优先级由措辞推断", report)
        self.assertIn("仅用于检索排序", report)
        self.assertIn("[R001] 待确认：相关经验是否用于生产环境？", report)
        self.assertIn("[R001] 未覆盖：生产环境经验", report)

    def test_report_lists_warnings(self):
        meta = make_meta(warnings=["第 3 段过长，已按段落拆分"])
        report_writer.write_report(self.run_dir, meta, [make_match()])
        report = (self.run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("警告：第 3 段过长，已按段落拆分", report)

    def test_report_renders_requirement_groups(self):
        second = make_match("R002", verdict="insufficient")
        second["text"] = "Java experience"
        groups = [
            {"group_id": "G001", "operator": "OR",
             "requirement_ids": ["R001", "R002"],
             "missing_texts": [], "complete": True,
             "source_quote": "Python or Java experience is required.",
             "source_span": {"line_start": 9, "line_end": 9,
                             "start": 0, "end": 10, "match": "exact"}},
            {"group_id": "G002", "operator": "AND",
             "requirement_ids": ["R001"],
             "missing_texts": ["Ghost requirement"],
             "complete": False,
             "source_quote": "not in the jd",
             "source_span": None},
        ]
        report_writer.write_report(self.run_dir, make_meta(),
                                   [make_match("R001"), second], groups)
        report = (self.run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("## AND/OR 关系", report)
        self.assertIn("G001 二选一：R001 Python experience；R002 Java experience",
                      report)
        self.assertIn("位置：JD 第 9-9 行", report)
        self.assertIn("G002 同时需要：R001 Python experience", report)
        self.assertIn("组引用了未解析的要求 text：Ghost requirement", report)
        self.assertIn("位置：原文定位待复核", report)
        self.assertIn("| 组关系 |", report)
        self.assertIn("G001 二选一、G002 同时需要", report)
        self.assertIn("不代表每条要求本身成立", report)

    def test_report_without_groups_uses_dash(self):
        report_writer.write_report(self.run_dir, make_meta(), [make_match()])
        report = (self.run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 组关系 |", report)
        self.assertNotIn("## AND/OR 关系", report)
        self.assertIn("| R001 | Python experience | 必需 | 直接支持 | Synthetic Project | — |",
                      report)


class WriteMatchesTests(unittest.TestCase):
    def test_matches_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            matches = [make_match()]
            report_writer.write_matches(run_dir, "1.0", matches)
            doc = json.loads((run_dir / "matches.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], "1.0")
            self.assertEqual(doc["matches"], matches)


if __name__ == "__main__":
    unittest.main()
