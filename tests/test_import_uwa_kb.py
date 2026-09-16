import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import import_uwa_kb
from search_experience import read_sections, select_sections

FACT = {"text": "我训练了模型，测试 accuracy 0.5117。", "evidence_type": "notebook_output",
        "sources": [{"source_id": "S0001", "locator": "cell 3 output 1", "source_path": "x/a.ipynb"}]}

PROJECTS = [
    {"project_id": "UWA-T01", "title_zh": "个人项目", "course": "CITS9999", "period": "2025 S2",
     "status": "已有输出", "ownership": "个人项目，notebook 署名 Test Author", "record_type": "project",
     "tags": ["CNN", "Keras"], "grade": None, "artifacts": ["x/a.ipynb"],
     "sections": [{"section": "方法与实现", "facts": [FACT]},
                  {"section": "版本与限制", "facts": [{"text": "未重新训练。", "evidence_type": "documented",
                                                      "sources": [{"source_id": "S0001", "locator": "cell 9", "source_path": "x/a.ipynb"}]}]}]},
    {"project_id": "UWA-T02", "title_zh": "团队项目", "course": "CITS8888", "period": None,
     "status": "已有报告", "ownership": "Test Author 与 A 两人项目，有明确分工", "record_type": "project",
     "tags": [], "grade": None, "artifacts": [],
     "sections": [{"section": "项目目标与数据", "facts": [FACT]},
                  {"section": "个人贡献", "facts": [{"text": "我负责 BiLSTM。", "evidence_type": "report_record",
                                                    "sources": [{"source_id": "S0002", "locator": "lines 1–2", "source_path": "y/r.tex"}]}]}]},
]


def make_kb(tmp):
    kb = Path(tmp) / "kb"
    (kb / "evidence").mkdir(parents=True)
    (kb / "projects.json").write_text(json.dumps(PROJECTS, ensure_ascii=False), encoding="utf-8")
    for sid in ("S0001", "S0002"):
        (kb / "evidence" / f"{sid}.txt").write_text(f"[cell]\n{sid} text", encoding="utf-8")
        (kb / "evidence" / f"{sid}.json").write_text("{}", encoding="utf-8")
    (kb / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    return kb


class ClassifyTests(unittest.TestCase):
    def test_ownership_and_sections(self):
        self.assertTrue(import_uwa_kb.is_personal_project(PROJECTS[0]))
        self.assertFalse(import_uwa_kb.is_personal_project(PROJECTS[1]))
        self.assertEqual(import_uwa_kb.classify("方法与实现", True), "personal")
        self.assertEqual(import_uwa_kb.classify("方法与实现", False), "team")
        self.assertEqual(import_uwa_kb.classify("个人贡献与限制", False), "personal")
        self.assertEqual(import_uwa_kb.classify("版本与限制", True), "boundary")


class ImportTests(unittest.TestCase):
    def test_import_writes_files_and_copies_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = make_kb(tmp)
            out = Path(tmp) / "experiences"
            out.mkdir()
            original_sources = import_uwa_kb.SOURCES
            import_uwa_kb.SOURCES = Path(tmp) / "_sources" / "uwa_kb"
            try:
                kb_dir, projects = import_uwa_kb.load_kb(kb)
                written, sids = import_uwa_kb.import_projects(kb_dir, projects, ["UWA-T01", "UWA-T02"], out_dir=out)
            finally:
                import_uwa_kb.SOURCES = original_sources
            self.assertEqual(sids, ["S0001", "S0002"])
            self.assertTrue((Path(tmp) / "_sources" / "uwa_kb" / "evidence" / "S0001.txt").exists())
            self.assertTrue((Path(tmp) / "_sources" / "uwa_kb" / "metrics.jsonl").exists())
            personal = written[0].read_text(encoding="utf-8")
            self.assertIn("## 我的工作 方法与实现", personal)
            self.assertIn("证据：[S0001: cell 3 output 1](../_sources/uwa_kb/evidence/S0001.txt)", personal)
            self.assertIn("## 证据边界\n未重新训练。", personal)
            self.assertIn("待写", personal)  # no curated resume line for the fake project
            team = written[1].read_text(encoding="utf-8")
            self.assertIn("## 项目内容与团队成果 项目目标与数据", team)
            self.assertIn("## 我的贡献 个人贡献", team)
            self.assertIn("时间：待确认", team)
            # The retriever must pick only the personal section of the team file by default.
            records = select_sections(read_sections(out))
            team_sections = [r["section"] for r in records if r["source"] == written[1].name]
            self.assertEqual(team_sections, ["我的贡献 个人贡献"])
            personal_sections = [r["section"] for r in records if r["source"] == written[0].name]
            self.assertEqual(personal_sections, ["我的工作 方法与实现"])


if __name__ == "__main__":
    unittest.main()
