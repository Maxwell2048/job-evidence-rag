import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval


class PublicEvalTests(unittest.TestCase):
    def test_missing_set_and_invalid_top_k_fail_before_model_loading(self):
        for args in (["--set", "missing-evaluation-file.json"], ["--top-k", "0"]):
            with patch.object(run_eval, "ExperienceRetriever") as retriever, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                run_eval.main(args)
            self.assertEqual(exc.exception.code, 2)
            retriever.assert_not_called()

    def test_default_public_set_and_cli_options(self):
        evaluation = json.loads(Path(__file__).with_name("public_eval_set.json").read_text(encoding="utf-8"))
        jd = Path(__file__).with_name(evaluation["extraction_jd"]).read_text(encoding="utf-8")
        for case in evaluation["cases"]:
            self.assertIn(case["jd_line"], jd)
        with patch.object(run_eval, "ExperienceRetriever") as cls, contextlib.redirect_stdout(io.StringIO()):
            retriever = cls.return_value.build.return_value
            retriever.search.return_value = [{"source": "demo_project.md", "section": "后端开发 SQL 与数据库 团队协作", "score": 1.0}]
            self.assertEqual(run_eval.main(["--offline", "--top-k", "2"]), 0)
            cls.assert_called_once_with(run_eval.ROOT / "examples" / "experiences", offline=True)
            self.assertTrue(all(c.kwargs["top_k"] == 2 for c in retriever.search.call_args_list))

    def test_missing_extraction_cannot_pass_judgment(self):
        evaluation = {"cases": [{"id": "synthetic", "text": "Flask", "search_query": "Flask", "jd_line": "Flask", "expected_verdict": "direct"}]}
        with patch("evidence_matcher.EvidenceMatcher") as cls:
            cls.return_value.match_requirement.return_value = {"processing_status": "ok", "verdict": "direct"}
            rows = run_eval.run_judgment(evaluation, None, None, 5, {"requirements": []}, "Flask")
        self.assertTrue(rows[0]["synthetic"])
        self.assertFalse(rows[0]["ok"])
