import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from location_gate import screen_location
import match_job
import webapp

JD = "Location: Sydney. Attendance in the Sydney office is mandatory three days a week."
REQ = [{"category": "location", "needs_review": False}]
BLOCK = {"location": "outside_wa", "attendance": "required", "wa_option": False,
         "location_quote": "Location: Sydney.", "attendance_quote": "Attendance in the Sydney office is mandatory three days a week."}


class LocationGateTests(unittest.TestCase):
    def test_grounded_mandatory_interstate_hybrid_blocks(self):
        llm = Mock()
        llm.generate_json.return_value = BLOCK
        self.assertEqual(screen_location(JD, REQ, llm)["status"], "blocked")
        self.assertEqual(llm.generate_json.call_args.args[2]["jd_text"], JD)

    def test_wa_remote_options_optional_and_unknown_do_not_block(self):
        for change in ({"location": "wa"}, {"wa_option": True}, {"attendance": "optional"},
                       {"attendance": "none"}, {"location": "unknown"}):
            with self.subTest(change=change):
                llm = Mock()
                llm.generate_json.return_value = {**BLOCK, **change}
                self.assertNotEqual(screen_location(JD, REQ, llm)["status"], "blocked")

    def test_invented_or_empty_quotes_stop_processing(self):
        for quote in ("", "Invented location requirement"):
            llm = Mock()
            llm.generate_json.return_value = {**BLOCK, "location_quote": quote}
            with self.assertRaises(ValueError):
                screen_location(JD, REQ, llm)

    def test_no_location_is_unresolved_not_an_eligibility_claim(self):
        llm = Mock()
        self.assertEqual(screen_location("Python required", [], llm)["status"], "review")
        llm.generate_json.assert_not_called()

    def test_cli_gate_stops_retrieval_and_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jd = root / "input.txt"
            jd.write_text(JD)
            run = root / "run"
            run.mkdir()
            args = match_job.build_parser().parse_args(["--jd", str(jd), "--config", "unused", "--data", str(root)])
            llm = Mock()
            llm.describe_model.return_value = {}
            llm.describe_parameters.return_value = {}
            llm.generate_json.return_value = BLOCK
            doc = {"requirements": REQ, "requirement_groups": [], "batches": [], "failed_batches": [], "warnings": []}
            with patch.object(match_job, "load_config", return_value={}), patch.object(match_job, "build_llm", return_value=llm), patch.object(match_job.jd_parser, "extract_requirements", return_value=doc), patch.object(match_job, "ExperienceRetriever") as retriever, patch.object(match_job, "generate_summary") as summary:
                code, result = match_job.execute(args, jd, root / "config", run, lambda _: None)
            self.assertEqual(code, 3)
            self.assertEqual(result["run_status"], "blocked")
            retriever.assert_not_called()
            summary.assert_not_called()
            self.assertFalse((run / "matches.json").exists())
            self.assertEqual(json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["run_status"], "blocked")

    def test_web_pipeline_skips_all_downstream_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "outputs" / "run"
            run.mkdir(parents=True)
            (run / "location_screen.json").write_text(json.dumps({**BLOCK, "message": "Location blocked"}))
            runner = Mock(return_value=(3, f"运行目录：{run}\n状态：blocked\n", ""))
            pipeline = webapp.Pipeline(project_root=root, runner=runner)
            job = webapp.Job("example", True)
            pipeline._run(job, JD * 3)
            self.assertEqual(job.status, "blocked")
            self.assertEqual(runner.call_count, 1)
            self.assertTrue(all(s["status"] == "skipped" for s in job.stages[2:]))
            self.assertIn(BLOCK["location_quote"], job.error)
