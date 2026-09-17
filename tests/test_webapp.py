import json
import tempfile
import time
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import webapp

JD = "Senior Data Engineer\n\nRequired\n" + "Experience with SQL and Python. " * 6


class FakeRunner:
    """Simulates the CLI tools by writing the files each stage would produce."""

    def __init__(self, outputs_dir, fail_at=None, match_status="complete"):
        self.outputs_dir = Path(outputs_dir)
        self.fail_at = fail_at
        self.match_status = match_status
        self.calls = []

    def __call__(self, args, cwd):
        script = Path(args[0]).name
        self.calls.append(script)
        if script == self.fail_at:
            return 1, "", "boom\nsecond line"
        if script == "match_job.py":
            run = self.outputs_dir / "20260912-000000-abcdef"
            run.mkdir(parents=True, exist_ok=True)
            errors = [] if self.match_status == "complete" else [
                {"error_type": "judgment_failed", "message": "R002（error）：引用不是章节原文"}]
            (run / "run_meta.json").write_text(json.dumps({
                "run_id": run.name, "run_status": self.match_status, "scope": "full", "errors": errors,
                "started_at": "2026-09-12T00:00:00", "input": {"jd_path": "jobs/x.txt"}}), encoding="utf-8")
            if self.match_status != "failed":
                matches = [{"requirement_id": "R001", "processing_status": "ok", "verdict": "direct"}]
                if self.match_status == "partial":
                    matches.append({"requirement_id": "R002", "processing_status": "error",
                                    "verdict": "insufficient"})
                (run / "matches.json").write_text(json.dumps({"matches": matches}), encoding="utf-8")
                (run / "report.md").write_text("# 报告\n", encoding="utf-8")
            code = 0 if self.match_status == "complete" else 1
            # The stderr tail deliberately looks like an unrelated retry log, as in the real failure.
            return code, f"运行目录：{run}\n状态：{self.match_status}（范围：full）\n", \
                "[local-llm] job_readiness_summary 第 1 次尝试未通过校验"
        run = self.outputs_dir / "20260912-000000-abcdef"
        if script == "tailor_cv.py":
            (run / "cv_suggestions.md").write_text("# CV 建议\n", encoding="utf-8")
            return 1, "状态：partial", ""
        if script == "build_resume.py":
            (run / "resume_tailored.md").write_text("# ALEX SAMPLE\n", encoding="utf-8")
            return 0, "已写入", ""
        if script == "export_docx.py":
            (run / "resume_tailored.docx").write_bytes(b"PK")
            return 0, "已写入（resume）：x\n已写入（cover_letter）：y\n", ""
        if script == "interview_prep.py":
            (run / "interview_prep.md").write_text("# 面试准备\n", encoding="utf-8")
            return 0, "", ""
        return 0, "", ""


class OnlinePipeline(webapp.Pipeline):
    def model_online(self):
        return True, "fake-model"


class OfflinePipeline(webapp.Pipeline):
    def model_online(self):
        return False, "connection refused"


def wait_for(job, timeout=5):
    deadline = time.time() + timeout
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.05)
    return job


class PipelineTests(unittest.TestCase):
    def test_full_job_runs_all_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            runner = FakeRunner(outputs)
            pipe = OnlinePipeline(project_root=Path(tmp), jobs_dir=Path(tmp) / "jobs", outputs_dir=outputs,
                                  resume_path=Path(tmp) / "r.docx", config_path=Path(tmp) / "c.json", runner=runner)
            job = wait_for(pipe.start(JD, "acme", True))
            self.assertEqual(job.status, "done", job.error)
            self.assertEqual([s["status"] for s in job.stages], ["done"] * 6)
            self.assertIn("partial", job.stage("tailor")["message"])
            self.assertEqual(job.stage("docx")["message"], "2 个文件")
            self.assertEqual(runner.calls, ["match_job.py", "tailor_cv.py", "build_resume.py",
                                            "export_docx.py", "interview_prep.py"])
            self.assertTrue(list((Path(tmp) / "jobs").glob("*-acme.txt")))
            docs, downloads = webapp.run_documents(job.run_dir)
            self.assertEqual([d["stem"] for d in docs], ["report", "cv_suggestions", "resume_tailored", "interview_prep"])
            self.assertIn("resume_tailored.docx", downloads)

    def test_partial_match_continues_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            runner = FakeRunner(outputs, match_status="partial")
            pipe = OnlinePipeline(project_root=Path(tmp), jobs_dir=Path(tmp) / "jobs", outputs_dir=outputs,
                                  runner=runner)
            job = wait_for(pipe.start(JD, "", False))
            self.assertEqual(job.status, "done", job.error)
            message = job.stage("match")["message"]
            self.assertIn("partial：2 条要求中 1 条未处理（R002）", message)
            self.assertIn("build_resume.py", runner.calls)  # the pipeline went on

    def test_failed_match_reports_recorded_errors_not_stderr_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            runner = FakeRunner(outputs, match_status="failed")
            pipe = OnlinePipeline(project_root=Path(tmp), jobs_dir=Path(tmp) / "jobs", outputs_dir=outputs,
                                  runner=runner)
            job = wait_for(pipe.start(JD, "", False))
            self.assertEqual(job.status, "failed")
            self.assertIn("匹配失败（failed）", job.error)
            self.assertIn("引用不是章节原文", job.error)
            self.assertNotIn("job_readiness_summary", job.error)
            self.assertEqual(runner.calls, ["match_job.py"])

    def test_failure_marks_stage_and_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            pipe = OnlinePipeline(project_root=Path(tmp), jobs_dir=Path(tmp) / "jobs", outputs_dir=outputs,
                                  runner=FakeRunner(outputs, fail_at="build_resume.py"))
            job = wait_for(pipe.start(JD, "", False))
            self.assertEqual(job.status, "failed")
            self.assertEqual(job.stage("build")["status"], "failed")
            self.assertIn("build_resume.py 退出码 1", job.error)
            self.assertEqual(job.stage("docx")["status"], "pending")
            self.assertEqual(job.stage("interview")["status"], "skipped")


class AppTests(unittest.TestCase):
    def make(self, pipeline_cls, tmp):
        outputs = Path(tmp) / "outputs"
        outputs.mkdir()
        pipe = pipeline_cls(project_root=Path(tmp), jobs_dir=Path(tmp) / "jobs", outputs_dir=outputs,
                            runner=FakeRunner(outputs))
        app = webapp.create_app(pipe)
        app.config["TESTING"] = True
        return app.test_client(), pipe

    def test_index_health_and_run_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, pipe = self.make(OnlinePipeline, tmp)
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("在线".encode("utf-8"), page.data)
            self.assertEqual(client.get("/api/health").get_json()["online"], True)
            self.assertIsNone(client.get("/api/current").get_json()["job"])
            short = client.post("/api/run", json={"jd_text": "too short"})
            self.assertEqual(short.status_code, 400)
            started = client.post("/api/run", json={"jd_text": JD, "jd_name": "acme", "with_interview": False})
            self.assertEqual(started.status_code, 202)
            job_id = started.get_json()["id"]
            wait_for(pipe.jobs[job_id])
            status = client.get(f"/api/status/{job_id}").get_json()
            self.assertEqual(status["status"], "done")
            self.assertEqual(status["run_id"], "20260912-000000-abcdef")
            self.assertIn("resume_tailored.docx", status["downloads"])
            self.assertEqual(status["counts"], {"direct": 1, "related": 0, "insufficient": 0, "error": 0})
            current = client.get("/api/current").get_json()["job"]
            self.assertEqual(current["id"], job_id)
            self.assertEqual(current["status"], "done")
            self.assertIn("resume_tailored.docx", current["downloads"])
            view = client.get("/run/20260912-000000-abcdef/resume_tailored.html")
            self.assertEqual(view.status_code, 200)
            self.assertIn(b"<h1>ALEX SAMPLE</h1>", view.data)
            self.assertIn(b"resume_tailored.docx", view.data)
            dl = client.get("/run/20260912-000000-abcdef/download/resume_tailored.docx")
            self.assertEqual(dl.status_code, 200)
            dl.close()  # release the file handle so the temp dir can be removed on Windows
            self.assertEqual(client.get("/run/20260912-000000-abcdef/download/../secret").status_code, 404)
            self.assertEqual(client.get("/run/nope/report.html").status_code, 404)
            self.assertEqual(client.get("/api/status/zzz").status_code, 404)

    def test_offline_model_blocks_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = self.make(OfflinePipeline, tmp)
            self.assertIn("离线".encode("utf-8"), client.get("/").data)
            resp = client.post("/api/run", json={"jd_text": JD})
            self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()
