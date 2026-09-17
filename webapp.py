"""Local web page: paste a JD, get a tailored resume, cover letter and
interview prep.

    python webapp.py            # http://127.0.0.1:5000

The page only orchestrates the existing command-line tools (match_job,
tailor_cv, build_resume, export_docx, interview_prep) as subprocesses; no
generation logic lives here. One job runs at a time because the local model
server has a single slot. Binds to 127.0.0.1 only: the run directories hold
personal material and the model API key stays in config.local.json.
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

import add_jd
import build_site
from local_llm import load_config
from tailor_cv import latest_output

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_RESUME = PROJECT_ROOT / "resume" / "base_resume.docx"
DEFAULT_CONFIG = PROJECT_ROOT / "config.local.json"

STAGES = [("save", "保存 JD"), ("match", "提取与匹配"), ("tailor", "CV 建议与 Cover Letter"),
          ("build", "组装简历"), ("docx", "导出 Word"), ("interview", "面试准备")]
DOCS = [("report", "匹配报告"), ("cv_suggestions", "CV 建议"),
        ("resume_tailored", "完整简历"), ("interview_prep", "面试准备")]


def run_command(args, cwd):
    """Default stage runner: (returncode, stdout, stderr)."""
    completed = subprocess.run([PYTHON, "-X", "utf8", *args], cwd=cwd, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
    return completed.returncode, completed.stdout, completed.stderr


class LocationBlocked(RuntimeError):
    pass


class Job:
    def __init__(self, jd_name, with_interview):
        self.id = uuid.uuid4().hex[:8]
        self.jd_name = jd_name
        self.with_interview = with_interview
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.stages = [{"key": k, "label": l, "status": "pending", "seconds": None, "message": ""}
                       for k, l in STAGES]
        if not with_interview:
            self.stages[-1]["status"] = "skipped"
        self.run_dir = None
        self.status = "queued"
        self.error = None

    def stage(self, key):
        return next(s for s in self.stages if s["key"] == key)

    def to_dict(self):
        return {"id": self.id, "jd_name": self.jd_name, "created_at": self.created_at,
                "status": self.status, "error": self.error,
                "run_id": Path(self.run_dir).name if self.run_dir else None,
                "stages": self.stages}


class Pipeline:
    """Runs the stages for one job; `runner` is injectable for tests."""

    def __init__(self, project_root=PROJECT_ROOT, jobs_dir=None, outputs_dir=None,
                 resume_path=DEFAULT_RESUME, config_path=DEFAULT_CONFIG, runner=run_command):
        self.root = Path(project_root)
        self.jobs_dir = Path(jobs_dir) if jobs_dir else self.root / "jobs"
        self.outputs_dir = Path(outputs_dir) if outputs_dir else self.root / "outputs"
        self.resume_path = Path(resume_path)
        self.config_path = Path(config_path)
        self.runner = runner
        self.jobs = {}
        self.lock = threading.Lock()
        self.current = None

    # -- model server -----------------------------------------------------
    def model_online(self):
        try:
            config = load_config(self.config_path)
            base = config["base_url"].rstrip("/")
            url = base + ("/models" if base.endswith("/v1") else "/v1/models")
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {config.get('api_key', '')}"})
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status == 200, config.get("model")
        except Exception as exc:  # noqa: BLE001 - any failure means "offline" for the page
            return False, str(exc)

    # -- job control ------------------------------------------------------
    def start(self, jd_text, jd_name, with_interview):
        with self.lock:
            if self.current is not None and self.current.status == "running":
                raise RuntimeError("已有任务在运行，请等待它完成")
            job = Job(jd_name or "", with_interview)
            self.jobs[job.id] = job
            self.current = job
        thread = threading.Thread(target=self._run, args=(job, jd_text), daemon=True)
        thread.start()
        return job

    def _step(self, job, key, func):
        stage = job.stage(key)
        stage["status"] = "running"
        started = time.time()
        try:
            message = func()
            stage["status"] = "done"
            stage["message"] = message or ""
        except Exception as exc:  # noqa: BLE001 - reported on the page, not raised
            stage["status"] = "failed"
            stage["message"] = str(exc)
            raise
        finally:
            stage["seconds"] = round(time.time() - started, 1)

    def _cli(self, script, *args, ok_codes=(0,)):
        code, stdout, stderr = self.runner([str(self.root / script), *args], str(self.root))
        if code not in ok_codes:
            tail = "\n".join((stderr or stdout).strip().splitlines()[-3:])
            raise RuntimeError(f"{script} 退出码 {code}：{tail}")
        return code, stdout, stderr

    @staticmethod
    def _partial_match_message(run_dir):
        """A partial match (some requirements unprocessed) continues: downstream
        tools list unprocessed items as gaps. A failed run, or one without
        matches, stops with the recorded errors rather than a stderr tail."""
        run_dir = Path(run_dir)
        try:
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        errors = "；".join(e.get("message", "")[:160] for e in meta.get("errors", [])[:3])
        if meta.get("run_status") != "partial" or not (run_dir / "matches.json").exists():
            raise RuntimeError(f"匹配失败（{meta.get('run_status', '状态未知')}）：{errors or '见 run_meta.json'}")
        matches = json.loads((run_dir / "matches.json").read_text(encoding="utf-8")).get("matches", [])
        unprocessed = [m["requirement_id"] for m in matches if m.get("processing_status") != "ok"]
        return (f"partial：{len(matches)} 条要求中 {len(unprocessed)} 条未处理"
                f"（{'、'.join(unprocessed[:6])}），已继续；详见匹配报告")

    def _run(self, job, jd_text):
        job.status = "running"
        try:
            def save():
                path = add_jd.save_jd(jd_text, self.jobs_dir, job.jd_name or None)
                job.jd_path = path
                return path.name
            self._step(job, "save", save)

            def match():
                # Exit 1 covers both "failed" and "partial"; only the run metadata tells
                # them apart, so accept it here and decide below.
                code, stdout, stderr = self._cli("match_job.py", "--jd", str(job.jd_path), "--config",
                                                 str(self.config_path), "--offline", "--out",
                                                 str(self.outputs_dir), ok_codes=(0, 1, 3))
                found = re.search(r"运行目录：(.+)", stdout)
                if not found:
                    tail = "\n".join((stderr or stdout).strip().splitlines()[-3:])
                    raise RuntimeError(f"match_job 未报告运行目录（退出码 {code}）：{tail}")
                job.run_dir = Path(found.group(1).strip())
                if code == 3:
                    screen = json.loads((job.run_dir / "location_screen.json").read_text(encoding="utf-8"))
                    raise LocationBlocked(screen["message"] + "\n地点依据：" + screen["location_quote"]
                                          + "\n出勤依据：" + screen["attendance_quote"])
                if code == 1:
                    return self._partial_match_message(job.run_dir)
                status = re.search(r"状态：(\S+)", stdout)
                return status.group(1) if status else ""
            self._step(job, "match", match)

            def tailor():
                code, stdout, _ = self._cli("tailor_cv.py", "--run", str(job.run_dir), "--resume",
                                            str(self.resume_path), "--config", str(self.config_path),
                                            ok_codes=(0, 1))
                return "partial（部分步骤失败，见 CV 建议末尾）" if code == 1 else ""
            self._step(job, "tailor", tailor)

            def build():
                self._cli("build_resume.py", "--run", str(job.run_dir), "--resume",
                          str(self.resume_path), "--config", str(self.config_path))
                return ""
            self._step(job, "build", build)

            def docx():
                _, stdout, _ = self._cli("export_docx.py", "--run", str(job.run_dir), "--letter")
                return f"{stdout.count('已写入')} 个文件"
            self._step(job, "docx", docx)

            if job.with_interview:
                def interview():
                    code, _, _ = self._cli("interview_prep.py", "--run", str(job.run_dir), "--config",
                                           str(self.config_path), ok_codes=(0, 1))
                    return "partial" if code == 1 else ""
                self._step(job, "interview", interview)
            job.status = "done"
        except LocationBlocked as exc:
            job.status = "blocked"
            job.error = str(exc)
            job.stage("match")["status"] = "blocked"
            for stage in job.stages:
                if stage["status"] == "pending":
                    stage["status"] = "skipped"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)


# ---------------------------------------------------------------- Flask app


def run_documents(run_dir):
    """Latest Markdown per document kind plus downloadable files (only the
    newest .docx of each kind, so stale exports do not show)."""
    run_dir = Path(run_dir)
    docs = []
    for stem, label in DOCS:
        path = latest_output(run_dir, stem)
        if path is not None:
            docs.append({"stem": stem, "label": label, "file": path.name})
    downloads = []
    resume_docx = latest_output(run_dir, "resume_tailored", ".docx")
    if resume_docx is not None:
        downloads.append(resume_docx.name)
    if (run_dir / "cover_letter.docx").exists():
        downloads.append("cover_letter.docx")
    downloads += [d["file"] for d in docs]
    return docs, downloads


def verdict_counts(run_dir):
    """direct/related/insufficient/error counts from matches.json, or {}."""
    path = Path(run_dir) / "matches.json"
    if not path.exists():
        return {}
    try:
        matches = json.loads(path.read_text(encoding="utf-8"))["matches"]
    except (OSError, ValueError, KeyError):
        return {}
    counts = {"direct": 0, "related": 0, "insufficient": 0, "error": 0}
    for m in matches:
        key = m.get("verdict") if m.get("processing_status") == "ok" else "error"
        counts[key if key in counts else "error"] += 1
    return counts


def create_app(pipeline=None):
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "webui"))
    app.config["PIPELINE"] = pipeline or Pipeline()

    @app.get("/")
    def index():
        pipe = app.config["PIPELINE"]
        online, model = pipe.model_online()
        runs = build_site.scan_runs(pipe.outputs_dir)
        history = []
        for run in runs[:20]:
            docs, _ = run_documents(run["dir"])
            history.append({"id": run["id"], "jd": Path(str(run["meta"].get("input", {}).get("jd_path", ""))).name,
                            "started": run["meta"].get("started_at", ""), "status": run["meta"].get("run_status"),
                            "counts": run["counts"], "docs": docs})
        return render_template("index.html", online=online, model=model, history=history,
                               resume=str(pipe.resume_path), stages=STAGES)

    @app.post("/api/run")
    def api_run():
        pipe = app.config["PIPELINE"]
        data = request.get_json(silent=True) or request.form
        jd_text = (data.get("jd_text") or "").strip()
        if len(jd_text) < add_jd.MIN_CHARS:
            return jsonify({"error": f"JD 文本太短（少于 {add_jd.MIN_CHARS} 字符）"}), 400
        online, model = pipe.model_online()
        if not online:
            return jsonify({"error": f"本机模型服务不在线：{model}"}), 503
        try:
            job = pipe.start(jd_text, (data.get("jd_name") or "").strip(),
                             str(data.get("with_interview", "1")).lower() in ("1", "true", "on", "yes"))
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(job.to_dict()), 202

    @app.get("/api/status/<job_id>")
    def api_status(job_id):
        job = app.config["PIPELINE"].jobs.get(job_id)
        if job is None:
            abort(404)
        payload = job.to_dict()
        if job.run_dir:
            docs, downloads = run_documents(job.run_dir)
            payload["docs"] = docs
            payload["downloads"] = downloads
            payload["counts"] = verdict_counts(job.run_dir)
        return jsonify(payload)

    @app.get("/api/current")
    def api_current():
        """The most recent job, so a reloaded page can pick up where it was."""
        job = app.config["PIPELINE"].current
        if job is None:
            return jsonify({"job": None})
        payload = job.to_dict()
        if job.run_dir:
            docs, downloads = run_documents(job.run_dir)
            payload["docs"], payload["downloads"] = docs, downloads
            payload["counts"] = verdict_counts(job.run_dir)
        return jsonify({"job": payload})

    @app.get("/api/health")
    def api_health():
        online, model = app.config["PIPELINE"].model_online()
        return jsonify({"online": online, "model": model})

    def _run_dir(run_id):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", run_id):
            abort(404)
        run_dir = app.config["PIPELINE"].outputs_dir / run_id
        if not run_dir.is_dir():
            abort(404)
        return run_dir

    @app.get("/run/<run_id>/<name>.html")
    def view_doc(run_id, name):
        run_dir = _run_dir(run_id)
        path = run_dir / f"{name}.md"
        if not re.fullmatch(r"[A-Za-z_]+(-[0-9]+)?", name) or not path.exists():
            abort(404)
        docs, downloads = run_documents(run_dir)
        nav = ['<a href="/">← 首页</a>'] + [
            (f'<span class="here">{d["label"]}</span>' if d["file"] == path.name
             else f'<a href="/run/{run_id}/{Path(d["file"]).stem}.html">{d["label"]}</a>') for d in docs]
        nav += [f'<a href="/run/{run_id}/download/{f}">⬇ {f}</a>' for f in downloads if f.endswith(".docx")]
        body = build_site.md_to_html(path.read_text(encoding="utf-8"))
        return build_site.page(f"{name} · {run_id}", body, " ".join(nav))

    @app.get("/run/<run_id>/download/<name>")
    def download(run_id, name):
        run_dir = _run_dir(run_id)
        if not re.fullmatch(r"[A-Za-z_]+(-[0-9]+)?\.(md|docx|json)", name):
            abort(404)
        path = run_dir / name
        if not path.exists():
            abort(404)
        return send_file(path, as_attachment=True)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="本机网页：粘贴 JD 生成简历")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    app = create_app(Pipeline(resume_path=args.resume, config_path=args.config))
    print(f"打开 http://127.0.0.1:{args.port}", file=sys.stderr)
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
