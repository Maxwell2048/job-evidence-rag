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
import os
import re
import shutil
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
from export_docx import latest_docx
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


def run_command(args, cwd, job=None):
    """Default stage runner: (returncode, stdout, stderr). The process is
    registered on the job so that a cancel request can stop it."""
    process = subprocess.Popen([PYTHON, "-X", "utf8", *args], cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if job is not None:
        job.process = process
    try:
        stdout, stderr = process.communicate()
    finally:
        if job is not None:
            job.process = None
    return process.returncode, stdout, stderr


def kill_tree(process):
    """Stop a stage and its children; on Windows the venv python.exe is a
    launcher whose real interpreter is a child process."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        process.kill()


class LocationBlocked(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class Job:
    """kind "full" runs every stage for a pasted JD; kind "interview" adds
    interview prep to a run that already exists."""

    def __init__(self, jd_name, with_interview, kind="full", run_dir=None):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.jd_name = jd_name
        self.with_interview = with_interview
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.started_stamp = time.strftime("%Y%m%d-%H%M%S")  # same format as run directory names
        self.stages = [{"key": k, "label": l, "status": "pending", "seconds": None, "message": ""}
                       for k, l in STAGES]
        for stage in self.stages:
            is_interview = stage["key"] == "interview"
            if (kind == "interview" and not is_interview) or (kind == "full" and is_interview and not with_interview):
                stage["status"] = "skipped"
        self.run_dir = Path(run_dir) if run_dir else None
        self.jd_path = None
        self.status = "queued"
        self.error = None
        self.process = None
        self.cancel_requested = False

    def stage(self, key):
        return next(s for s in self.stages if s["key"] == key)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "jd_name": self.jd_name, "created_at": self.created_at,
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
    def _claim(self, job):
        with self.lock:
            if self.current is not None and self.current.status in ("queued", "running"):
                raise RuntimeError("已有任务在运行，请等待它完成或先中止")
            self.jobs[job.id] = job
            self.current = job

    def start(self, jd_text, jd_name, with_interview=False):
        job = Job(jd_name or "", with_interview)
        self._claim(job)
        threading.Thread(target=self._run, args=(job, jd_text), daemon=True).start()
        return job

    def start_interview(self, run_dir):
        """Interview prep for a finished run, requested later from the page."""
        job = Job(Path(run_dir).name, True, kind="interview", run_dir=run_dir)
        self._claim(job)
        threading.Thread(target=self._run_interview, args=(job,), daemon=True).start()
        return job

    def cancel(self, job):
        """Stop the running stage; the job thread then removes what it created."""
        if job.status not in ("queued", "running"):
            return False
        job.cancel_requested = True
        kill_tree(job.process)
        return True

    def _discard_unfinished(self, job):
        """A cancelled full job leaves nothing behind: its run directory and the
        JD file it saved. A cancelled interview job only stops; the finished
        run it was adding to is kept."""
        removed = []
        if job.kind != "full":
            return removed
        outputs = self.outputs_dir.resolve()
        if job.run_dir and Path(job.run_dir).resolve().parent == outputs and Path(job.run_dir).is_dir():
            shutil.rmtree(job.run_dir, ignore_errors=True)
            removed.append(Path(job.run_dir).name)
        if job.jd_path and Path(job.jd_path).resolve().parent == self.jobs_dir.resolve() \
                and Path(job.jd_path).is_file():
            Path(job.jd_path).unlink()
            removed.append(Path(job.jd_path).name)
        return removed

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

    def _cli(self, script, *args, ok_codes=(0,), job=None):
        if job is not None and job.cancel_requested:
            raise Cancelled()
        code, stdout, stderr = self.runner([str(self.root / script), *args], str(self.root), job=job)
        if job is not None and job.cancel_requested:
            raise Cancelled()
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
                                                 str(self.outputs_dir), ok_codes=(0, 1, 3), job=job)
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
                                            ok_codes=(0, 1), job=job)
                return "partial（部分步骤失败，见 CV 建议末尾）" if code == 1 else ""
            self._step(job, "tailor", tailor)

            def build():
                self._cli("build_resume.py", "--run", str(job.run_dir), "--resume",
                          str(self.resume_path), "--config", str(self.config_path), job=job)
                return ""
            self._step(job, "build", build)

            def docx():
                # Company and job title for the file names; a failure only means plainer names.
                code, identity, _ = self._cli("job_identity.py", "--run", str(job.run_dir), "--config",
                                              str(self.config_path), ok_codes=(0, 1), job=job)
                label = ["--label", job.jd_name] if job.jd_name else []
                _, stdout, _ = self._cli("export_docx.py", "--run", str(job.run_dir), "--letter", *label, job=job)
                named = identity.strip().splitlines()[-1] if code == 0 and identity.strip() else "未识别出公司"
                return f"{stdout.count('已写入')} 个文件 · {named}"
            self._step(job, "docx", docx)

            if job.with_interview:
                self._step(job, "interview", lambda: self._interview(job))
            job.status = "done"
        except Cancelled:
            self._finish_cancelled(job)
        except LocationBlocked as exc:
            job.status = "blocked"
            job.error = str(exc)
            job.stage("match")["status"] = "blocked"
            for stage in job.stages:
                if stage["status"] == "pending":
                    stage["status"] = "skipped"
        except Exception as exc:  # noqa: BLE001
            if job.cancel_requested:  # the killed stage surfaces as an ordinary failure
                self._finish_cancelled(job)
            else:
                job.status = "failed"
                job.error = str(exc)

    def _interview(self, job):
        code, _, _ = self._cli("interview_prep.py", "--run", str(job.run_dir), "--config",
                               str(self.config_path), ok_codes=(0, 1), job=job)
        return "partial" if code == 1 else ""

    def _run_interview(self, job):
        job.status = "running"
        try:
            self._step(job, "interview", lambda: self._interview(job))
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            if job.cancel_requested:
                self._finish_cancelled(job)
            else:
                job.status = "failed"
                job.error = str(exc)

    def _run_dir_for(self, job):
        """The run directory match_job created for this job, when the stage was
        stopped before it reported one. run_meta.json is only written at the
        end, so the run is recognised by its copy of the JD, and only among
        directories made after the job began: an earlier, finished run of the
        same JD must never be taken for the unfinished one."""
        if job.jd_path is None or not self.outputs_dir.is_dir():
            return None
        try:
            jd_text = Path(job.jd_path).read_text(encoding="utf-8")
        except OSError:
            return None
        for run in sorted(self.outputs_dir.iterdir(), reverse=True):
            if not run.is_dir() or run.name[:15] < job.started_stamp:
                continue
            try:
                if (run / "jd.txt").read_text(encoding="utf-8") == jd_text:
                    return run
            except OSError:
                continue
        return None

    def _finish_cancelled(self, job):
        if job.kind == "full" and job.run_dir is None:
            job.run_dir = self._run_dir_for(job)
        removed = self._discard_unfinished(job)
        if job.kind == "full":
            job.run_dir = None
        job.status = "cancelled"
        job.error = "已中止" + (f"，已删除未完成的内容：{'、'.join(removed)}" if removed else "")
        for stage in job.stages:
            if stage["status"] in ("running", "failed"):
                stage["status"], stage["message"] = "cancelled", "已中止"
            elif stage["status"] == "pending":
                stage["status"] = "skipped"


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
    downloads = [path.name for path in (latest_docx(run_dir, "resume"), latest_docx(run_dir, "cover_letter"))
                 if path is not None]
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
            docs, downloads = run_documents(run["dir"])
            stems = {d["stem"] for d in docs}
            history.append({"id": run["id"], "jd": Path(str(run["meta"].get("input", {}).get("jd_path", ""))).name,
                            "started": run["meta"].get("started_at", ""), "status": run["meta"].get("run_status"),
                            "counts": run["counts"], "docs": docs,
                            "downloads": [f for f in downloads if f.endswith(".docx")],
                            "can_interview": "resume_tailored" in stems and "interview_prep" not in stems})
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
                             str(data.get("with_interview", "0")).lower() in ("1", "true", "on", "yes"))
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(job.to_dict()), 202

    @app.post("/api/interview/<run_id>")
    def api_interview(run_id):
        """Interview prep for a finished run, on request."""
        pipe = app.config["PIPELINE"]
        run_dir = _run_dir(run_id)
        if latest_output(run_dir, "resume_tailored") is None:
            return jsonify({"error": "这次运行没有生成简历，无法准备面试"}), 400
        online, model = pipe.model_online()
        if not online:
            return jsonify({"error": f"本机模型服务不在线：{model}"}), 503
        try:
            job = pipe.start_interview(run_dir)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(job.to_dict()), 202

    @app.post("/api/cancel/<job_id>")
    def api_cancel(job_id):
        pipe = app.config["PIPELINE"]
        job = pipe.jobs.get(job_id)
        if job is None:
            abort(404)
        if not pipe.cancel(job):
            return jsonify({"error": "任务已经结束，无需中止"}), 409
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
        # File names now carry the JD name, which may hold any letters; what matters
        # is that the file sits directly in this run directory.
        if not re.fullmatch(r"[\w\-]+(\.[\w\-]+)*\.(md|docx|json)", name):
            abort(404)
        path = run_dir / name
        if not path.is_file() or path.resolve().parent != run_dir.resolve():
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
