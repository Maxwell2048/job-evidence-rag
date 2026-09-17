"""Build a reviewed source-only copy in a new or empty directory.

No recursive cleanup or overwriting is supported. Generic secret detection
runs on every copied text file. User-specific regex patterns can be supplied
from a private, untracked JSON file; they are never included in the release.
Automated scanning is a limited check, not a guarantee of complete redaction.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

INCLUDE_FILES = [
    "README.md", "README.zh-CN.md", "CHANGELOG.md", "LICENSE", "requirements.txt", "requirements-test.txt",
    ".github/workflows/tests.yml", "config.local.example.json",
    "project_template.md", "start.bat", "make_public_release.py",
    "add_jd.py", "build_resume.py", "build_site.py", "evidence_matcher.py", "export_docx.py",
    "import_uwa_kb.py", "interview_prep.py", "jd_parser.py", "job_identity.py", "job_summary.py", "local_llm.py",
    "location_gate.py", "match_job.py", "report_writer.py", "search_experience.py", "tailor_cv.py", "webapp.py",
    "docs/jd_matcher_spec.md",
    "examples/experiences/demo_project.md", "examples/junior_developer_jd.txt",
    "webui/index.html",
    "tests/run_eval.py", "tests/eval_jd.txt", "tests/public_eval_set.json", "tests/public_eval_jd.txt",
]
# Tests that need only synthetic data. test_eval_set.py and eval_set.json refer
# to the private corpus and stay out.
INCLUDE_TESTS = [
    "test_add_jd.py", "test_build_resume.py", "test_build_site.py", "test_evidence_matcher.py",
    "test_export_docx.py", "test_import_uwa_kb.py", "test_interview_prep.py", "test_jd_parser.py",
    "test_job_identity.py", "test_job_summary.py", "test_local_llm.py", "test_match_job.py", "test_report_writer.py",
    "test_retriever.py", "test_sources.py", "test_tailor_cv.py", "test_webapp.py",
    "test_public_release.py", "test_public_eval.py", "test_location_gate.py",
    "test_quote_tolerance.py",
]
KEEP_DIRS = ["experiences", "resume", "jobs", "outputs"]

PUBLIC_GITIGNORE = """# Python
__pycache__/
*.pyc
.venv/

# Local model service config (credentials, machine-specific settings)
config.local.json

# Private material: never commit your own corpus, resume, JDs or generated runs
/experiences/*
!/experiences/.gitkeep
/resume/*
!/resume/.gitkeep
/jobs/*
!/jobs/.gitkeep
/outputs/*
!/outputs/.gitkeep
/_sources/
/_review/
site/
.claude/
*.log
.env
.env.*
private_patterns*.json
"""

# Generic secret formats only. Personal patterns belong in an ignored local file.
FORBIDDEN = [r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}",
             r"sk-[A-Za-z0-9_-]{20,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".html", ".bat", ".css", ".js", ""}


def validate_destination(out):
    out = Path(out).resolve()
    root = PROJECT_ROOT.resolve()
    if out == Path(out.anchor) or out == root or root in out.parents or out in root.parents:
        raise ValueError("Output must not be a filesystem root, source directory, ancestor or descendant")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be a new or empty directory; existing files are never cleaned or overwritten")
    return out


def load_patterns(path=None):
    import json
    patterns = list(FORBIDDEN)
    if path is not None:
        custom = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(custom, list) or not all(isinstance(x, str) and x for x in custom):
            raise ValueError("Private patterns must be a JSON array of nonempty regex strings")
        patterns.extend(custom)
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def copy_tree(out):
    out = validate_destination(out)
    files = INCLUDE_FILES + [f"tests/{name}" for name in INCLUDE_TESTS]
    # Resolve and validate every allowlisted input before writing anything.
    # A link anywhere below the project root changes the resolved path. Compare
    # against the resolved root: the root itself may legitimately be spelled
    # differently (Windows 8.3 short names such as RUNNER~1 in temp paths).
    root = PROJECT_ROOT.resolve()
    for rel in files:
        src = PROJECT_ROOT / rel
        if not src.is_file() or src.resolve() != root / rel:
            raise ValueError(f"Missing or linked source file: {rel}")
    out.mkdir(parents=True, exist_ok=True)
    for rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / rel, dst)
    for name in KEEP_DIRS:
        (out / name).mkdir(exist_ok=True)
        (out / name / ".gitkeep").write_text("", encoding="utf-8")
    (out / ".gitignore").write_text(PUBLIC_GITIGNORE, encoding="utf-8")
    return files


def scan(out, patterns=None):
    """Report file/line and rule number, never the sensitive matching value."""
    compiled = load_patterns() if patterns is None else patterns
    hits = []
    for path in sorted(Path(out).rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # No self-exemption: the release script and LICENSE are scanned too.
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            for rule, pattern in enumerate(compiled, 1):
                if pattern.search(line):
                    hits.append((path.relative_to(out).as_posix(), number, f"rule-{rule}"))
    return hits


def main(argv=None):
    parser = argparse.ArgumentParser(description="Copy an allowlisted source-only release to a new directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--private-patterns", type=Path,
                        help="Optional private JSON regex list; never copied to the release")
    args = parser.parse_args(argv)
    try:
        out = validate_destination(args.out)
        if args.private_patterns:
            private = args.private_patterns.resolve()
            if private == out or out in private.parents:
                raise ValueError("Private patterns must be stored outside the release directory")
        patterns = load_patterns(args.private_patterns)
        copied = copy_tree(out)
        hits = scan(out, patterns)
    except (OSError, ValueError, re.error) as exc:
        # Do not echo regex contents or a secret read from configuration.
        print(f"Release failed ({type(exc).__name__}); verify paths, source files and private pattern syntax", file=sys.stderr)
        return 2
    if hits:
        print("Release scan failed; do not publish this directory:", file=sys.stderr)
        for rel, number, rule in hits:
            print(f"  {rel}:{number}: {rule}", file=sys.stderr)
        return 1
    print(f"Copied {len(copied)} files to {out}. Automated checks passed; manually review before publishing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
