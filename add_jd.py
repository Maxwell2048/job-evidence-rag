"""Save a job description from the clipboard (or stdin/file) into jobs/.

    python add_jd.py                 # clipboard -> jobs/<date>-<slug>.txt
    python add_jd.py --name acme-ml  # choose the slug yourself
    python add_jd.py --run           # then run match_job.py with config.local.json
    Get-Content jd.txt | python add_jd.py --stdin

Files are never overwritten: a numeric suffix is added on collision. The text
is saved verbatim (UTF-8, newlines normalised); no cleaning or parsing here.
"""
import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
JOBS_DIR = PROJECT_ROOT / "jobs"
MIN_CHARS = 80
SLUG_MAX = 40


def read_clipboard():
    """Windows clipboard via PowerShell, decoded as UTF-8."""
    command = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
               "Get-Clipboard -Raw")
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"无法读取剪贴板：{exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError("无法读取剪贴板："
                           + completed.stderr.decode("utf-8", "replace").strip())
    return completed.stdout.decode("utf-8", "replace")


def normalise(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def make_slug(text, explicit=None):
    """Filename-safe slug from --name or the first non-empty line."""
    source = explicit if explicit else next(
        (line.strip() for line in text.split("\n") if line.strip()), "jd")
    slug = re.sub(r"[^\w一-鿿]+", "-", source).strip("-")
    slug = slug[:SLUG_MAX].rstrip("-").lower()
    return slug or "jd"


def unique_path(folder, stem, suffix=".txt"):
    path = folder / f"{stem}{suffix}"
    counter = 2
    while path.exists():
        path = folder / f"{stem}-{counter}{suffix}"
        counter += 1
    return path


def save_jd(text, folder=JOBS_DIR, name=None, now=None, min_chars=MIN_CHARS):
    body = normalise(text)
    if len(body.strip()) < min_chars:
        raise ValueError(f"文本只有 {len(body.strip())} 个字符，少于 {min_chars}；"
                         "确认已复制完整 JD，或加 --force")
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M")
    folder.mkdir(parents=True, exist_ok=True)
    path = unique_path(folder, f"{stamp}-{make_slug(body, name)}")
    path.write_text(body, encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="把剪贴板里的 JD 保存到 jobs/")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--stdin", action="store_true", help="从标准输入读取而不是剪贴板")
    source.add_argument("--file", type=Path, help="从文件读取而不是剪贴板")
    parser.add_argument("--name", help="文件名中的说明部分，默认取 JD 第一行")
    parser.add_argument("--jobs", type=Path, default=JOBS_DIR, help="保存目录")
    parser.add_argument("--force", action="store_true", help="允许保存很短的文本")
    parser.add_argument("--run", action="store_true",
                        help="保存后立即运行 match_job.py（使用 --config）")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.local.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        if args.stdin:
            text = sys.stdin.read()
        elif args.file:
            text = args.file.read_text(encoding="utf-8-sig")
        else:
            text = read_clipboard()
        path = save_jd(text, args.jobs, args.name,
                       min_chars=0 if args.force else MIN_CHARS)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    size = len(path.read_text(encoding="utf-8"))
    print(f"已保存：{path}（{size} 字符）")
    if not args.run:
        print(f"下一步：{Path(sys.executable).name} match_job.py "
              f"--jd \"{path}\" --config \"{args.config}\"")
        return 0
    command = [sys.executable, "-X", "utf8", str(PROJECT_ROOT / "match_job.py"),
               "--jd", str(path), "--config", str(args.config),
               "--top-k", str(args.top_k)]
    print("运行：" + " ".join(command), file=sys.stderr)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
