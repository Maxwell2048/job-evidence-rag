"""Build a static local site that lists every match_job run and renders its
Markdown outputs (report, CV suggestions, tailored resume, interview prep).

    python build_site.py            # writes site/index.html + site/<run>/*.html
    python build_site.py --open     # then opens it in the default browser

No dependencies and no server: the pages are plain HTML opened from disk.
The Markdown converter covers only what this project writes (headings,
bullet/numbered lists, tables, paragraphs, bold, code, links, rules).
"""
import argparse
import html
import json
import re
import shutil
import sys
import webbrowser
from pathlib import Path

from tailor_cv import output_number

PROJECT_ROOT = Path(__file__).resolve().parent
DOC_ORDER = [("report", "匹配报告"), ("cv_suggestions", "CV 建议"),
             ("resume_tailored", "完整简历"), ("interview_prep", "面试准备")]
VERDICT_LABELS = {"direct": "直接", "related": "相关", "insufficient": "不足", "error": "错误"}

CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif;max-width:960px;
margin:0 auto;padding:24px;line-height:1.55;color:#1f2933;background:#fff}
nav{font-size:14px;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #d9dee3}
nav a{margin-right:14px}nav .here{font-weight:600;color:#1f2933}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid #d9dee3;padding:6px 8px;vertical-align:top;text-align:left}
th{background:#f3f5f7}tr:nth-child(even) td{background:#fafbfc}
code{background:#f3f5f7;padding:1px 4px;border-radius:3px;font-size:90%}
h1{font-size:26px}h2{font-size:20px;margin-top:28px;border-bottom:1px solid #e5e9ed;padding-bottom:4px}
h3{font-size:16px;margin-top:20px}ul{padding-left:22px}li{margin:3px 0}
.badge{display:inline-block;padding:0 6px;border-radius:9px;font-size:12px;background:#eef2f6;margin-right:4px}
.muted{color:#6b7785;font-size:13px}hr{border:0;border-top:1px solid #d9dee3;margin:24px 0}
"""


# ---------------------------------------------------------------- markdown

INLINE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{m.group(1)}</code>"),
    (re.compile(r"\*\*(.+?)\*\*"), lambda m: f"<strong>{m.group(1)}</strong>"),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"),
     lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>'),
]


def inline(text):
    out = html.escape(text, quote=False)
    for pattern, repl in INLINE:
        out = pattern.sub(repl, out)
    return out


def md_to_html(text):
    """Small converter for this project's Markdown; unknown constructs fall
    back to paragraphs so nothing is silently dropped."""
    lines = text.splitlines()
    out, i = [], 0
    list_stack = []  # (indent, tag)

    def close_lists(to_indent=-1):
        while list_stack and list_stack[-1][0] > to_indent:
            out.append(f"</{list_stack.pop()[1]}>")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            close_lists()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if re.match(r"^-{3,}$", stripped):
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", lines[i + 1].strip()):
            close_lists()
            header = [c.strip() for c in stripped.strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header)
                       + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in _split_row(lines[i].strip())]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue
        item = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if item:
            indent = len(item.group(1))
            tag = "ol" if item.group(2)[0].isdigit() else "ul"
            close_lists(indent)
            if not list_stack or list_stack[-1][0] < indent:
                list_stack.append((indent, tag))
                out.append(f"<{tag}>")
            out.append(f"<li>{inline(item.group(3))}</li>")
            i += 1
            continue
        close_lists()
        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||-{3,}|\s*[-*]\s|\s*\d+\.\s)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + "<br>".join(inline(p) for p in para) + "</p>")
    close_lists()
    return "\n".join(out)


def _split_row(row):
    cells, current, escaped = [], [], False
    for ch in row.strip("|"):
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "|":
            cells.append("".join(current))
            current = []
        else:
            current.append(ch)
    cells.append("".join(current))
    return cells


# ---------------------------------------------------------------- runs


def scan_runs(outputs_dir):
    runs = []
    for run_dir in sorted(Path(outputs_dir).iterdir(), reverse=True):
        meta_path = run_dir / "run_meta.json"
        if not run_dir.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        counts = {}
        matches_path = run_dir / "matches.json"
        if matches_path.exists():
            try:
                for m in json.loads(matches_path.read_text(encoding="utf-8"))["matches"]:
                    key = m["verdict"] if m.get("processing_status") == "ok" else "error"
                    counts[key] = counts.get(key, 0) + 1
            except (OSError, ValueError, KeyError):
                pass
        docs = []
        for stem, label in DOC_ORDER:
            files = sorted((p for p in run_dir.glob(f"{stem}*.md")
                            if p.stem == stem or p.stem[len(stem):].lstrip("-").isdigit()),
                           key=lambda p: output_number(p, stem))
            for path in files:
                docs.append({"stem": stem, "label": label, "path": path,
                             "latest": path == files[-1]})
        runs.append({"id": run_dir.name, "dir": run_dir, "meta": meta, "counts": counts,
                     "docs": docs})
    return runs


def page(title, body, nav_html=""):
    return (f"<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><nav>{nav_html}</nav>{body}</body></html>")


def build(outputs_dir, site_dir):
    site_dir = Path(site_dir)
    if site_dir.exists():
        shutil.rmtree(site_dir)
    site_dir.mkdir(parents=True)
    runs = scan_runs(outputs_dir)
    rows = []
    for run in runs:
        run_out = site_dir / run["id"]
        run_out.mkdir()
        links = []
        for doc in run["docs"]:
            name = doc["path"].stem + ".html"
            nav_items = ['<a href="../index.html">← 运行列表</a>'] + [
                (f'<span class="here">{d["label"]}</span>' if d["path"] == doc["path"]
                 else f'<a href="{d["path"].stem}.html">{d["label"]}</a>')
                + ("" if d["latest"] else '<span class="muted">(旧)</span>')
                for d in run["docs"]]
            body = md_to_html(doc["path"].read_text(encoding="utf-8"))
            (run_out / name).write_text(page(f"{doc['label']} · {run['id']}", body,
                                             " ".join(nav_items)), encoding="utf-8")
            if doc["latest"]:
                links.append(f'<a href="{run["id"]}/{name}">{doc["label"]}</a>')
        meta = run["meta"]
        jd = Path(str(meta.get("input", {}).get("jd_path", ""))).name
        badges = "".join(f'<span class="badge">{VERDICT_LABELS.get(k, k)} {v}</span>'
                         for k, v in sorted(run["counts"].items()))
        rows.append(f"<tr><td>{html.escape(meta.get('started_at', run['id']))}</td>"
                    f"<td>{html.escape(jd)}</td><td>{html.escape(str(meta.get('run_status')))}"
                    f"<span class=\"muted\"> {html.escape(str(meta.get('scope', '')))}</span></td>"
                    f"<td>{badges}</td><td>{' · '.join(links) or '—'}</td></tr>")
    body = ("<h1>求职 RAG 运行记录</h1>"
            "<p class=\"muted\">每次 match_job.py 运行一行；链接指向该运行目录里最新的一份文档。"
            "重新运行 build_site.py 即可刷新。</p>"
            "<table><thead><tr><th>时间</th><th>JD</th><th>状态</th><th>判断</th><th>文档</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5>outputs/ 里还没有运行记录</td></tr>'}</tbody></table>")
    (site_dir / "index.html").write_text(page("求职 RAG 运行记录", body, '<span class="here">运行列表</span>'),
                                         encoding="utf-8")
    return len(runs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成本地静态查看页面")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--site", type=Path, default=PROJECT_ROOT / "site")
    parser.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    args = parser.parse_args(argv)
    count = build(args.outputs, args.site)
    index = (args.site / "index.html").resolve()
    print(f"已生成 {count} 次运行的页面：{index}")
    if args.open:
        webbrowser.open(index.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
