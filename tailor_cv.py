"""Tailor a base CV to one JD using a finished match_job run.

    python tailor_cv.py --run outputs/<run> --resume resume/base_resume.docx --config config.local.json

Everything the model writes must be a rewrite of existing material: verified
experience sections (experiences/*.md), self-declared profile facts, or the
base resume text. Each bullet/paragraph cites a material id plus a verbatim
quote; the program checks the quote is a substring of that material and that
every number in the bullet appears in the project's material. Quotes that
fail are sent back as validation errors within the adapter's retry budget.

Outputs cv_suggestions.md and cv_suggestions.json in the run directory.
Nothing here is a final CV: it is an evidence-linked draft for human review.
"""
import argparse
import datetime
import json
import re
import sys
import zipfile
from html import unescape
from pathlib import Path

from local_llm import LLMError, build_llm, load_config
from search_experience import read_sections

PROJECT_ROOT = Path(__file__).resolve().parent
TAILOR_PROMPT_VERSION = "1"
PROFILE_FILE = "profile.md"
RESUME_ID = "RESUME"
UNVERIFIED_MARK = re.compile(r"事实状态\s*[:：]\s*待核对")
EVIDENCE_KINDS = ("experience", "unverified")
# Sections that describe the candidate's own work. Everything else in an
# experience file (项目背景, 项目内容与团队成果, 基本信息, 证据边界...) is
# context: it may be cited alongside, never as a bullet's only basis.
PERSONAL_SECTION = re.compile(
    r"^(我的贡献|我的工作|个人贡献|个人工作|本人自述|STAR|英文简历表述|My Contributions?\b|My Work\b)", re.I)
MAX_BULLETS = 6
ACTIONS = ["keep", "move_up", "move_down", "trim", "remove"]

# ---------------------------------------------------------------- inputs


def read_resume(path):
    """Plain text of a .docx (paragraphs in document order, <w:br/> as newline),
    or a UTF-8 .md/.txt file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".markdown"}:
        return path.read_text(encoding="utf-8-sig")
    if suffix != ".docx":
        raise ValueError(f"底稿必须是 .docx、.md 或 .txt：{path}")
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    paragraphs = []
    for para in re.findall(r"<w:p(?:\s[^>]*)?>.*?</w:p>", xml, flags=re.S):
        para = re.sub(r"<w:(?:br|cr)\s*/>", "\n", para)
        para = re.sub(r"<w:tab\s*/>", "\t", para)
        runs = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>|(\n)", para, flags=re.S)
        text = "".join(unescape(t) if t else nl for t, nl in runs)
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs) + "\n"


def load_run(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    matches = json.loads((run_dir / "matches.json").read_text(encoding="utf-8"))["matches"]
    jd_text = (run_dir / "jd.txt").read_text(encoding="utf-8")
    if meta.get("scope") != "full":
        raise ValueError(f"运行 {run_dir.name} 只做了提取（scope={meta.get('scope')}），"
                         "先完整运行 match_job.py 再定制简历")
    return meta, matches, jd_text


def load_materials(data_dir, resume_text=None):
    """Numbered material entries grouped by project; the resume (if given) is
    one extra entry, otherwise resume_entry is None."""
    materials, by_project = [], {}
    records = read_sections(Path(data_dir))
    # A file whose 基本信息 says 事实状态：待核对 is resume-derived, not verified.
    unverified_sources = {r["source"] for r in records
                          if UNVERIFIED_MARK.search(r["text"])}
    for index, record in enumerate(records, 1):
        if Path(record["source"]).name == PROFILE_FILE:
            kind = "profile"
        elif record["source"] in unverified_sources:
            kind = "unverified"
        else:
            kind = "experience"
        entry = {
            "material_id": f"M{index:03d}",
            "kind": kind,
            "scope": "personal" if kind == "profile" or PERSONAL_SECTION.match(record["section"])
            else "team",
            "project": record["project"],
            "section": record["section"],
            "source": record["source"],
            "line_start": record["line_start"],
            "line_end": record["line_end"],
            "text": record["text"],
        }
        materials.append(entry)
        by_project.setdefault(record["project"], []).append(entry)
    if resume_text is None:
        return materials, by_project, None
    resume_entry = {"material_id": RESUME_ID, "kind": "resume", "scope": "personal",
                    "project": "简历底稿", "section": "全文", "source": "resume", "line_start": 1,
                    "line_end": resume_text.count("\n") + 1, "text": resume_text}
    materials.append(resume_entry)
    return materials, by_project, resume_entry


# ---------------------------------------------------------------- analysis


def requirement_summary(matches):
    return [{
        "requirement_id": m["requirement_id"],
        "text": m["text"],
        "category": m["category"],
        "importance": m["importance"],
        "verdict": m["verdict"] if m["processing_status"] == "ok" else "unprocessed",
        "missing_aspects": m.get("missing_aspects", []),
        "evidence_projects": sorted({e["project"] for e in m.get("evidence", [])}),
    } for m in matches]


def rank_projects(matches, by_project):
    """Projects ordered by how much verified evidence they gave this JD."""
    scores = {name: 0 for name in by_project}
    weight = {"direct": 3, "related": 1}
    for match in matches:
        if match["processing_status"] != "ok":
            continue
        for evidence in match.get("evidence", []):
            project = evidence["project"]
            if project in scores and Path(evidence["source"]).name != PROFILE_FILE:
                scores[project] += weight.get(match["verdict"], 0)
    ordered = sorted(scores, key=lambda p: (-scores[p], p))
    return [{"project": p, "evidence_score": scores[p]} for p in ordered
            if any(e["kind"] in EVIDENCE_KINDS for e in by_project[p])]


def gaps(matches):
    """Requirements the material cannot support; never to be written as claims."""
    out = []
    for m in matches:
        if m["processing_status"] != "ok":
            out.append({**_gap_base(m), "reason": "处理未完成", "kind": "unprocessed"})
        elif m["verdict"] == "insufficient":
            out.append({**_gap_base(m), "reason": m.get("reason", ""), "kind": "insufficient"})
        elif m["verdict"] == "related" and m.get("missing_aspects"):
            out.append({**_gap_base(m), "reason": "；".join(m["missing_aspects"]),
                        "kind": "partial"})
    return out


def _gap_base(m):
    return {"requirement_id": m["requirement_id"], "text": m["text"],
            "importance": m["importance"]}


# ---------------------------------------------------------------- validation

NUMBER = re.compile(r"\d[\d,.]*\d|\d")
# University unit codes (CITS5505, GENG5505...) read as student work on a resume.
COURSE_CODE = re.compile(r"\b[A-Z]{4}\d{4}\b")


_LOOSE = re.compile(r"[\s\-–—_/,.;:()\[\]\"'“”‘’]+")


def _loose(text):
    return _LOOSE.sub("", text).casefold()


def quote_in(quote, text):
    """Verbatim substring, or equal after ignoring case, whitespace and
    punctuation (so 'problem-solving' matches 'problem solving'). Words must
    still appear in order and in full."""
    if not quote or not quote.strip():
        return False
    if quote in text:
        return True
    loose_quote = _loose(quote)
    return bool(loose_quote) and loose_quote in _loose(text)


def personal_ids(materials_by_id, allowed_ids):
    return {i for i in allowed_ids if materials_by_id[i].get("scope", "personal") == "personal"}


def validate_basis(items, materials_by_id, allowed_ids, text_key="text",
                   label="第 {index} 条", jd_text=None, require_personal=False,
                   no_course_codes=False):
    """Every item must cite existing materials with verbatim quotes, and every
    number in its text must appear in the cited materials (or the allowed
    material pool for that call). A quote that is JD wording rather than
    material text is named as such so the retry can correct it. With
    require_personal, an item citing only team/background sections is rejected;
    with no_course_codes, resume text containing unit codes is rejected."""
    errors = []
    pool_text = "\n".join(materials_by_id[i]["text"] for i in allowed_ids)
    own = personal_ids(materials_by_id, allowed_ids)
    for index, item in enumerate(items, 1):
        who = label.format(index=index)
        text = item.get(text_key, "")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{who} {text_key} 为空")
            continue
        if no_course_codes and COURSE_CODE.search(text):
            errors.append(f"{who}含课程代码 {COURSE_CODE.search(text).group(0)}；简历内容不写课程代码，"
                          "删除它或改为项目性质描述（如 individual project / group project）")
        basis = item.get("basis") or []
        if not basis:
            errors.append(f"{who}缺少 basis 来源引用")
        elif require_personal and not any(b.get("material_id") in own for b in basis):
            errors.append(f"{who}只引用了 scope=team 的项目背景/团队成果章节；"
                          "必须至少引用一处 scope=personal 的本人材料（我的贡献/我的工作/STAR/英文简历表述/自述/profile）")
        for b in basis:
            mid, quote = b.get("material_id"), b.get("quote", "")
            if mid not in allowed_ids:
                errors.append(f"{who}引用了不存在或本次不可用的 material_id {mid!r}")
                continue
            if not quote or not quote_in(quote, materials_by_id[mid]["text"]):
                hint = ""
                if jd_text and quote and quote.lower() in jd_text.lower():
                    hint = "（这是 JD 的措辞，不是材料原文；quote 只能摘自 materials/profile/resume_draft）"
                errors.append(f"{who}对 {mid} 的 quote 不是该材料的逐字连续原文："
                              f"{quote[:60]!r}{hint}")
        for number in NUMBER.findall(text):
            if number not in pool_text:
                errors.append(f"{who}包含材料中没有的数字 {number!r}；"
                              "不要新增或换算数字")
    return errors


def validate_quotes_in(items, text, key="quote"):
    return [f"第 {i} 条 {key} 不是简历底稿的逐字原文：{item.get(key, '')[:60]!r}"
            for i, item in enumerate(items, 1)
            if not item.get(key) or item[key] not in text]


# ---------------------------------------------------------------- prompts

SYSTEM_PROMPT = """你为求职者定制英文简历内容。所有输出都必须是对提供材料的改写，不得新增材料里没有的事实、数字、工具、职位或成果。
材料分三类：experience（已核对的项目经历）、profile（本人自述的基本事实）、resume（现有简历底稿，未经证据核对）。
每条材料还有 scope：personal 是本人的工作或自述，team 是项目背景或团队成果。团队成果不是本人成果：
要点可以引用 team 材料交代背景，但每条要点必须至少引用一处 personal 材料，且不能把 team 材料里的工作写成本人完成。
每条输出都要在 basis 里给出 material_id 和该材料中逐字连续的 quote，说明依据；改写可以精简、换序、突出与岗位相关的部分，但不能夸大。
JD 要求列表里 verdict=insufficient 的要求没有证据支持，不要在简历中声称具备；verdict=related 的要求只能写材料实际覆盖的部分。
简历正文不出现课程代码（CITS5505 之类），用 individual project / group project 等描述项目性质。
简历正文用英文；reason/note 字段用中文。输出符合 schema 的紧凑 JSON（不缩进、不换行、无代码围栏）。"""

BULLETS_SCHEMA = {
    "type": "object",
    "required": ["bullets", "note"],
    "additionalProperties": False,
    "properties": {
        "bullets": {
            "type": "array",
            "maxItems": MAX_BULLETS,
            "items": {
                "type": "object",
                "required": ["text", "basis", "requirement_ids"],
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "basis": {"type": "array", "minItems": 1, "items": {
                        "type": "object", "required": ["material_id", "quote"],
                        "additionalProperties": False,
                        "properties": {"material_id": {"type": "string"},
                                       "quote": {"type": "string", "minLength": 1}}}},
                    "requirement_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "note": {"type": "string"},
    },
}

REVIEW_SCHEMA = {
    "type": "object",
    "required": ["items"],
    "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["quote", "action", "reason"],
        "additionalProperties": False,
        "properties": {"quote": {"type": "string", "minLength": 1},
                       "action": {"type": "string", "enum": ACTIONS},
                       "reason": {"type": "string"}}}}},
}

LETTER_SCHEMA = {
    "type": "object",
    "required": ["summary", "cover_letter"],
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "object", "required": ["text", "basis"],
                    "additionalProperties": False,
                    "properties": {"text": {"type": "string", "minLength": 1},
                                   "basis": BULLETS_SCHEMA["properties"]["bullets"]["items"]["properties"]["basis"]}},
        "cover_letter": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object", "required": ["text", "basis"],
            "additionalProperties": False,
            "properties": {"text": {"type": "string", "minLength": 1},
                           "basis": BULLETS_SCHEMA["properties"]["bullets"]["items"]["properties"]["basis"]}}},
    },
}


# ---------------------------------------------------------------- generation


class CVTailor:
    def __init__(self, llm, materials, by_project, resume_entry, timeout_seconds=None):
        self.llm = llm
        self.materials = materials
        self.by_id = {m["material_id"]: m for m in materials}
        self.by_project = by_project
        self.resume = resume_entry
        self.timeout = timeout_seconds
        self.warnings = []

    @staticmethod
    def _visible(entries):
        return [{"material_id": e["material_id"], "kind": e["kind"],
                 "scope": e.get("scope", "personal"),
                 "section": e["section"], "text": e["text"]} for e in entries]

    def _fits(self, task, payload, schema):
        return self.llm.estimate_context(task, SYSTEM_PROMPT, payload, schema) \
            <= self.llm.prompt_char_budget

    def project_bullets(self, project, requirements):
        entries = self.by_project[project]
        allowed = [e["material_id"] for e in entries] + [RESUME_ID]
        payload = {
            "task_note": f"为项目“{project}”写最多 {MAX_BULLETS} 条针对本 JD 的简历要点；"
                         "优先覆盖 verdict=direct/related 的要求；requirement_ids 列出每条要点对应的 JD 要求。",
            "jd_requirements": requirements,
            "materials": self._visible(entries),
            "resume_draft": {"material_id": RESUME_ID, "text": self.resume["text"]},
        }
        task = f"cv_bullets:{project}"
        if not self._fits(task, payload, BULLETS_SCHEMA):
            payload.pop("resume_draft")
            allowed = allowed[:-1]
            self.warnings.append(f"项目“{project}”的材料加简历底稿超出提示词预算，本次未提供底稿")
            if not self._fits(task, payload, BULLETS_SCHEMA):
                raise LLMError(f"项目“{project}”的材料超出提示词预算，请调大 context_budget",
                               error_type="context_budget")
        data = self.llm.generate_json(
            task, SYSTEM_PROMPT, payload, BULLETS_SCHEMA, self.timeout,
            validate=lambda d: validate_basis(d.get("bullets", []), self.by_id, allowed,
                                              require_personal=True, no_course_codes=True))
        return data

    def resume_review(self, requirements):
        payload = {
            "task_note": "逐条审阅简历底稿里的项目要点和技能行：对本 JD 该保留、上移、下移、精简还是删除；"
                         "quote 必须是底稿逐字原文（可以只引用一句的开头部分，但要连续）。只列需要动作或值得强调的条目。",
            "jd_requirements": requirements,
            "resume_draft": self.resume["text"],
        }
        return self.llm.generate_json(
            "cv_review", SYSTEM_PROMPT, payload, REVIEW_SCHEMA, self.timeout,
            validate=lambda d: validate_quotes_in(d.get("items", []), self.resume["text"]))

    def summary_and_letter(self, requirements, ordered_bullets, gap_list, jd_text):
        profile_entries = [e for e in self.materials if e["kind"] == "profile"]
        allowed = [e["material_id"] for e in profile_entries] + [RESUME_ID] + [
            b["material_id"] for p in ordered_bullets for bl in p["bullets"] for b in bl["basis"]]
        allowed = list(dict.fromkeys(allowed))
        cited = [self.by_id[i] for i in allowed if i != RESUME_ID and self.by_id[i]["kind"] != "profile"]
        payload = {
            "task_note": "写一段 2-3 句的英文简历摘要（summary）和一封 3-4 段的英文 Cover Letter（cover_letter）。"
                         "只能使用 materials、profile 和 resume_draft 里的事实；gaps 里的要求不要声称具备，"
                         "可以坦诚说明愿意学习。每段给出 basis：material_id 必须是 materials/profile/resume_draft "
                         "里的 id，quote 必须逐字摘自该材料的文本。JD 的措辞（jd_text、jd_requirements、gaps 里的话）"
                         "可以在正文里呼应，但绝不能作为 quote；引用 JD 会被程序拒绝。",
            "jd_text": jd_text,
            "jd_requirements": [{k: r[k] for k in ("requirement_id", "text", "importance", "verdict")}
                                for r in requirements],
            "gaps": [{"requirement_id": g["requirement_id"], "text": g["text"],
                      "kind": g["kind"], "reason": g["reason"][:160]} for g in gap_list],
            "tailored_bullets": [{"project": p["project"], "bullets": [b["text"] for b in p["bullets"]]}
                                 for p in ordered_bullets],
            "materials": self._visible(cited),
            "profile": self._visible(profile_entries),
            "resume_draft": {"material_id": RESUME_ID, "text": self.resume["text"]},
        }
        task = "cv_summary_letter"
        if not self._fits(task, payload, LETTER_SCHEMA):
            payload["materials"] = []
            self.warnings.append("摘要/Cover Letter 请求超出预算，未附带项目材料全文，仅提供已生成要点")
            allowed = [i for i in allowed if i == RESUME_ID or self.by_id[i]["kind"] == "profile"]
            if not self._fits(task, payload, LETTER_SCHEMA):
                raise LLMError("摘要/Cover Letter 请求超出提示词预算，请调大 context_budget",
                               error_type="context_budget")

        def validate(d):
            return (validate_basis([d.get("summary", {})], self.by_id, allowed,
                                   label="summary ", jd_text=jd_text, require_personal=True,
                                   no_course_codes=True)
                    + validate_basis(d.get("cover_letter", []), self.by_id, allowed,
                                     label="cover_letter 第 {index} 段", jd_text=jd_text,
                                     require_personal=True, no_course_codes=True))
        return self.llm.generate_json(task, SYSTEM_PROMPT, payload, LETTER_SCHEMA,
                                      self.timeout, validate=validate)


def tailor(llm, meta, matches, jd_text, materials, by_project, resume_entry,
           timeout_seconds=None):
    """Run all generation steps; per-step failures are recorded, not fatal."""
    tailor_obj = CVTailor(llm, materials, by_project, resume_entry, timeout_seconds)
    requirements = requirement_summary(matches)
    order = rank_projects(matches, by_project)
    result = {
        "schema_version": "1.0",
        "prompt_version": TAILOR_PROMPT_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": meta["run_id"],
        "project_order": order,
        "projects": [],
        "resume_review": [],
        "summary": None,
        "cover_letter": [],
        "gaps": gaps(matches),
        "errors": [],
        "warnings": [],
    }
    for item in order:
        project = item["project"]
        try:
            data = tailor_obj.project_bullets(project, requirements)
        except LLMError as exc:
            result["errors"].append({"step": f"bullets:{project}", "error_type": exc.error_type,
                                     "message": str(exc)})
            continue
        result["projects"].append({"project": project, "note": data.get("note", ""),
                                   "bullets": [_attach_sources(b, tailor_obj.by_id)
                                               for b in data.get("bullets", [])]})
    try:
        result["resume_review"] = tailor_obj.resume_review(requirements)["items"]
    except LLMError as exc:
        result["errors"].append({"step": "resume_review", "error_type": exc.error_type,
                                 "message": str(exc)})
    try:
        data = tailor_obj.summary_and_letter(requirements, result["projects"],
                                             result["gaps"], jd_text)
        result["summary"] = _attach_sources(data["summary"], tailor_obj.by_id)
        result["cover_letter"] = [_attach_sources(p, tailor_obj.by_id) for p in data["cover_letter"]]
    except LLMError as exc:
        result["errors"].append({"step": "summary_letter", "error_type": exc.error_type,
                                 "message": str(exc)})
    result["warnings"] = tailor_obj.warnings
    result["status"] = "complete" if not result["errors"] else "partial"
    return result


def _attach_sources(item, by_id):
    basis = []
    for b in item.get("basis", []):
        m = by_id[b["material_id"]]
        basis.append({**b, "kind": m["kind"], "project": m["project"], "section": m["section"],
                      "source": m["source"], "line_start": m["line_start"],
                      "line_end": m["line_end"]})
    return {**item, "basis": basis}


# ---------------------------------------------------------------- output

VERDICT_LABELS = {"direct": "直接支持", "related": "相关但不足",
                  "insufficient": "证据不足", "unprocessed": "未处理"}
IMPORTANCE_LABELS = {"required": "必需", "preferred": "优先", "unspecified": "未说明"}
ACTION_LABELS = {"keep": "保留并强调", "move_up": "上移", "move_down": "下移",
                 "trim": "精简", "remove": "删除"}
KIND_LABELS = {"experience": "已核对经历", "unverified": "待核对经历（仅底稿）",
               "profile": "自述事实", "resume": "简历底稿（未核对）"}
UNVERIFIED_KINDS = ("unverified", "resume")


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _source_line(basis):
    parts = []
    for b in basis:
        label = KIND_LABELS.get(b["kind"], b["kind"])
        where = "简历底稿" if b["kind"] == "resume" else \
            f"{b['source']} › {b['section']}（行 {b['line_start']}-{b['line_end']}）"
        parts.append(f"[{label}] {where}：“{b['quote']}”")
    return "；".join(parts)


def write_markdown(result, meta, matches, resume_path, out_path):
    lines = ["# CV 定制建议", ""]
    lines.append(f"- 匹配运行：{result['run_id']}（JD：{meta['input']['jd_path']}）")
    lines.append(f"- 简历底稿：{resume_path}")
    gen = meta.get("models", {}).get("generator", {})
    lines.append(f"- 生成模型：{gen.get('name')}；定制提示词版本 {result['prompt_version']}；"
                 f"生成时间 {result['generated_at']}")
    lines.append(f"- 状态：{result['status']}" + (
        "" if not result["errors"] else "（部分步骤失败，见文末）"))
    lines.append("- 说明：以下所有内容都是对已有材料的改写，每条附来源与逐字引用。"
                 "标为“简历底稿（未核对）”的依据没有经过项目证据核对；"
                 "判断为证据不足的要求列在“不要硬写的缺口”，不要写进简历。")
    lines += ["", "## 1. 岗位要求对照", "",
              "| ID | 要求 | 优先级 | 判断 | 证据项目 |", "| --- | --- | --- | --- | --- |"]
    for r in requirement_summary(matches):
        lines.append(f"| {r['requirement_id']} | {_cell(r['text'])} | "
                     f"{IMPORTANCE_LABELS.get(r['importance'], r['importance'])} | "
                     f"{VERDICT_LABELS.get(r['verdict'], r['verdict'])} | "
                     f"{_cell('、'.join(r['evidence_projects']) or '—')} |")
    lines += ["", "## 2. 建议的项目顺序", ""]
    for i, p in enumerate(result["project_order"], 1):
        lines.append(f"{i}. {p['project']}（证据得分 {p['evidence_score']}；"
                     "得分 = 直接支持×3 + 相关×1，仅用于排序）")
    lines += ["", "## 3. 各项目要点改写建议", ""]
    if not result["projects"]:
        lines.append("（无：要点生成步骤失败或没有可用项目）")
    for p in result["projects"]:
        lines += [f"### {p['project']}", ""]
        if p.get("note"):
            lines += [f"说明：{p['note']}", ""]
        for b in p["bullets"]:
            req = "、".join(b.get("requirement_ids") or []) or "—"
            lines.append(f"- {b['text']}")
            lines.append(f"  - 对应要求：{req}")
            lines.append(f"  - 来源：{_source_line(b['basis'])}")
        lines.append("")
    lines += ["## 4. 简历摘要建议", ""]
    if result["summary"]:
        lines += [result["summary"]["text"], "",
                  f"来源：{_source_line(result['summary']['basis'])}", ""]
    else:
        lines += ["（未生成）", ""]
    lines += ["## 5. 底稿逐条审阅", ""]
    if result["resume_review"]:
        lines += ["| 动作 | 底稿原文 | 理由 |", "| --- | --- | --- |"]
        for item in result["resume_review"]:
            lines.append(f"| {ACTION_LABELS.get(item['action'], item['action'])} | "
                         f"{_cell(item['quote'])} | {_cell(item['reason'])} |")
    else:
        lines.append("（未生成）")
    lines += ["", "## 6. 不要硬写的缺口", ""]
    if not result["gaps"]:
        lines.append("（无）")
    for g in result["gaps"]:
        kind = {"insufficient": "证据不足", "partial": "仅部分覆盖", "unprocessed": "未处理"}[g["kind"]]
        lines.append(f"- {g['requirement_id']} {g['text']}（{IMPORTANCE_LABELS.get(g['importance'], g['importance'])}，{kind}）：{g['reason']}")
    lines += ["", "## 7. Cover Letter 草稿", ""]
    if result["cover_letter"]:
        for para in result["cover_letter"]:
            lines += [para["text"], "", f"来源：{_source_line(para['basis'])}", ""]
    else:
        lines += ["（未生成）", ""]
    lines += ["## 8. 人工复核清单", "",
              "- 逐条核对“来源”里的引用是否真的支持改写后的说法，尤其是形容词和程度副词。",
              "- 标为“简历底稿（未核对）”的内容，确认底稿本身属实后再使用。",
              "- 第 6 节的缺口不要写进简历；面试可坦诚说明学习计划。",
              "- 数字已由程序核对存在于材料中，但语境是否一致仍需人工确认。"]
    if result["warnings"]:
        lines += ["", "## 警告", ""] + [f"- {w}" for w in result["warnings"]]
    if result["errors"]:
        lines += ["", "## 失败步骤", ""] + [
            f"- {e['step']}（{e['error_type']}）：{e['message']}" for e in result["errors"]]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_output(run_dir, stem):
    path = run_dir / f"{stem}.md"
    counter = 2
    while path.exists():
        path = run_dir / f"{stem}-{counter}.md"
        counter += 1
    return path


def output_number(path, stem):
    """cv_suggestions.md -> 1, cv_suggestions-3.md -> 3 (the unique_output scheme)."""
    rest = Path(path).stem[len(stem):]
    return int(rest[1:]) if rest.startswith("-") and rest[1:].isdigit() else 1


def latest_output(run_dir, stem, suffix=".md"):
    """Newest file written by unique_output for this stem, or None."""
    files = [p for p in Path(run_dir).glob(f"{stem}*{suffix}")
             if p.stem == stem or (p.stem.startswith(stem + "-") and p.stem[len(stem) + 1:].isdigit())]
    return max(files, key=lambda p: output_number(p, stem)) if files else None


# ---------------------------------------------------------------- CLI


def main(argv=None):
    parser = argparse.ArgumentParser(description="根据匹配结果定制简历内容（Markdown）")
    parser.add_argument("--run", required=True, type=Path, help="match_job 的运行目录")
    parser.add_argument("--resume", required=True, type=Path, help="简历底稿（.docx/.md/.txt）")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "experiences")
    parser.add_argument("--json", action="store_true", help="stdout 只输出结果 JSON")
    args = parser.parse_args(argv)
    try:
        meta, matches, jd_text = load_run(args.run)
        resume_text = read_resume(args.resume)
        config = load_config(args.config)
        llm = build_llm(config)
    except (OSError, ValueError, LLMError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    materials, by_project, resume_entry = load_materials(args.data, resume_text)
    print(f"[tailor-cv] 材料 {len(materials)} 条，项目 {len(by_project)} 个，"
          f"要求 {len(matches)} 条；开始生成……", file=sys.stderr)
    result = tailor(llm, meta, matches, jd_text, materials, by_project, resume_entry,
                    config.get("timeout_seconds"))
    result["resume_path"] = str(args.resume)
    out_md = unique_output(args.run, "cv_suggestions")
    out_json = out_md.with_suffix(".json")
    write_markdown(result, meta, matches, args.resume, out_md)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps({**result, "output_md": str(out_md)}, ensure_ascii=False))
    else:
        print(f"已写入：{out_md}")
        print(f"状态：{result['status']}；要点 "
              f"{sum(len(p['bullets']) for p in result['projects'])} 条，缺口 {len(result['gaps'])} 条")
        for e in result["errors"]:
            print(f"失败步骤 {e['step']}：{e['message']}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
