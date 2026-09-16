"""Convert project cards from UWA_Project_Knowledge_Base into experiences/*.md.

    python import_uwa_kb.py --kb "<path to the knowledge base directory>" --projects UWA-P10,UWA-P11
    python import_uwa_kb.py --kb ... --list

Every fact keeps its source id, locator and evidence type; the cited evidence
extracts (evidence/<S-id>.txt/.json) are copied into _sources/uwa_kb/ so the
citations resolve on this machine. Sections are mapped to this project's
conventions: personal sections become "我的贡献 …" (retrieved by default),
whole-project sections of an individually owned project become "我的工作 …",
team-project narrative becomes "项目内容与团队成果 …", and limitation sections
go under "证据边界". The English resume line per project is curated in
RESUME_LINES; a project without one gets a 待写 placeholder.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIENCES = PROJECT_ROOT / "experiences"
SOURCES = PROJECT_ROOT / "_sources" / "uwa_kb"
KB_VERSION_NOTE = "UWA_Project_Knowledge_Base（整理日期 2026-09-12）"

BOUNDARY_WORDS = ("限制", "边界", "差异", "缺口", "待补充", "问题")

# project_id -> (file stem, Chinese title, English resume line)
PROFILES = {
    "UWA-P05": ("cits4401_cyberwin_requirements", "CyberWin 网络安全培训系统需求与架构设计",
                "CyberWin | Requirements Engineering | group project\n"
                "Recorder for structured stakeholder interviews; contributed to user stories, functional/non-functional "
                "requirements and a three-layer architecture proposal for a gamified security-training system "
                "(design deliverables only, no implementation)."),
    "UWA-P10": ("cits5017_cnn_transfer_learning", "小型 CNN 与 MobileNetV3Small 图像分类比较",
                "CNN & Transfer Learning | Computer Vision | individual project\n"
                "Compared a Bayesian-tuned 4-layer CNN (20,908 parameters, 10 keras_tuner trials) with a frozen-backbone "
                "MobileNetV3Small on 12-class 64×64 images; held-out test accuracy 51.17% vs 61.33% (n=600), "
                "a 10.17 percentage-point gain."),
    "UWA-P11": ("cits5017_temperature_forecasting_vae", "GRU 温度预测与 VAE 气候序列生成",
                "Temperature Forecasting & VAE | Sequence Modelling | individual project\n"
                "Benchmarked direct, rolling and seq2seq GRU models on 84-month WA station temperature sequences "
                "(12-month test MAE 4.11 / 4.10 / 4.12 on 105 sequences) and trained an 8-dimensional-latent VAE "
                "to generate synthetic maximum-temperature series."),
    "UWA-P12": ("cits4012_medical_abstract_classification", "Word2Vec 与 BioWordVec 医学摘要分类",
                "Medical Abstract Classification | NLP | individual assignment\n"
                "Compared locally trained Word2Vec with pretrained BioWordVec embeddings in a frozen Embedding + LSTM "
                "classifier on 11,550 abstracts (5 classes); BioWordVec reached 63.38% accuracy / 0.6006 weighted-F1 "
                "on the 2,310-sample validation set that was also used for early stopping."),
    "UWA-P13": ("cits4012_natural_language_inference", "BoW BiLSTM 与 Self-Attention 自然语言推理",
                "Natural Language Inference | NLP | two-person project\n"
                "Built and improved the BoW and BiLSTM encoders (validation accuracy 70.61% / 72.99%), contributed to "
                "the Self-Attention model and attention ablations (best validation accuracy 73.75%), wrote the "
                "ACL-format report and produced attention visualisations and confidence analysis."),
    "UWA-P15": ("cits5504_aviation_data_warehouse", "航空旅客数据仓库 OLAP 与关联规则挖掘",
                "Aviation Data Warehouse | DW & BI | two-person project\n"
                "Star-schema warehouse (passenger-flight fact + 5 dimensions) in PostgreSQL from 98,619 → 98,073 cleaned "
                "records; ROLLUP/CUBE OLAP views, FP-Growth rules (lift ≈ 1.03), Power BI dashboard and a simplified "
                "Databricks/Delta Lake warehouse; individual module split not recorded."),
    "UWA-P16": ("cits5504_airline_route_graph_db", "Neo4j 航空路线图数据库与 Cypher 分析",
                "Airline Route Graph DB | Neo4j · Cypher | individual report\n"
                "ETL with pandas into Neo4j (2,795 airports, 488 airlines, 57,301 ROUTE, 16,768 OPERATES); Cypher "
                "queries for Beijing→Perth paths within 3 hops (650 airport sequences), most-shared-route airline pairs "
                "and APOC country counts."),
}


def load_kb(kb_dir):
    kb_dir = Path(kb_dir)
    projects = json.loads((kb_dir / "projects.json").read_text(encoding="utf-8"))
    return kb_dir, {p["project_id"]: p for p in projects}


def is_personal_project(project):
    own = project.get("ownership", "")
    return own.startswith("个人") and not any(w in own for w in ("合作", "团队", "两人", "Group"))


def classify(section_name, personal_project):
    if section_name.startswith("个人") or "个人贡献" in section_name or "个人工作" in section_name:
        return "personal"
    if any(w in section_name for w in BOUNDARY_WORDS):
        return "boundary"
    return "personal" if personal_project else "team"


def slug(text):
    return re.sub(r"[^\w一-鿿]+", "_", text).strip("_").lower() or "project"


def copy_evidence(kb_dir, source_ids):
    copied = []
    (SOURCES / "evidence").mkdir(parents=True, exist_ok=True)
    for sid in sorted(source_ids):
        for suffix in (".txt", ".json"):
            src = kb_dir / "evidence" / f"{sid}{suffix}"
            if src.exists():
                shutil.copy2(src, SOURCES / "evidence" / src.name)
                copied.append(src.name)
    return copied


def evidence_line(fact):
    parts = []
    for s in fact.get("sources", []):
        sid = s["source_id"]
        parts.append(f"[{sid}: {s.get('locator', '')}](../_sources/uwa_kb/evidence/{sid}.txt) "
                     f"原文件 {s.get('source_path', '')}")
    kind = fact.get("evidence_type", "")
    return "证据：" + "；".join(parts) + (f"；证据类型 {kind}" if kind else "") + "。"


def render(project, stem, title, resume_line, kb_dir):
    personal_project = is_personal_project(project)
    tags = project.get("tags") or []
    lines = [f"# {title}", "", "## 基本信息"]
    lines.append(f"- 项目类型：{'个人' if personal_project else '团队'}课程项目；知识库编号 {project['project_id']}。")
    lines.append(f"- 课程：{project.get('course') or '待确认'}")
    lines.append(f"- 时间：{project.get('period') or '待确认（知识库未确认学期）'}")
    lines.append(f"- 归属：{project.get('ownership') or '待确认'}")
    lines.append(f"- 状态：{project.get('status') or '待确认'}")
    lines.append(f"- 技术与关键词：{'、'.join(tags) if tags else '待补充'}")
    lines.append(f"- 成绩：{'未找到记录' if project.get('grade') is None else project['grade']}")
    lines.append(f"- 事实状态：已核对（二手核对：依据 {KB_VERSION_NOTE} 的原文提取，原始文件位于知识库整理机器；"
                 "提取文本与哈希见 _sources/uwa_kb）。")
    lines.append("")

    groups = {"personal": [], "team": [], "boundary": []}
    for section in project.get("sections", []):
        groups[classify(section["section"], personal_project)].append(section)

    def emit(heading_prefix, sections):
        for section in sections:
            lines.append(f"## {heading_prefix} {section['section']}".rstrip())
            for fact in section.get("facts", []):
                lines.append(fact["text"].strip())
                lines.append(evidence_line(fact))
            lines.append("")

    if groups["team"]:
        emit("项目内容与团队成果", groups["team"])
    emit("我的工作" if personal_project else "我的贡献", groups["personal"])
    if tags:
        lines.append("## 相关能力")
        lines.append("相关能力：" + ", ".join(tags) + "。")
        lines.append("")
    lines.append("## 英文简历表述")
    lines.append(resume_line or "待写：请依据上述已核对事实撰写。")
    lines.append("")
    lines.append("## 证据边界")
    for section in groups["boundary"]:
        for fact in section.get("facts", []):
            lines.append(fact["text"].strip())
            lines.append(evidence_line(fact))
    lines.append(f"本文件由 import_uwa_kb.py 从 {KB_VERSION_NOTE} 生成；未重新运行代码或训练模型。"
                 "成绩缺失表示材料中没有记录，不表示零分。")
    artifacts = project.get("artifacts") or []
    if artifacts:
        lines.append("主要交付文件（知识库整理机器上的相对路径）：" + "；".join(artifacts) + "。")
    lines.append("")
    return "\n".join(lines)


def import_projects(kb_dir, projects, ids, out_dir=EXPERIENCES, dry_run=False):
    written, source_ids = [], set()
    for pid in ids:
        if pid not in projects:
            raise ValueError(f"知识库里没有 {pid}")
        project = projects[pid]
        stem, title, resume_line = PROFILES.get(pid, (
            f"{slug(project.get('course') or 'uwa')}_{slug(project['title_zh'])[:40]}", project["title_zh"], ""))
        text = render(project, stem, title, resume_line, kb_dir)
        for section in project.get("sections", []):
            for fact in section.get("facts", []):
                source_ids.update(s["source_id"] for s in fact.get("sources", []))
        path = Path(out_dir) / f"{stem}.md"
        if not dry_run:
            path.write_text(text, encoding="utf-8")
        written.append(path)
    if not dry_run:
        copy_evidence(kb_dir, source_ids)
        for name in ("来源索引.md", "project_index.csv", "metrics.jsonl", "RAG使用说明.md", "覆盖范围与待补充.md"):
            src = kb_dir / name
            if src.exists():
                shutil.copy2(src, SOURCES / name)
    return written, sorted(source_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description="从 UWA 知识库导入项目经历")
    parser.add_argument("--kb", required=True, type=Path)
    parser.add_argument("--projects", help="逗号分隔的 project_id；默认导入 PROFILES 里的全部")
    parser.add_argument("--list", action="store_true", help="只列出知识库项目")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    kb_dir, projects = load_kb(args.kb)
    if args.list:
        for pid, p in projects.items():
            mark = "*" if pid in PROFILES else " "
            print(f"{mark} {pid} | {p.get('course')} | {p['title_zh']} | {p.get('ownership')}")
        return 0
    ids = [x.strip() for x in args.projects.split(",")] if args.projects else list(PROFILES)
    written, sids = import_projects(kb_dir, projects, ids, dry_run=args.dry_run)
    for path in written:
        print(("将写入：" if args.dry_run else "已写入：") + str(path.relative_to(PROJECT_ROOT)))
    print(f"引用来源 {len(sids)} 个" + ("" if args.dry_run else f"，已复制到 {SOURCES.relative_to(PROJECT_ROOT)}/evidence"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
