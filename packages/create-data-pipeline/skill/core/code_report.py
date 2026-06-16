"""
Code Atlas — 报告渲染器

把 ``code_analyzer.Analysis`` 渲染成两种产物：

1. ``render_markdown`` → 结构化文本（codemap.md），为 LLM 优化，token 紧凑，
   包含：项目概览、目录树、逐文件符号大纲、符号索引、依赖图、类继承、调用图。
2. ``render_html``     → 自包含 vis.js 交互式符号知识图谱（codemap.html）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from core.code_analyzer import Analysis, Symbol


# 符号类型 → emoji / 中文名（用于文本可读性）
KIND_LABEL = {
    "class": "🟦 class", "interface": "🟪 interface", "struct": "🟫 struct",
    "enum": "🟧 enum", "trait": "🟨 trait", "function": "🟩 func",
    "method": "▫️ method", "constant": "🔶 const", "field": "▪️ field",
    "type": "🔷 type", "macro": "⬛ macro", "namespace": "📦 ns",
    "object": "🟢 object", "module": "📦 module", "union": "🟫 union",
    "variable": "▪️ var", "impl": "⚙️ impl",
}

# 图谱节点颜色（按符号类型）
KIND_COLORS = {
    "file": "#607D8B",
    "class": "#2196F3", "interface": "#9C27B0", "struct": "#795548",
    "enum": "#FF9800", "trait": "#FFC107", "function": "#4CAF50",
    "method": "#8BC34A", "constant": "#E91E63", "field": "#BDBDBD",
    "type": "#00BCD4", "macro": "#455A64", "namespace": "#3F51B5",
    "object": "#009688", "module": "#3F51B5", "union": "#795548",
    "impl": "#FF5722", "variable": "#9E9E9E",
}

EDGE_LABEL = {
    "contains": "包含", "imports": "导入", "inherits": "继承", "calls": "调用",
}


# ──────────────────────────────────────────────────────────────────────────
# Markdown / 文本报告
# ──────────────────────────────────────────────────────────────────────────

def _human_kind(kind: str) -> str:
    return KIND_LABEL.get(kind, kind)


def _build_tree(paths: List[str]) -> dict:
    tree: dict = {}
    for p in paths:
        parts = p.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None
    return tree


def _render_tree(tree: dict, file_sym_count: Dict[str, int], prefix: str = "",
                 path_prefix: str = "", lines: List[str] = None) -> List[str]:
    if lines is None:
        lines = []
    entries = sorted(tree.items(), key=lambda kv: (kv[1] is None, kv[0]))
    for i, (name, child) in enumerate(entries):
        last = i == len(entries) - 1
        connector = "└── " if last else "├── "
        if child is None:  # 文件
            full = path_prefix + name
            n = file_sym_count.get(full, 0)
            suffix = f"  ({n} symbols)" if n else ""
            lines.append(f"{prefix}{connector}{name}{suffix}")
        else:  # 目录
            lines.append(f"{prefix}{connector}{name}")
            ext = "    " if last else "│   "
            _render_tree(child, file_sym_count, prefix + ext, path_prefix + name, lines)
    return lines


def render_markdown(analysis: Analysis, *, max_files_outline: int = 400,
                    max_symbols_per_file: int = 60) -> str:
    a = analysis
    counts = a.symbol_counts()
    out: List[str] = []

    # ── 标题 ──
    out.append(f"# 🗺️ Code Atlas — {a.name}")
    out.append("")
    out.append(f"> 由 Knowledge Pipeline `/pipeline-code` 生成 · {a.generated}")
    out.append(f"> 工作区根目录：`{a.root}`")
    out.append("")
    out.append("本文件是该代码库的**结构化符号地图**，供 LLM 直接阅读理解项目。"
               "包含目录结构、逐文件符号大纲、符号索引、模块依赖、类继承与调用关系。")
    out.append("")

    # ── 1. 概览 ──
    out.append("## 1. 概览 (Overview)")
    out.append("")
    out.append(f"- **文件**：{len(a.files)} 个已分析"
               + (f"（共扫描 {a.scanned}，跳过 {a.skipped}）" if a.skipped else ""))
    out.append(f"- **代码行**：~{a.total_loc():,} LOC")
    out.append(f"- **符号总数**：{len(a.symbols)}")
    out.append(f"- **关系边**：{len(a.edges)}")
    if a.truncated:
        out.append(f"- ⚠️ 已达文件上限，结果被截断（可用 `--max-files` 提高上限）")
    out.append("")
    if counts:
        kinds_line = " · ".join(f"{_human_kind(k)} {v}" for k, v in counts.items())
        out.append(f"**符号构成**：{kinds_line}")
        out.append("")
    if a.lang_stats:
        out.append("| 语言 | 文件 | 代码行 | 符号 |")
        out.append("|------|------|--------|------|")
        for lang, st in a.lang_stats.items():
            out.append(f"| {lang} | {st['files']} | {st['loc']:,} | {st['symbols']} |")
        out.append("")

    # ── 2. 目录结构 ──
    file_sym_count: Dict[str, int] = defaultdict(int)
    for s in a.symbols:
        file_sym_count[s.file] += 1
    out.append("## 2. 目录结构 (Structure)")
    out.append("")
    out.append("```")
    out.append(f"{a.name}/")
    tree = _build_tree([fi.path for fi in a.files])
    out.extend(_render_tree(tree, file_sym_count))
    out.append("```")
    out.append("")

    # ── 3. 逐文件符号大纲 ──
    out.append("## 3. 文件符号大纲 (File Outlines)")
    out.append("")
    syms_by_file: Dict[str, List[Symbol]] = defaultdict(list)
    for s in a.symbols:
        syms_by_file[s.file].append(s)

    files_sorted = sorted(a.files, key=lambda f: (-file_sym_count.get(f.path, 0), f.path))
    shown = 0
    for fi in files_sorted:
        syms = syms_by_file.get(fi.path, [])
        if shown >= max_files_outline:
            out.append(f"_… 其余 {len(files_sorted) - shown} 个文件略（详见 symbols.json）_")
            out.append("")
            break
        shown += 1
        out.append(f"### `{fi.path}`")
        meta = f"{fi.language} · {fi.loc} LOC"
        if fi.deps:
            meta += f" · 依赖 {len(fi.deps)} 个内部文件"
        out.append(f"_{meta}_" + (f" — {fi.doc}" if fi.doc else ""))
        out.append("")
        if fi.imports:
            imp_preview = ", ".join(fi.imports[:12])
            if len(fi.imports) > 12:
                imp_preview += f" … (+{len(fi.imports) - 12})"
            out.append(f"- **imports**: {imp_preview}")

        # 组织：顶层符号 + 其子成员缩进
        top = [s for s in syms if not s.parent]
        children: Dict[str, List[Symbol]] = defaultdict(list)
        for s in syms:
            if s.parent:
                children[s.parent].append(s)

        emitted = 0
        for s in top:
            if emitted >= max_symbols_per_file:
                out.append(f"  - _… 其余 {len(syms) - emitted} 个符号略_")
                break
            ref = f"  ·  ⮐{s.ref_count}" if s.ref_count else ""
            base = f" : {', '.join(s.bases)}" if s.bases else ""
            out.append(f"- {_human_kind(s.kind)} **{s.name}**{base}  `L{s.line}`{ref}")
            emitted += 1
            for c in children.get(s.name, []):
                if emitted >= max_symbols_per_file:
                    break
                out.append(f"    - {_human_kind(c.kind)} {c.name}  `L{c.line}`")
                emitted += 1
        out.append("")

    # ── 4. 符号索引 ──
    out.append("## 4. 符号索引 (Symbol Index)")
    out.append("")
    out.append("> 按名称排序。格式：`名称` — 类型 · 位置 · 引用数")
    out.append("")
    indexable = sorted(
        [s for s in a.symbols if s.kind in (
            "class", "interface", "struct", "enum", "trait", "function",
            "method", "constant", "type", "macro", "namespace", "object")],
        key=lambda s: s.name.lower())
    for s in indexable[:1500]:
        loc = f"{s.file}:{s.line}"
        ref = f" · ⮐{s.ref_count}" if s.ref_count else ""
        parent = f" (in {s.parent})" if s.parent else ""
        out.append(f"- `{s.name}`{parent} — {s.kind} · {loc}{ref}")
    if len(indexable) > 1500:
        out.append(f"- _… 其余 {len(indexable) - 1500} 个符号见 symbols.json_")
    out.append("")

    # ── 5. 模块依赖图 ──
    dep_edges = [e for e in a.edges if e["type"] == "imports"]
    if dep_edges:
        out.append("## 5. 模块依赖图 (Module Dependencies)")
        out.append("")
        out.append("> `A → B` 表示 A 导入/依赖 B（仅内部文件）。")
        out.append("")
        deps_by_file: Dict[str, List[str]] = defaultdict(list)
        for e in dep_edges:
            deps_by_file[e["from"][5:]].append(e["to"][5:])
        for frm in sorted(deps_by_file):
            tos = deps_by_file[frm]
            preview = ", ".join(tos[:8]) + (f" … (+{len(tos) - 8})" if len(tos) > 8 else "")
            out.append(f"- `{frm}` → {preview}")
        out.append("")
        # 最被依赖的文件
        fan_in: Dict[str, int] = defaultdict(int)
        for e in dep_edges:
            fan_in[e["to"][5:]] += 1
        hot = sorted(fan_in.items(), key=lambda kv: -kv[1])[:10]
        if hot:
            out.append("**核心模块**（被依赖最多）：")
            for path, n in hot:
                out.append(f"- `{path}` ← {n} 个文件")
            out.append("")

    # ── 6. 类继承 ──
    inherit_edges = [e for e in a.edges if e["type"] == "inherits"]
    id_to_sym = {s.node_id: s for s in a.symbols}
    if inherit_edges:
        out.append("## 6. 类继承关系 (Class Hierarchy)")
        out.append("")
        for e in inherit_edges[:200]:
            src = id_to_sym.get(e["from"])
            dst = id_to_sym.get(e["to"])
            if src and dst:
                out.append(f"- `{src.name}` ⟶ `{dst.name}`  ({src.file}:{src.line})")
        out.append("")

    # ── 7. 调用图 ──
    call_edges = [e for e in a.edges if e["type"] == "calls"]
    if call_edges:
        out.append("## 7. 调用图 (Call Graph)")
        out.append("")
        out.append("> `A → B` 表示函数 A 调用函数 B（Python 精确捕获）。")
        out.append("")
        calls_by: Dict[str, List[str]] = defaultdict(list)
        for e in call_edges:
            src = id_to_sym.get(e["from"])
            dst = id_to_sym.get(e["to"])
            if src and dst:
                calls_by[src.qualname].append(dst.name)
        for frm in sorted(calls_by)[:150]:
            tos = sorted(set(calls_by[frm]))
            preview = ", ".join(tos[:10]) + (f" … (+{len(tos) - 10})" if len(tos) > 10 else "")
            out.append(f"- `{frm}()` → {preview}")
        out.append("")

    # ── 8. 关键符号 ──
    hot_syms = sorted([s for s in a.symbols if s.ref_count > 0],
                      key=lambda s: -s.ref_count)[:25]
    if hot_syms:
        out.append("## 8. 高频引用符号 (Most Referenced)")
        out.append("")
        out.append("> 跨文件被引用最多的符号 — 通常是项目的核心抽象。")
        out.append("")
        for s in hot_syms:
            out.append(f"- `{s.name}` ({s.kind}) — ⮐{s.ref_count} 次引用 · {s.file}:{s.line}")
        out.append("")

    out.append("---")
    out.append(f"_完整机器可读数据见 `symbols.json` · 交互式图谱见 `codemap.html`_")
    out.append("")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────
# HTML 交互式图谱
# ──────────────────────────────────────────────────────────────────────────

def _graph_payload(analysis: Analysis, max_nodes: int = 1200) -> dict:
    """构建用于 vis.js 的 nodes/edges（节点过多时降级为文件级）。"""
    a = analysis
    id_to_sym = {s.node_id: s for s in a.symbols}

    # 节点重要性：被引用数 + 是否为容器类型
    def importance(s: Symbol) -> int:
        base = s.ref_count
        if s.kind in ("class", "interface", "struct", "trait", "enum"):
            base += 5
        if s.kind in ("function",):
            base += 1
        return base

    file_level = len(a.symbols) > max_nodes
    nodes = []
    node_ids = set()

    # 文件节点始终包含
    for fi in a.files:
        nid = f"file:{fi.path}"
        node_ids.add(nid)
        nodes.append({
            "id": nid, "label": fi.path.split("/")[-1], "title": fi.path,
            "type": "file", "color": KIND_COLORS["file"],
            "shape": "box", "value": max(1, fi.loc // 20),
        })

    if not file_level:
        # 符号级：纳入全部（或重要符号）
        ranked = sorted(a.symbols, key=importance, reverse=True)
        budget = max_nodes - len(nodes)
        for s in ranked[:max(0, budget)]:
            nid = s.node_id
            if nid in node_ids:
                continue
            node_ids.add(nid)
            nodes.append({
                "id": nid, "label": s.name,
                "title": f"{s.kind} {s.qualname} — {s.file}:{s.line}"
                         + (f" · ⮐{s.ref_count}" if s.ref_count else ""),
                "type": s.kind, "color": KIND_COLORS.get(s.kind, "#9E9E9E"),
                "shape": "dot", "value": 1 + s.ref_count,
            })

    edges = []
    for e in a.edges:
        frm, to, et = e["from"], e["to"], e["type"]
        if file_level or et == "imports":
            # 把符号端点折叠到所属文件
            if frm not in node_ids and frm.startswith("sym:"):
                s = id_to_sym.get(frm)
                frm = f"file:{s.file}" if s else frm
            if to not in node_ids and to.startswith("sym:"):
                s = id_to_sym.get(to)
                to = f"file:{s.file}" if s else to
        if frm in node_ids and to in node_ids and frm != to:
            edges.append({"from": frm, "to": to, "type": et,
                          "color": "#999" if et == "imports" else "#bbb"})

    # 去重边
    uniq = {}
    for e in edges:
        uniq[(e["from"], e["to"], e["type"])] = e
    return {"nodes": nodes, "edges": list(uniq.values()), "file_level": file_level}


def render_html(analysis: Analysis, max_nodes: int = 1200) -> str:
    payload = _graph_payload(analysis, max_nodes)
    nodes_json = json.dumps(payload["nodes"], ensure_ascii=False)
    edges_json = json.dumps(payload["edges"], ensure_ascii=False)
    a = analysis

    legend = "".join(
        f'<span style="background:{KIND_COLORS[k]};padding:2px 7px;margin:2px;'
        f'border-radius:3px;font-size:11px;color:#fff">{k}</span>'
        for k in ("file", "class", "interface", "function", "method", "constant", "enum")
    )
    mode = "文件级依赖图（符号过多已降级）" if payload["file_level"] else "符号级知识图谱"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Code Atlas — {a.name}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin:0; background:#0f1117; font-family:"Segoe UI","Microsoft YaHei",sans-serif; color:#e6e6e6; }}
  #graph {{ width:100vw; height:100vh; }}
  #panel {{ position:fixed; top:12px; left:12px; background:rgba(20,22,30,.92);
    padding:14px 16px; border-radius:10px; z-index:10; max-width:300px; border:1px solid #2a2d3a; }}
  #panel h3 {{ margin:0 0 8px; font-size:15px; }}
  #search {{ width:100%; box-sizing:border-box; padding:6px; margin:6px 0 10px; background:#1a1d27;
    color:#eee; border:1px solid #333; border-radius:5px; }}
  #stats {{ position:fixed; top:12px; right:12px; background:rgba(20,22,30,.92); padding:10px 14px;
    border-radius:10px; font-size:12px; border:1px solid #2a2d3a; }}
  #info {{ position:fixed; bottom:12px; left:12px; background:rgba(20,22,30,.95); padding:12px 14px;
    border-radius:10px; z-index:10; max-width:420px; display:none; border:1px solid #2a2d3a; font-size:13px; }}
  .muted {{ color:#9aa0b0; font-size:11px; }}
</style>
</head>
<body>
<div id="panel">
  <h3>🗺️ Code Atlas — {a.name}</h3>
  <input id="search" placeholder="搜索符号 / 文件…" oninput="doSearch(this.value)">
  <div>{legend}</div>
  <div class="muted" style="margin-top:8px">视图：{mode}</div>
  <div class="muted">实线=导入 · 细线=包含/继承/调用</div>
</div>
<div id="stats"></div>
<div id="info"></div>
<div id="graph"></div>
<script>
const rawNodes = {nodes_json};
const rawEdges = {edges_json};
const nodes = new vis.DataSet(rawNodes);
const edges = new vis.DataSet(rawEdges.map((e,i) => Object.assign({{id:'e'+i}}, e)));
const container = document.getElementById("graph");
const network = new vis.Network(container, {{nodes, edges}}, {{
  nodes: {{ font:{{color:"#e6e6e6",size:13}}, borderWidth:1.5,
    scaling:{{min:6,max:34}} }},
  edges: {{ width:0.7, color:{{color:"#555",opacity:0.55}}, smooth:{{type:"continuous"}},
    arrows:{{to:{{enabled:true,scaleFactor:0.45}}}} }},
  physics: {{ stabilization:{{iterations:180}},
    barnesHut:{{gravitationalConstant:-9000,springLength:120,avoidOverlap:0.2}} }},
  interaction: {{ hover:true, tooltipDelay:150 }},
}});
document.getElementById("stats").innerHTML =
  "节点 <b>"+nodes.length+"</b> · 边 <b>"+edges.length+"</b>";

network.on("click", p => {{
  const info = document.getElementById("info");
  if (p.nodes.length) {{
    const n = nodes.get(p.nodes[0]);
    info.style.display = "block";
    info.innerHTML = "<b>"+(n.label||n.id)+"</b><br><span class='muted'>"+
      (n.type||"")+"</span><br>"+(n.title||"");
  }} else {{ info.style.display = "none"; }}
}});

function doSearch(q) {{
  const lo = q.toLowerCase();
  nodes.forEach(n => {{
    const hit = !q || (n.label||"").toLowerCase().includes(lo) ||
                (n.title||"").toLowerCase().includes(lo);
    nodes.update({{id:n.id, opacity: hit ? 1 : 0.12}});
  }});
}}
</script>
</body>
</html>"""
