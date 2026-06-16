#!/usr/bin/env python3
"""
pipeline_code.py — Code Atlas: 把一个文件夹当作项目工作区进行符号级分析。

像 Source Insight 一样，递归扫描源码、提取所有符号与关系，构建完整的
代码知识图谱，并输出**结构化文本**供 LLM 直接消费。

特点：
    - 零 LLM、零依赖（纯标准库）：确定性、离线、秒级完成。
    - 多语言：Python 精确解析；MATLAB/Octave 文法解析；JS/TS/Java/C/C++/C#/Go/Rust/... 启发式解析。

用法：
    python tools/pipeline_code.py [FOLDER] [选项]

    python tools/pipeline_code.py .                       # 分析当前目录
    python tools/pipeline_code.py /path/to/project        # 分析指定文件夹
    python tools/pipeline_code.py . --open                # 分析并打开图谱
    python tools/pipeline_code.py . --out .codemap        # 自定义输出目录
    python tools/pipeline_code.py . --stdout              # 把文本地图打到 stdout
    python tools/pipeline_code.py . --no-html             # 跳过 HTML 图谱
    python tools/pipeline_code.py . --max-files 2000      # 提高/降低文件上限

输出（默认写到 <FOLDER>/codemap/ 或 --out 指定目录）：
    codemap.md     — 结构化文本地图（为 LLM 优化，主产物）
    symbols.json   — 完整机器可读符号数据库
    codemap.html   — 自包含 vis.js 交互式符号图谱
"""

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

# 让脚本能 import core.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.code_analyzer import analyze_workspace, DEFAULT_MAX_FILES, DEFAULT_MAX_FILE_SIZE
from core.code_report import render_markdown, render_html


def main():
    parser = argparse.ArgumentParser(
        description="Code Atlas — 工作区符号知识图谱（Source Insight 风格）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", nargs="?", default=".",
                        help="要分析的项目文件夹（默认：当前目录）")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认：<FOLDER>/codemap）")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                        help=f"最多分析的文件数（默认 {DEFAULT_MAX_FILES}）")
    parser.add_argument("--max-file-size", type=int, default=DEFAULT_MAX_FILE_SIZE,
                        help="单文件最大字节数，超过则跳过")
    parser.add_argument("--no-html", action="store_true", help="不生成 codemap.html")
    parser.add_argument("--no-json", action="store_true", help="不生成 symbols.json")
    parser.add_argument("--stdout", action="store_true",
                        help="把文本地图打印到标准输出（便于直接喂给 LLM）")
    parser.add_argument("--open", action="store_true", help="生成后在浏览器打开图谱")
    args = parser.parse_args()

    root = Path(args.folder).resolve()
    if not root.is_dir():
        print(f"❌ 不是有效文件夹: {root}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out).resolve() if args.out else (root / "codemap")
    out_name = out_dir.name

    t0 = time.time()
    print(f"🔍 正在分析工作区: {root}")
    analysis = analyze_workspace(
        root,
        max_files=args.max_files,
        max_file_size=args.max_file_size,
        output_dirname=out_name,
    )
    dt = time.time() - t0

    if not analysis.files:
        print("⚠️  未发现可分析的源代码文件。")
        print("   支持的语言：Python / MATLAB / Octave / JS / TS / Java / C / C++ / C# / Go / Rust / Ruby / PHP / ...")
        sys.exit(0)

    # 渲染文本地图
    markdown = render_markdown(analysis)

    if args.stdout:
        print(markdown)

    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "codemap.md"
    md_path.write_text(markdown, encoding="utf-8")

    if not args.no_json:
        json_path = out_dir / "symbols.json"
        json_path.write_text(
            json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")

    html_path = out_dir / "codemap.html"
    if not args.no_html:
        html_path.write_text(render_html(analysis), encoding="utf-8")

    # 摘要
    counts = analysis.symbol_counts()
    print()
    print("📊 Code Atlas 分析完成")
    print(f"   文件: {len(analysis.files)} 个 · 代码行: ~{analysis.total_loc():,} LOC")
    print(f"   符号: {len(analysis.symbols)} 个 · 关系: {len(analysis.edges)} 条")
    if analysis.lang_stats:
        langs = " · ".join(f"{k} {v['files']}" for k, v in list(analysis.lang_stats.items())[:6])
        print(f"   语言: {langs}")
    if counts:
        kind_line = " · ".join(f"{k} {v}" for k, v in list(counts.items())[:8])
        print(f"   构成: {kind_line}")
    if analysis.truncated:
        print(f"   ⚠️  已达文件上限 {args.max_files}，结果被截断（--max-files 可调高）")
    print()
    print("✅ 输出:")
    print(f"   📄 {md_path}   ← 结构化文本地图（喂给 LLM）")
    if not args.no_json:
        print(f"   🧩 {out_dir / 'symbols.json'}   ← 完整符号数据库")
    if not args.no_html:
        print(f"   🌐 {html_path}   ← 交互式符号图谱")
    print(f"\n⏱️  耗时 {dt:.2f}s")

    if args.open and not args.no_html:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
