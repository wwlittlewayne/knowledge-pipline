"""
Unit tests for the Code Atlas engine (core.code_analyzer + core.code_report).
"""

import json
from pathlib import Path

import pytest

from core.code_analyzer import analyze_workspace, parse_python, _scan_regex, _LANG_SPECS
from core.code_report import render_markdown, render_html


# ──────────────────────────────────────────────────────────────────────────
# Python (ast) parsing
# ──────────────────────────────────────────────────────────────────────────

PY_SAMPLE = '''\
"""Module docstring."""
import os
from collections import defaultdict

MAX_SIZE = 100

class Base:
    """Base class."""
    shared = 1

    def hello(self):
        return helper()

class Worker(Base):
    def __init__(self, name):
        self.name = name
        self.count = 0

    def run(self):
        return self.name

def helper():
    return MAX_SIZE
'''


def test_python_symbols():
    syms, imports, doc = parse_python(PY_SAMPLE, "sample.py")
    by_name = {s.name: s for s in syms}

    assert doc == "Module docstring."
    assert "os" in imports
    assert "collections" in imports

    assert by_name["MAX_SIZE"].kind == "constant"
    assert by_name["Base"].kind == "class"
    assert by_name["Worker"].kind == "class"
    assert by_name["Worker"].bases == ["Base"]

    assert by_name["run"].kind == "method"
    assert by_name["run"].parent == "Worker"
    assert by_name["helper"].kind == "function"
    assert by_name["helper"].parent == ""


def test_python_self_fields_dedup():
    syms, _, _ = parse_python(PY_SAMPLE, "sample.py")
    fields = [s for s in syms if s.kind == "field"]
    names = sorted(s.name for s in fields)
    # self.name, self.count, and class var `shared`
    assert "name" in names
    assert "count" in names
    assert "shared" in names
    # local variables inside methods must NOT be captured
    assert all(s.name not in ("return",) for s in syms)


def test_python_no_local_variable_noise():
    code = "def f():\n    temp = 1\n    other = 2\n    return temp + other\n"
    syms, _, _ = parse_python(code, "f.py")
    names = {s.name for s in syms}
    assert "f" in names
    assert "temp" not in names      # locals are not symbols
    assert "other" not in names


def test_python_call_graph():
    syms, _, _ = parse_python(PY_SAMPLE, "sample.py")
    hello = next(s for s in syms if s.name == "hello")
    assert "helper" in hello.calls


# ──────────────────────────────────────────────────────────────────────────
# Heuristic regex parsers (JS / Java / Go / C)
# ──────────────────────────────────────────────────────────────────────────

def test_javascript_parsing():
    js = (
        "import { x } from './m';\n"
        "export class Animal extends Base {\n"
        "  constructor(n) { this.n = n; }\n"
        "  speak() { return this.n; }\n"
        "}\n"
        "export function makeNoise(z) { return z; }\n"
        "const handler = async (req) => req;\n"
        "export const MAX_RETRIES = 5;\n"
    )
    syms = _scan_regex(js, "a.js", _LANG_SPECS["JavaScript"])
    by_name = {s.name: s for s in syms}
    assert by_name["Animal"].kind == "class"
    assert by_name["Animal"].bases == ["Base"]
    assert by_name["speak"].kind == "method"
    assert by_name["speak"].parent == "Animal"
    assert by_name["makeNoise"].kind == "function"
    assert by_name["handler"].kind == "function"
    assert by_name["MAX_RETRIES"].kind == "constant"


def test_typescript_parsing():
    ts = (
        "import { Server } from './server';\n"
        "export interface Config { port: number; }\n"
        "export class App extends Base implements Config {\n"
        "  start(): void { this.run(); }\n"
        "  async fetch(id: number): Promise<Config> { return null; }\n"
        "  private run() {}\n"
        "}\n"
        "export type Handler = () => void;\n"
    )
    syms = _scan_regex(ts, "app.ts", _LANG_SPECS["TypeScript"])
    by_name = {s.name: s for s in syms}
    assert by_name["Config"].kind == "interface"
    assert by_name["App"].kind == "class"
    # methods with TS return-type annotations must be captured with the right parent
    assert by_name["start"].kind == "method" and by_name["start"].parent == "App"
    assert by_name["fetch"].kind == "method" and by_name["fetch"].parent == "App"
    assert by_name["Handler"].kind == "type"


def test_java_parsing():
    java = (
        "package com.example;\n"
        "import java.util.List;\n"
        "public class Service extends Base implements Runnable {\n"
        "    public void run() { return; }\n"
        "    private int compute(int a) { return a; }\n"
        "}\n"
    )
    syms = _scan_regex(java, "S.java", _LANG_SPECS["Java"])
    by_name = {s.name: s for s in syms}
    assert by_name["Service"].kind == "class"
    assert "Base" in by_name["Service"].bases
    assert by_name["run"].kind == "method"
    assert by_name["run"].parent == "Service"
    assert by_name["compute"].parent == "Service"


def test_go_parsing():
    go = (
        "package main\n"
        "type Server struct {\n}\n"
        "func (s *Server) Start() {}\n"
        "func Helper(x int) int { return x }\n"
    )
    syms = _scan_regex(go, "m.go", _LANG_SPECS["Go"])
    by_name = {s.name: s for s in syms}
    assert by_name["Server"].kind == "struct"
    assert "Start" in by_name
    assert "Helper" in by_name


def test_c_parsing():
    c = (
        "#include <stdio.h>\n"
        "#define MAX 10\n"
        "struct Point { int x; int y; };\n"
        "int add(int a, int b) {\n    return a + b;\n}\n"
    )
    syms = _scan_regex(c, "m.c", _LANG_SPECS["C"])
    kinds = {s.name: s.kind for s in syms}
    assert kinds.get("MAX") == "macro"
    assert kinds.get("Point") == "struct"
    assert kinds.get("add") == "function"


# ──────────────────────────────────────────────────────────────────────────
# Full workspace analysis
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_workspace(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "core.py").write_text(
        "class Engine:\n    def start(self):\n        return 1\n")
    (tmp_path / "app.py").write_text(
        "from pkg.core import Engine\n"
        "def main():\n    e = Engine()\n    return e.start()\n")
    # noise dirs that must be skipped
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("function ignored(){}\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.py").write_text("def nope(): pass\n")
    return tmp_path


def test_analyze_workspace_basic(sample_workspace):
    a = analyze_workspace(sample_workspace)
    paths = {f.path for f in a.files}
    assert "app.py" in paths
    assert "pkg/core.py" in paths
    # skipped dirs
    assert not any("node_modules" in p for p in paths)
    assert not any(".git" in p for p in paths)


def test_analyze_workspace_dependency_edge(sample_workspace):
    a = analyze_workspace(sample_workspace)
    import_edges = [(e["from"], e["to"]) for e in a.edges if e["type"] == "imports"]
    assert ("file:app.py", "file:pkg/core.py") in import_edges


def test_analyze_workspace_references(sample_workspace):
    a = analyze_workspace(sample_workspace)
    engine = next(s for s in a.symbols if s.name == "Engine" and s.kind == "class")
    # Engine is referenced in app.py (import + instantiation)
    assert engine.ref_count >= 1


def test_analyze_workspace_output_dir_skipped(tmp_path):
    (tmp_path / "x.py").write_text("def a(): pass\n")
    out = tmp_path / "codemap"
    out.mkdir()
    (out / "old.py").write_text("def should_be_ignored(): pass\n")
    a = analyze_workspace(tmp_path, output_dirname="codemap")
    names = {s.name for s in a.symbols}
    assert "a" in names
    assert "should_be_ignored" not in names


def test_not_a_directory(tmp_path):
    f = tmp_path / "file.py"
    f.write_text("x = 1\n")
    with pytest.raises(NotADirectoryError):
        analyze_workspace(f)


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────

def test_render_markdown_sections(sample_workspace):
    a = analyze_workspace(sample_workspace)
    md = render_markdown(a)
    assert "# 🗺️ Code Atlas" in md
    assert "## 1. 概览" in md
    assert "## 2. 目录结构" in md
    assert "## 3. 文件符号大纲" in md
    assert "## 4. 符号索引" in md
    assert "Engine" in md


def test_render_html_selfcontained(sample_workspace):
    a = analyze_workspace(sample_workspace)
    html = render_html(a)
    assert html.startswith("<!DOCTYPE html>")
    assert "vis-network" in html
    assert "Code Atlas" in html


def test_to_dict_serializable(sample_workspace):
    a = analyze_workspace(sample_workspace)
    d = a.to_dict()
    # must be JSON-serializable
    s = json.dumps(d, ensure_ascii=False)
    assert "symbols" in d
    assert "edges" in d
    assert d["stats"]["files"] >= 2
