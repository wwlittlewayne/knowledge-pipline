"""
Code Atlas — 工作区代码符号分析引擎 (Source Insight 风格)

把一个文件夹当作"项目工作区"，递归扫描源码，提取所有符号
（类 / 函数 / 方法 / 接口 / 结构体 / 枚举 / 常量 / 宏 / 类型 ...）
以及它们之间的关系（包含 / 导入依赖 / 继承 / 调用 / 引用），
构建一张完整的代码知识图谱。

设计原则：
- **零依赖、零 LLM**：纯标准库，确定性解析，离线即可运行。
- **多语言**：Python 使用 `ast` 精确解析；其余语言使用带花括号深度
  跟踪的启发式正则解析器（C/C++/Java/C#/JS/TS/Go/Rust/...）。
- **可消费**：输出结构化数据，供 LLM 直接阅读或下游程序使用。

入口：``analyze_workspace(root) -> Analysis``
"""

from __future__ import annotations

import ast
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
# 语言识别与扫描配置
# ──────────────────────────────────────────────────────────────────────────

LANG_BY_EXT: Dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin", ".kts": "Kotlin",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cxx": "C++", ".cc": "C++", ".hpp": "C++", ".hh": "C++", ".hxx": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".scala": "Scala",
    ".m": "Objective-C", ".mm": "Objective-C",
    ".dart": "Dart",
    ".lua": "Lua",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".pl": "Perl", ".pm": "Perl",
    ".r": "R",
    ".sql": "SQL",
    ".vue": "Vue", ".svelte": "Svelte",
}

# 默认跳过的目录（依赖、构建产物、版本控制等）
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bower_components", "vendor",
    ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", "dist", "build", "out", "target", ".next",
    ".nuxt", ".svelte-kit", "coverage", ".coverage", ".idea", ".vscode",
    ".gradle", "Pods", ".dart_tool", "bin", "obj", ".terraform",
    "site-packages", ".cache", ".parcel-cache", "__snapshots__",
}

# 不解析符号、但可计入文件统计的二进制 / 资源扩展名
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar", ".jar", ".war",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".class", ".pyc",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".ttf", ".otf", ".woff",
    ".woff2", ".eot", ".bin", ".dat", ".db", ".sqlite", ".lock",
    # MATLAB/Simulink 二进制或非纯文本格式
    ".mlx", ".mat", ".fig", ".slx", ".mdl", ".mexa64", ".mexw64",
}

DEFAULT_MAX_FILES = 5000
DEFAULT_MAX_FILE_SIZE = 1_500_000  # bytes


# ──────────────────────────────────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    """一个代码符号（定义）。"""
    name: str
    kind: str                      # class/function/method/interface/struct/enum/...
    file: str                      # 相对 posix 路径
    line: int
    end_line: int = 0
    signature: str = ""
    parent: str = ""               # 所属类/命名空间名（用于方法/字段）
    bases: List[str] = field(default_factory=list)     # 继承的基类/接口
    decorators: List[str] = field(default_factory=list)
    doc: str = ""                  # 文档字符串首行
    calls: List[str] = field(default_factory=list)     # 调用的其它符号名（call graph）
    language: str = ""
    ref_count: int = 0             # 跨文件被引用次数（启发式）

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def node_id(self) -> str:
        return f"sym:{self.file}#{self.qualname}@{self.line}"

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "signature": self.signature,
        }
        if self.end_line:
            d["end_line"] = self.end_line
        if self.parent:
            d["parent"] = self.parent
        if self.bases:
            d["bases"] = self.bases
        if self.decorators:
            d["decorators"] = self.decorators
        if self.doc:
            d["doc"] = self.doc
        if self.calls:
            d["calls"] = self.calls
        if self.ref_count:
            d["ref_count"] = self.ref_count
        return d


@dataclass
class FileInfo:
    """一个源文件的元信息。"""
    path: str
    language: str
    loc: int = 0
    size: int = 0
    imports: List[str] = field(default_factory=list)   # 原始 import 目标
    deps: List[str] = field(default_factory=list)       # 解析到的内部文件依赖
    doc: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "loc": self.loc,
            "size": self.size,
            "imports": self.imports,
            "deps": self.deps,
            "doc": self.doc,
        }


@dataclass
class Analysis:
    """完整的工作区分析结果。"""
    root: str
    name: str
    generated: str
    files: List[FileInfo] = field(default_factory=list)
    symbols: List[Symbol] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    lang_stats: Dict[str, dict] = field(default_factory=dict)
    scanned: int = 0
    skipped: int = 0
    truncated: bool = False

    def symbol_counts(self) -> Dict[str, int]:
        c: Counter = Counter()
        for s in self.symbols:
            c[s.kind] += 1
        return dict(c.most_common())

    def total_loc(self) -> int:
        return sum(f.loc for f in self.files)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "name": self.name,
            "generated": self.generated,
            "stats": {
                "files": len(self.files),
                "scanned": self.scanned,
                "skipped": self.skipped,
                "truncated": self.truncated,
                "loc": self.total_loc(),
                "symbols": len(self.symbols),
                "symbol_kinds": self.symbol_counts(),
                "languages": self.lang_stats,
                "edges": len(self.edges),
            },
            "files": [f.to_dict() for f in self.files],
            "symbols": [s.to_dict() for s in self.symbols],
            "edges": self.edges,
        }


# ──────────────────────────────────────────────────────────────────────────
# 文本工具
# ──────────────────────────────────────────────────────────────────────────

_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`')
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _blank_strings(line: str) -> str:
    """把字符串字面量替换成空串，降低对花括号 / 关键字匹配的干扰。"""
    return _STRING_RE.sub('""', line)


def _count_loc(content: str) -> int:
    return sum(1 for ln in content.splitlines() if ln.strip())


# 各语言的控制流关键字（避免被误判为函数名）
_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "else", "do",
    "function", "func", "class", "struct", "interface", "enum", "new",
    "typeof", "instanceof", "in", "of", "case", "default", "break",
    "continue", "throw", "try", "finally", "await", "async", "yield",
    "public", "private", "protected", "static", "final", "const", "let",
    "var", "void", "int", "string", "bool", "true", "false", "null",
    "super", "this", "self", "sizeof", "delete", "with", "use", "using",
    "namespace", "import", "export", "from", "package", "module", "def",
    "and", "or", "not", "is", "lambda", "pass", "raise", "assert",
}


# ──────────────────────────────────────────────────────────────────────────
# Python 解析器（基于 ast，精确）
# ──────────────────────────────────────────────────────────────────────────

def _py_signature(node) -> str:
    """从 ast 函数节点重建签名。"""
    try:
        args = []
        a = node.args
        posonly = getattr(a, "posonlyargs", [])
        for arg in posonly:
            args.append(arg.arg)
        if posonly:
            args.append("/")
        for arg in a.args:
            args.append(arg.arg)
        if a.vararg:
            args.append("*" + a.vararg.arg)
        elif a.kwonlyargs:
            args.append("*")
        for arg in a.kwonlyargs:
            args.append(arg.arg)
        if a.kwarg:
            args.append("**" + a.kwarg.arg)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"
    except Exception:
        return f"def {getattr(node, 'name', '?')}(...)"


def _py_decorator_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _py_decorator_name(node.func)
    return ""


def _py_calls(node) -> List[str]:
    """收集函数体内调用的名字（用于调用图）。"""
    names: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.append(f.id)
            elif isinstance(f, ast.Attribute):
                names.append(f.attr)
    # 去重保序
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def parse_python(content: str, rel: str) -> Tuple[List[Symbol], List[str], str]:
    """用 ast 解析 Python，返回 (symbols, imports, module_doc)。"""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # py2 或语法错误 → 退回正则解析器
        spec = _LANG_SPECS["Python-regex"]
        syms = _scan_regex(content, rel, spec)
        return syms, _regex_imports(content, spec), ""

    symbols: List[Symbol] = []
    imports: List[str] = []
    module_doc = (ast.get_docstring(tree) or "").strip().split("\n")[0]
    seen_fields = set()   # (class, attr) 去重，避免 self.x 在多个方法中重复

    def collect_self_attrs(func_node, class_name: str):
        """从方法体收集 self.<attr> = ... 成员变量（去重）。"""
        for sub in ast.walk(func_node):
            if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for t in targets:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == "self"):
                        key = (class_name, t.attr)
                        if key not in seen_fields:
                            seen_fields.add(key)
                            symbols.append(Symbol(
                                name=t.attr, kind="field", file=rel, line=sub.lineno,
                                signature=f"self.{t.attr}", parent=class_name, language="Python"))

    def handle_body(body, parent: str, capture_vars: bool):
        for node in body:
            if isinstance(node, (ast.Import,)):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = ("." * (node.level or 0)) + (node.module or "")
                imports.append(mod)
            elif isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                sym = Symbol(
                    name=node.name, kind="class", file=rel, line=node.lineno,
                    end_line=getattr(node, "end_lineno", 0) or 0,
                    signature=f"class {node.name}" + (f"({', '.join(bases)})" if bases else ""),
                    parent=parent, bases=bases,
                    decorators=[_py_decorator_name(d) for d in node.decorator_list if _py_decorator_name(d)],
                    doc=(ast.get_docstring(node) or "").strip().split("\n")[0],
                    language="Python",
                )
                symbols.append(sym)
                # 类体内部：捕获类变量(字段)，递归方法/嵌套类
                handle_body(node.body, node.name, capture_vars=True)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                sym = Symbol(
                    name=node.name, kind=kind, file=rel, line=node.lineno,
                    end_line=getattr(node, "end_lineno", 0) or 0,
                    signature=_py_signature(node), parent=parent,
                    decorators=[_py_decorator_name(d) for d in node.decorator_list if _py_decorator_name(d)],
                    doc=(ast.get_docstring(node) or "").strip().split("\n")[0],
                    calls=_py_calls(node), language="Python",
                )
                symbols.append(sym)
                # 方法体内收集 self 成员变量；递归捕获嵌套类/函数但**不**记录局部变量
                if parent:
                    collect_self_attrs(node, parent)
                handle_body(node.body, parent, capture_vars=False)
            elif capture_vars and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        nm = t.id
                        if parent:
                            kind = "field"          # 类变量
                        elif nm.isupper() or (nm[:1].isupper() and "_" in nm):
                            kind = "constant"       # 模块级常量
                        else:
                            kind = "variable"
                        if kind in ("constant", "field"):
                            symbols.append(Symbol(
                                name=nm, kind=kind, file=rel, line=node.lineno,
                                signature=nm, parent=parent, language="Python"))

    handle_body(tree.body, "", capture_vars=True)
    return symbols, imports, module_doc


# ──────────────────────────────────────────────────────────────────────────
# MATLAB / GNU Octave 解析器（end 分隔块，遵循 Octave/MATLAB 语法）
# ──────────────────────────────────────────────────────────────────────────
#
# 不依赖 Octave 二进制（其解析器为内部 C++，无公开的"文件→符号/AST"接口，
# 且需执行代码）。此处用纯 Python 实现一个遵循 Octave/MATLAB 文法的解析器：
# 以 end 系列关键字（含 Octave 的 endfunction/endif/...）跟踪块嵌套，
# 识别 function / classdef / methods / properties / events / enumeration，
# 支持 % 与 # 注释、%{ %} / #{ #} 块注释、`...` 续行。

# 仅控制流块（function/classdef/section 单独处理）
_MAT_CTRL_OPENERS = {
    "if", "for", "parfor", "while", "switch", "try",
    "do", "unwind_protect", "spmd",
}
_MAT_CLOSERS = {
    "end", "endif", "endfor", "endparfor", "endwhile", "endswitch",
    "end_try_catch", "endfunction", "endclassdef", "endmethods",
    "endproperties", "endevents", "endenumeration", "until",
    "end_unwind_protect", "endspmd", "endparblock",
}


def _strip_matlab_comment(line: str) -> str:
    """去掉 MATLAB/Octave 行内注释（% 或 #），跳过字符串字面量。"""
    out = []
    in_s = in_d = False
    for c in line:
        if in_s:
            out.append(c)
            if c == "'":
                in_s = False
            continue
        if in_d:
            out.append(c)
            if c == '"':
                in_d = False
            continue
        if c == "'":
            # 区分字符串起始 vs 转置运算符：转置跟在标识符/) /] /. /' 之后
            prev = next((ch for ch in reversed(out) if not ch.isspace()), "")
            if prev and (prev.isalnum() or prev in "_)]}.'"):
                out.append(c)          # 转置
            else:
                in_s = True
                out.append(c)
            continue
        if c == '"':
            in_d = True
            out.append(c)
            continue
        if c in "%#":
            break
        out.append(c)
    return "".join(out)


def _matlab_logical_lines(content: str):
    """产出 (code, lineno)：处理 %{ %} 块注释与 `...` 续行。"""
    lines = content.splitlines()
    i, n = 0, len(lines)
    in_block = False
    while i < n:
        st = lines[i].strip()
        if in_block:
            if st in ("%}", "#}"):
                in_block = False
            i += 1
            continue
        if st in ("%{", "#{"):
            in_block = True
            i += 1
            continue
        start = i + 1
        code = _strip_matlab_comment(lines[i])
        while code.rstrip().endswith("..."):
            code = code.rstrip()[:-3]
            i += 1
            if i < n:
                code += " " + _strip_matlab_comment(lines[i])
            else:
                break
        yield code, start
        i += 1


def _parse_matlab_function(s: str) -> Tuple[str, str]:
    """从 `function [a,b] = name(args)` 提取函数名。"""
    body = s[len("function"):].strip() if s.startswith("function") else s
    paren = body.find("(")
    eq = body.find("=")
    if eq != -1 and (paren == -1 or eq < paren):
        body = body[eq + 1:].strip()
    m = re.match(r"([A-Za-z_]\w*)", body)
    return (m.group(1) if m else ""), s


def parse_matlab(content: str, rel: str) -> Tuple[List[Symbol], List[str], str]:
    """解析 MATLAB / Octave 源文件，返回 (symbols, imports, doc)。"""
    symbols: List[Symbol] = []
    imports: List[str] = []
    doc = ""
    for raw in content.splitlines():
        st = raw.strip()
        if not st:
            continue
        if (st.startswith("%") or st.startswith("#")) and st not in ("%{", "#{"):
            doc = st.lstrip("%#").strip()
        break

    stack: List[dict] = []
    current_class = ""

    def section() -> str:
        for f in reversed(stack):
            if f["kw"] in ("properties", "methods", "events", "enumeration"):
                return f["kw"]
            if f["kw"] == "classdef":
                return ""
        return ""

    for code, lineno in _matlab_logical_lines(content):
        s = code.strip()
        if not s:
            continue
        mkw = re.match(r"([A-Za-z_]\w*)", s)
        kw = mkw.group(1) if mkw else ""
        inside_classdef = any(f["kw"] == "classdef" for f in stack)
        top_is_classdef = bool(stack) and stack[-1]["kw"] == "classdef"

        # 块结束关键字（end / endfunction / until / ...）
        if kw in _MAT_CLOSERS:
            if stack:
                popped = stack.pop()
                if popped["kw"] == "classdef":
                    current_class = ""
            continue

        if kw == "classdef":
            m = re.match(r"classdef\s*(?:\([^)]*\)\s*)?([A-Za-z_]\w*)(.*)", s)
            name = m.group(1) if m else ""
            bases: List[str] = []
            if m:
                lt = re.search(r"<\s*(.+)$", m.group(2))
                if lt:
                    for b in re.split(r"&", lt.group(1)):
                        bm = re.match(r"\s*([A-Za-z_][\w.]*)", b)
                        if bm:
                            bases.append(bm.group(1))
            if name:
                symbols.append(Symbol(name=name, kind="class", file=rel, line=lineno,
                                      signature=s[:140], bases=bases, language="MATLAB"))
                current_class = name
            stack.append({"kw": "classdef", "name": name})
            continue

        # properties/methods/events/enumeration 仅作为 classdef 的直接子块
        if kw in ("methods", "properties", "events", "enumeration") and top_is_classdef:
            stack.append({"kw": kw, "name": ""})
            continue

        if kw == "function":
            name, sig = _parse_matlab_function(s)
            sec = section()
            if current_class and sec in ("methods", ""):
                kind, parent = "method", current_class
            else:
                kind, parent = "function", ""
            if name:
                symbols.append(Symbol(name=name, kind=kind, file=rel, line=lineno,
                                      signature=sig[:140], parent=parent, language="MATLAB"))
            # 仅 classdef 内的函数（方法）保证 end 终止，可安全入栈跟踪
            if inside_classdef:
                stack.append({"kw": "function", "name": name})
            continue

        if kw in _MAT_CTRL_OPENERS:
            stack.append({"kw": kw, "name": ""})
            continue

        if kw == "import":
            im = re.match(r"import\s+([A-Za-z_][\w.]*)", s)
            if im:
                imports.append(im.group(1))
            continue

        # properties / enumeration 块内的成员声明
        sec = section()
        if current_class and sec == "properties":
            pm = re.match(r"([A-Za-z_]\w*)", s)
            if pm and pm.group(1) not in _MAT_CTRL_OPENERS:
                symbols.append(Symbol(name=pm.group(1), kind="field", file=rel, line=lineno,
                                      signature=s[:140], parent=current_class, language="MATLAB"))
        elif current_class and sec == "enumeration":
            em = re.match(r"([A-Za-z_]\w*)", s)
            if em:
                symbols.append(Symbol(name=em.group(1), kind="constant", file=rel, line=lineno,
                                      signature=s[:140], parent=current_class, language="MATLAB"))

    return symbols, imports, doc


def _classify_dot_m(text: str) -> str:
    """`.m` 扩展名同时被 MATLAB 与 Objective-C 使用 — 按内容判别。"""
    head = text[:4000]
    if any(mk in head for mk in ("#import", "@interface", "@implementation",
                                 "@protocol", "@end", "@property")):
        return "Objective-C"
    if re.search(r"^\s*[-+]\s*\([\w\s*]+\)\s*\w+", head, re.M):   # - (void)foo / + (id)bar
        return "Objective-C"
    if re.search(r"^\s*(function\b|classdef\b)", head, re.M) or "endfunction" in head:
        return "MATLAB"
    if re.search(r"^\s*%", head, re.M):
        return "MATLAB"
    if "#include" in head:
        return "Objective-C"
    return "MATLAB"


# ──────────────────────────────────────────────────────────────────────────
# 通用正则解析器（C 家族 / JS / Go / Rust / ...）
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class LangSpec:
    name: str
    line_comments: Tuple[str, ...] = ("//",)
    block_comment: Optional[Tuple[str, str]] = ("/*", "*/")
    containers: List[Tuple] = field(default_factory=list)   # (pattern, kind)
    defs: List[Tuple] = field(default_factory=list)         # (pattern, kind)
    imports: List = field(default_factory=list)             # patterns, group(1)=module
    base_split: Optional[str] = None                        # 关键字提取基类


def _C(p: str):
    return re.compile(p)


# JS / TS 共享
_JS_CONTAINERS = [
    (_C(r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
]
_JS_DEFS = [
    (_C(r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("), "function"),
    (_C(r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^;=]*\)\s*=>"), "function"),
    (_C(r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?function"), "function"),
    (_C(r"(?:export\s+)?(?:const|let|var)\s+([A-Z][A-Z0-9_]{2,})\s*="), "constant"),
    # 类方法（允许可选的 TS 返回类型注解，如 `start(): Promise<void> {`）
    (_C(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|get\s+|set\s+|readonly\s+)*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::\s*[^{;]+?)?\{"), "function"),
]
_TS_CONTAINERS = _JS_CONTAINERS + [
    (_C(r"(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)"), "interface"),
    (_C(r"(?:export\s+)?(?:declare\s+)?enum\s+([A-Za-z_$][\w$]*)"), "enum"),
    (_C(r"(?:export\s+)?namespace\s+([A-Za-z_$][\w$]*)"), "namespace"),
]
_TS_DEFS = _JS_DEFS + [
    (_C(r"(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*="), "type"),
]
_JS_IMPORTS = [
    _C(r"import\s+(?:.+?\s+from\s+)?[\'\"]([^\'\"]+)[\'\"]"),
    _C(r"require\(\s*[\'\"]([^\'\"]+)[\'\"]\s*\)"),
    _C(r"export\s+.+?\s+from\s+[\'\"]([^\'\"]+)[\'\"]"),
]

_LANG_SPECS: Dict[str, LangSpec] = {
    "JavaScript": LangSpec("JavaScript", containers=_JS_CONTAINERS, defs=_JS_DEFS,
                           imports=_JS_IMPORTS, base_split="extends"),
    "TypeScript": LangSpec("TypeScript", containers=_TS_CONTAINERS, defs=_TS_DEFS,
                           imports=_JS_IMPORTS, base_split="extends"),
    "Java": LangSpec(
        "Java",
        containers=[
            (_C(r"(?:public|private|protected|abstract|final|static|\s)*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"(?:public|private|protected|\s)*interface\s+([A-Za-z_]\w*)"), "interface"),
            (_C(r"(?:public|private|protected|\s)*enum\s+([A-Za-z_]\w*)"), "enum"),
        ],
        defs=[
            (_C(r"(?:public|private|protected|static|final|abstract|synchronized|native|\s)+[\w<>\[\],.\s?]+\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:throws[\w,.\s]+)?\{"), "method"),
        ],
        imports=[_C(r"import\s+(?:static\s+)?([\w.]+)")],
        base_split="extends",
    ),
    "C#": LangSpec(
        "C#",
        containers=[
            (_C(r"(?:public|private|protected|internal|abstract|sealed|static|partial|\s)*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"(?:public|private|protected|internal|\s)*interface\s+([A-Za-z_]\w*)"), "interface"),
            (_C(r"(?:public|private|protected|internal|\s)*struct\s+([A-Za-z_]\w*)"), "struct"),
            (_C(r"(?:public|private|protected|internal|\s)*enum\s+([A-Za-z_]\w*)"), "enum"),
            (_C(r"namespace\s+([A-Za-z_][\w.]*)"), "namespace"),
        ],
        defs=[
            (_C(r"(?:public|private|protected|internal|static|virtual|override|async|sealed|\s)+[\w<>\[\],.\s?]+\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*\{"), "method"),
        ],
        imports=[_C(r"using\s+(?:static\s+)?([\w.]+)")],
        base_split=":",
    ),
    "C": LangSpec(
        "C",
        containers=[
            (_C(r"(?:typedef\s+)?struct\s+([A-Za-z_]\w*)"), "struct"),
            (_C(r"(?:typedef\s+)?enum\s+([A-Za-z_]\w*)"), "enum"),
            (_C(r"(?:typedef\s+)?union\s+([A-Za-z_]\w*)"), "union"),
        ],
        defs=[
            (_C(r"^\s*#define\s+([A-Za-z_]\w*)"), "macro"),
            (_C(r"^[\w\*][\w\s\*]+\s+\*?([A-Za-z_]\w*)\s*\([^;]*\)\s*\{"), "function"),
        ],
        imports=[_C(r'#include\s+[<"]([^>"]+)[>"]')],
    ),
    "C++": LangSpec(
        "C++",
        containers=[
            (_C(r"(?:template\s*<[^>]*>\s*)?class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"(?:typedef\s+)?struct\s+([A-Za-z_]\w*)"), "struct"),
            (_C(r"enum\s+(?:class\s+)?([A-Za-z_]\w*)"), "enum"),
            (_C(r"namespace\s+([A-Za-z_]\w*)"), "namespace"),
        ],
        defs=[
            (_C(r"^\s*#define\s+([A-Za-z_]\w*)"), "macro"),
            (_C(r"^[\w:<>\*&][\w\s:<>\*&,]+\s+\*?&?([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const)?\s*\{"), "function"),
        ],
        imports=[_C(r'#include\s+[<"]([^>"]+)[>"]')],
        base_split=":",
    ),
    "Go": LangSpec(
        "Go",
        containers=[
            (_C(r"type\s+([A-Za-z_]\w*)\s+struct"), "struct"),
            (_C(r"type\s+([A-Za-z_]\w*)\s+interface"), "interface"),
        ],
        defs=[
            (_C(r"func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("), "function"),
            (_C(r"type\s+([A-Za-z_]\w*)\s+(?!struct|interface)\w"), "type"),
            (_C(r"^\s*const\s+([A-Za-z_]\w*)\s*="), "constant"),
        ],
        imports=[_C(r'"([^"]+)"')],   # 仅 import 块内有效，下方按上下文过滤
    ),
    "Rust": LangSpec(
        "Rust",
        containers=[
            (_C(r"(?:pub\s+)?struct\s+([A-Za-z_]\w*)"), "struct"),
            (_C(r"(?:pub\s+)?enum\s+([A-Za-z_]\w*)"), "enum"),
            (_C(r"(?:pub\s+)?trait\s+([A-Za-z_]\w*)"), "trait"),
            (_C(r"impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?([A-Za-z_]\w*)"), "impl"),
        ],
        defs=[
            (_C(r"(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"), "function"),
            (_C(r"(?:pub\s+)?(?:const|static)\s+([A-Za-z_]\w*)\s*:"), "constant"),
            (_C(r"(?:pub\s+)?type\s+([A-Za-z_]\w*)\s*="), "type"),
        ],
        imports=[_C(r"use\s+([\w:]+)")],
    ),
    "Ruby": LangSpec(
        "Ruby", line_comments=("#",), block_comment=None,
        containers=[
            (_C(r"^\s*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"^\s*module\s+([A-Za-z_]\w*)"), "module"),
        ],
        defs=[(_C(r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[!?=]?)"), "method")],
        imports=[_C(r"require(?:_relative)?\s+[\'\"]([^\'\"]+)[\'\"]")],
        base_split="<",
    ),
    "PHP": LangSpec(
        "PHP",
        containers=[
            (_C(r"(?:abstract\s+|final\s+)*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"interface\s+([A-Za-z_]\w*)"), "interface"),
            (_C(r"trait\s+([A-Za-z_]\w*)"), "trait"),
        ],
        defs=[(_C(r"(?:public|private|protected|static|\s)*function\s+([A-Za-z_]\w*)\s*\("), "function")],
        imports=[_C(r"use\s+([\w\\]+)"), _C(r"(?:require|include)(?:_once)?\s+[\'\"]([^\'\"]+)[\'\"]")],
        base_split="extends",
    ),
    "Swift": LangSpec(
        "Swift",
        containers=[
            (_C(r"(?:public|private|internal|open|final|\s)*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"(?:public|private|internal|\s)*struct\s+([A-Za-z_]\w*)"), "struct"),
            (_C(r"(?:public|private|internal|\s)*protocol\s+([A-Za-z_]\w*)"), "interface"),
            (_C(r"(?:public|private|internal|\s)*enum\s+([A-Za-z_]\w*)"), "enum"),
        ],
        defs=[(_C(r"(?:public|private|internal|static|override|\s)*func\s+([A-Za-z_]\w*)"), "function")],
        imports=[_C(r"import\s+([A-Za-z_]\w*)")],
        base_split=":",
    ),
    "Kotlin": LangSpec(
        "Kotlin",
        containers=[
            (_C(r"(?:public|private|internal|abstract|open|sealed|data\s+|\s)*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"interface\s+([A-Za-z_]\w*)"), "interface"),
            (_C(r"object\s+([A-Za-z_]\w*)"), "object"),
        ],
        defs=[(_C(r"(?:public|private|internal|override|suspend|\s)*fun\s+(?:<[^>]+>\s+)?([A-Za-z_]\w*)"), "function")],
        imports=[_C(r"import\s+([\w.]+)")],
        base_split=":",
    ),
    "Scala": LangSpec(
        "Scala",
        containers=[
            (_C(r"(?:abstract\s+|final\s+|sealed\s+|case\s+)*class\s+([A-Za-z_]\w*)"), "class"),
            (_C(r"trait\s+([A-Za-z_]\w*)"), "trait"),
            (_C(r"object\s+([A-Za-z_]\w*)"), "object"),
        ],
        defs=[(_C(r"def\s+([A-Za-z_]\w*)"), "function")],
        imports=[_C(r"import\s+([\w.]+)")],
        base_split="extends",
    ),
    "Python-regex": LangSpec(
        "Python", line_comments=("#",), block_comment=None,
        containers=[(_C(r"^\s*class\s+([A-Za-z_]\w*)"), "class")],
        defs=[(_C(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"), "function")],
        imports=[_C(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")],
    ),
}

# 复用通用 C 家族 spec 给相近语言
_LANG_SPECS["Objective-C"] = _LANG_SPECS["C"]


def _extract_bases(line: str, keyword: Optional[str]) -> List[str]:
    """从声明行提取基类/接口名。"""
    if not keyword:
        return []
    if keyword == ":":
        m = re.search(r":\s*([^{<\n]+)", line)
    else:
        m = re.search(keyword + r"\s+([^{<\n]+?)(?:\{|implements|where|$)", line)
    if not m:
        return []
    chunk = re.sub(r"\b(public|private|protected|virtual|abstract)\b", "", m.group(1))
    names = re.findall(r"[A-Za-z_][\w.]*", chunk)
    return [n for n in names if n and n not in _KEYWORDS][:6]


def _scan_regex(content: str, rel: str, spec: LangSpec) -> List[Symbol]:
    """带花括号深度跟踪的通用扫描器。"""
    symbols: List[Symbol] = []
    stack: List[dict] = []      # 容器栈: {name, depth}
    depth = 0
    in_block = False
    block_close = spec.block_comment[1] if spec.block_comment else None
    lines = content.splitlines()

    for lineno, raw in enumerate(lines, 1):
        line = raw

        # 块注释跨行处理
        if in_block:
            if block_close and block_close in line:
                line = line.split(block_close, 1)[1]
                in_block = False
            else:
                continue
        if spec.block_comment:
            open_b, close_b = spec.block_comment
            while open_b in line:
                before, _, after = line.partition(open_b)
                if close_b in after:
                    line = before + " " + after.split(close_b, 1)[1]
                else:
                    line = before
                    in_block = True
                    break

        clean = _blank_strings(line)
        # 去行注释
        for c in spec.line_comments:
            idx = clean.find(c)
            if idx != -1:
                clean = clean[:idx]
        if not clean.strip():
            depth += clean.count("{") - clean.count("}")
            continue

        parent = stack[-1]["name"] if stack else ""
        matched = False

        # 容器声明优先
        for pat, kind in spec.containers:
            m = pat.search(clean)
            if m and m.group(1):
                name = m.group(1)
                bases = _extract_bases(clean, spec.base_split) if kind in ("class", "struct", "interface", "trait", "impl") else []
                sym = Symbol(name=name, kind=kind, file=rel, line=lineno,
                             signature=clean.strip()[:140], parent=parent,
                             bases=bases, language=spec.name)
                symbols.append(sym)
                # 仅在该行（或后续）真正开启代码块时入栈
                stack.append({"name": name, "depth": depth})
                matched = True
                break

        if not matched:
            for pat, kind in spec.defs:
                m = pat.search(clean)
                if m and m.group(1):
                    name = m.group(1)
                    if name in _KEYWORDS:
                        continue
                    k = kind
                    if kind == "function" and parent and parent != "":
                        k = "method"
                    sym = Symbol(name=name, kind=k, file=rel, line=lineno,
                                 signature=clean.strip()[:140], parent=parent,
                                 language=spec.name)
                    symbols.append(sym)
                    break

        # 更新深度并出栈
        depth += clean.count("{") - clean.count("}")
        while stack and depth <= stack[-1]["depth"]:
            stack.pop()

    return symbols


def _regex_imports(content: str, spec: LangSpec) -> List[str]:
    out: List[str] = []
    for pat in spec.imports:
        for m in pat.finditer(content):
            val = next((g for g in m.groups() if g), None)
            if val:
                out.append(val)
    # 去重保序
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 依赖解析（import → 内部文件）
# ──────────────────────────────────────────────────────────────────────────

def _resolve_python_dep(imp: str, file_rel: str, path_set, module_index) -> Optional[str]:
    imp = imp.strip()
    if not imp:
        return None
    if imp.startswith("."):
        # 相对导入
        base = Path(file_rel).parent
        ups = len(imp) - len(imp.lstrip("."))
        for _ in range(ups - 1):
            base = base.parent
        tail = imp.lstrip(".").replace(".", "/")
        cand = (base / tail).as_posix() if tail else base.as_posix()
        for suffix in (".py", "/__init__.py"):
            if (cand + suffix) in path_set:
                return cand + suffix
        return None
    # 优先匹配完整模块路径，再逐级回退到顶层包
    return (module_index.get(imp)
            or module_index.get(imp.replace(".", "/"))
            or module_index.get(imp.split(".")[0]))


def _resolve_relative_dep(imp: str, file_rel: str, path_set) -> Optional[str]:
    """JS/TS 等相对路径导入解析。"""
    if not (imp.startswith(".") or imp.startswith("/")):
        return None
    base = Path(file_rel).parent
    target = (base / imp).as_posix()
    target = os.path.normpath(target).replace(os.sep, "/")
    cands = [target] + [target + ext for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue")]
    cands += [target + "/index" + ext for ext in (".ts", ".js", ".tsx", ".jsx")]
    for c in cands:
        if c in path_set:
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────

def _iter_source_files(root: Path, max_files: int, max_size: int,
                       output_dirname: str, extra_skip: set, stats: dict):
    """遍历工作区，产出 (abs_path, rel_posix, language)。统计写入 ``stats``。"""
    skip = SKIP_DIRS | {output_dirname} | (extra_skip or set())
    count = 0
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地裁剪要跳过的目录（跳过依赖/构建/隐藏目录，但保留 .github）
        dirnames[:] = [d for d in dirnames
                       if d == ".github" or (d not in skip and not d.startswith("."))]
        for fn in sorted(filenames):
            ext = Path(fn).suffix.lower()
            if ext in BINARY_EXTS:
                continue
            lang = LANG_BY_EXT.get(ext)
            if not lang:
                continue
            scanned += 1
            ap = Path(dirpath) / fn
            try:
                if ap.stat().st_size > max_size:
                    continue
            except OSError:
                continue
            if count >= max_files:
                stats["truncated"] = True
                continue
            count += 1
            rel = ap.relative_to(root).as_posix()
            yield ap, rel, lang
    stats["scanned"] = scanned


def analyze_workspace(root, *, max_files: int = DEFAULT_MAX_FILES,
                      max_file_size: int = DEFAULT_MAX_FILE_SIZE,
                      output_dirname: str = "codemap",
                      extra_skip: Optional[set] = None) -> Analysis:
    """分析一个工作区文件夹，返回完整的 Analysis。"""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"不是文件夹: {root}")

    analysis = Analysis(
        root=str(root),
        name=root.name,
        generated=date.today().isoformat(),
    )

    # 1) 扫描 + 解析
    file_symbols: Dict[str, List[Symbol]] = {}
    scan_stats: dict = {"scanned": 0, "truncated": False}
    for ap, rel, lang in _iter_source_files(root, max_files, max_file_size,
                                            output_dirname, extra_skip or set(), scan_stats):
        try:
            content = ap.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        fi = FileInfo(path=rel, language=lang, loc=_count_loc(content), size=len(content))

        # `.m` 同时是 MATLAB 与 Objective-C 的扩展名 — 按内容判别
        if ap.suffix.lower() == ".m":
            lang = _classify_dot_m(content)
            fi.language = lang

        if lang == "Python":
            syms, imports, doc = parse_python(content, rel)
            fi.doc = doc
        elif lang == "MATLAB":
            syms, imports, doc = parse_matlab(content, rel)
            fi.doc = doc
        else:
            spec = _LANG_SPECS.get(lang)
            if spec is None:
                # 未知但已识别语言 → 用最朴素的通用 spec
                spec = LangSpec(lang, containers=[(_C(r"\bclass\s+([A-Za-z_]\w*)"), "class")],
                                defs=[(_C(r"\bfunction\s+([A-Za-z_]\w*)"), "function")], imports=[])
            syms = _scan_regex(content, rel, spec)
            imports = _regex_imports(content, spec)
            for s in syms:
                s.language = lang

        fi.imports = imports
        analysis.files.append(fi)
        analysis.symbols.extend(syms)
        file_symbols[rel] = syms

    analysis.scanned = scan_stats.get("scanned", len(analysis.files))
    analysis.truncated = scan_stats.get("truncated", False)
    analysis.skipped = max(0, analysis.scanned - len(analysis.files))

    # 2) 语言统计
    lang_stats: Dict[str, dict] = defaultdict(lambda: {"files": 0, "loc": 0, "symbols": 0})
    for fi in analysis.files:
        lang_stats[fi.language]["files"] += 1
        lang_stats[fi.language]["loc"] += fi.loc
    for s in analysis.symbols:
        if s.language in lang_stats:
            lang_stats[s.language]["symbols"] += 1
    analysis.lang_stats = dict(sorted(lang_stats.items(), key=lambda kv: -kv[1]["loc"]))

    # 3) 依赖解析（import → 内部文件）
    path_set = {fi.path for fi in analysis.files}
    module_index: Dict[str, str] = {}
    for fi in analysis.files:
        if fi.language == "Python":
            mod = fi.path[:-3].replace("/", ".") if fi.path.endswith(".py") else fi.path
            module_index[mod] = fi.path
            module_index[fi.path[:-3]] = fi.path
            if fi.path.endswith("/__init__.py"):
                module_index[fi.path[:-len("/__init__.py")].replace("/", ".")] = fi.path
    for fi in analysis.files:
        deps = []
        for imp in fi.imports:
            dep = None
            if fi.language == "Python":
                dep = _resolve_python_dep(imp, fi.path, path_set, module_index)
            else:
                dep = _resolve_relative_dep(imp, fi.path, path_set)
            if dep and dep != fi.path and dep not in deps:
                deps.append(dep)
        fi.deps = deps

    # 4) 引用计数（启发式：跨文件标识符出现次数）
    _compute_references(analysis)

    # 5) 构建图谱边
    analysis.edges = _build_edges(analysis, file_symbols)

    return analysis


def _compute_references(analysis: Analysis):
    """统计每个符号被跨文件引用的次数（Source Insight 的 'where used' 近似）。"""
    tracked_kinds = {"class", "function", "method", "interface", "struct",
                     "enum", "trait", "constant", "type", "macro", "object", "module"}
    name_to_syms: Dict[str, List[Symbol]] = defaultdict(list)
    for s in analysis.symbols:
        if s.kind in tracked_kinds and len(s.name) >= 3 and s.name not in _KEYWORDS:
            name_to_syms[s.name].append(s)

    if not name_to_syms:
        return

    name_totals: Counter = Counter()
    root = Path(analysis.root)
    for fi in analysis.files:
        try:
            content = (root / fi.path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        tokens = Counter(_IDENT_RE.findall(content))
        for name in name_to_syms:
            if name in tokens:
                name_totals[name] += tokens[name]

    for name, syms in name_to_syms.items():
        # 总出现次数减去定义自身出现的一次
        refs = max(0, name_totals[name] - len(syms))
        for s in syms:
            s.ref_count = refs


def _build_edges(analysis: Analysis, file_symbols: Dict[str, List[Symbol]]) -> List[dict]:
    """构建知识图谱的边：包含 / 导入 / 继承 / 调用。"""
    edges: List[dict] = []
    seen = set()

    def add(frm, to, etype):
        key = (frm, to, etype)
        if key not in seen and frm != to:
            seen.add(key)
            edges.append({"from": frm, "to": to, "type": etype})

    # 名称索引（用于继承/调用解析）
    by_name: Dict[str, List[Symbol]] = defaultdict(list)
    for s in analysis.symbols:
        by_name[s.name].append(s)

    # 4a) 包含：文件 → 顶层符号；类 → 方法/字段
    for rel, syms in file_symbols.items():
        file_node = f"file:{rel}"
        class_node = {}  # qualname → node id（同文件内类）
        for s in syms:
            if s.kind in ("class", "struct", "interface", "trait", "enum", "namespace", "object", "module", "impl"):
                class_node[s.name] = s.node_id
        for s in syms:
            if s.parent and s.parent in class_node:
                add(class_node[s.parent], s.node_id, "contains")
            else:
                add(file_node, s.node_id, "contains")

    # 4b) 导入：文件 → 文件
    for fi in analysis.files:
        for dep in fi.deps:
            add(f"file:{fi.path}", f"file:{dep}", "imports")

    # 4c) 继承：类 → 基类（解析到内部符号）
    for s in analysis.symbols:
        for base in s.bases:
            targets = by_name.get(base, [])
            if targets:
                add(s.node_id, targets[0].node_id, "inherits")

    # 4d) 调用：函数 → 函数（仅 Python 精确捕获）
    for s in analysis.symbols:
        for callee in s.calls:
            targets = [t for t in by_name.get(callee, []) if t.kind in ("function", "method")]
            if targets:
                add(s.node_id, targets[0].node_id, "calls")

    return edges
