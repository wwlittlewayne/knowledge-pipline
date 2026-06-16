把一个文件夹当作项目工作区，分析其完整的代码符号知识图谱（Source Insight 风格），并输出结构化文本供 LLM 使用。

使用方法：
- `/pipeline-code` — 分析当前工作目录
- `/pipeline-code /path/to/project` — 分析指定文件夹
- `/pipeline-code . --open` — 分析并在浏览器打开交互式图谱

---

## 这个命令做什么

像 Source Insight 一样，递归扫描一个代码库，提取**所有符号**
（类 / 函数 / 方法 / 接口 / 结构体 / 枚举 / 常量 / 宏 / 类型 / 字段 …）
以及它们之间的**关系**（包含 / 导入依赖 / 继承 / 调用 / 引用），
构建一张完整的代码知识图谱，并输出：

| 产物 | 说明 |
|------|------|
| `codemap/codemap.md` | **结构化文本地图** — 为 LLM 优化，直接喂给模型即可"读懂"整个项目 |
| `codemap/symbols.json` | 完整机器可读符号数据库（文件 / 符号 / 边 / 统计） |
| `codemap/codemap.html` | 自包含 vis.js 交互式符号图谱 |

**重要：本命令是确定性的纯静态分析，不需要 LLM API，也不需要 `/pipeline-config`。**
离线即可运行，秒级完成。支持 Python（精确 `ast` 解析）、
JavaScript / TypeScript / Java / C / C++ / C# / Go / Rust / Ruby / PHP / Swift / Kotlin / Scala 等。

---

## 执行步骤

### 第一步：确定要分析的工作区 FOLDER

- 如果用户在命令后给了路径 → 用该路径作为 FOLDER。
- 否则 → 使用**当前工作目录**（用户打开的项目）作为 FOLDER。

### 第二步：定位分析工具 pipeline_code.py

按顺序检查，取第一个存在的：
1. 当前仓库的 `tools/pipeline_code.py`（在本项目内运行时）
2. `~/.agents/skills/knowledge-pipline/tools/pipeline_code.py`
3. `~/.claude/skills/knowledge-pipline/tools/pipeline_code.py`

在 Windows 上 `~` 展开为 `C:\Users\{用户名}`。

### 第三步：运行分析

在终端运行（把 TOOL 和 FOLDER 替换为实际路径）：
```
python "TOOL" "FOLDER" --open
```

常用选项：
- `--out DIR`：自定义输出目录（默认 `FOLDER/codemap`）
- `--max-files N`：调整文件上限（默认 5000）
- `--no-html`：跳过 HTML 图谱
- `--stdout`：把文本地图直接打到标准输出（便于直接读取喂给模型）

分析结果默认写入 `FOLDER/codemap/`。提示用户：该目录可加入 `.gitignore`。

### 第四步：Python 不可用时的回退

如果 `python` 不可用，用 Claude Code 内置能力做轻量分析：
1. 用 Glob 列出源码文件（如 `**/*.py`、`**/*.{ts,tsx,js}` 等），跳过
   `node_modules`、`.git`、`dist`、`build`、`venv`、`__pycache__` 等目录。
2. 对关键文件用 Read 读取，用 Grep 抽取定义
   （`class `、`def `、`function `、`func `、`struct `、`interface ` 等）。
3. 汇总成与 `codemap.md` 相同的结构：概览 → 目录树 → 文件符号大纲 →
   符号索引 → 模块依赖 → 类继承 → 调用关系。
4. 写入 `FOLDER/codemap/codemap.md`。

### 第五步：输出摘要

向用户报告：
- 文件数、代码行数、符号总数、关系边数
- 语言分布、符号构成（类 / 函数 / 方法 …）
- 三个产物的路径，并点明 `codemap.md` 就是"喂给 LLM 的结构化文本"
- 可提示："把 `codemap/codemap.md` 贴给任意 AI 模型，它就能快速理解你的项目结构。"
