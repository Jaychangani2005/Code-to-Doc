"""
Code-to-Doc: Phase 4 – Function-Level Documentation Generation
================================================================
Uses **LangChain** + Groq (Llama-3.3-70B)
to generate function-level documentation for every parsed module.

Three specialised prompt agents:
    Prompt 4.1 – Python Docstring Generator   (Google-style)
    Prompt 4.2 – Java Doc Generator           (JavaDoc format)
    Prompt 4.3 – Complex Logic Explanation     (algorithm / edge-case analysis)

Workflow per module:
    1. Read source code & parsed AST data
    2. For each function/method:
       a. Generate Google-style docstring          (Prompt 4.1)
       b. Explain complex logic if CC ≥ threshold  (Prompt 4.3)
    3. Save per-module report  → output/phase4_reports/modules/<module>.md
    4. Build MASTER_REPORT.md  → output/phase4_reports/MASTER_REPORT.md

Technology: LangChain, Groq, AST, rich
Output    : output/phase4_reports/
"""

import os
import ast
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

# ── LLM imports ──────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────
load_dotenv()
_p2_env = Path(__file__).parent.parent / "project2" / ".env"
if _p2_env.exists():
    load_dotenv(_p2_env)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_ID       = "llama-3.3-70b-versatile"
MAX_CODE_CHARS = 10000

BASE_DIR       = Path(__file__).parent
ANALYSIS_FILE  = BASE_DIR / "output" / "analysis_results.json"
OUTPUT_DIR     = BASE_DIR / "output" / "phase4_reports"
MODULES_DIR    = OUTPUT_DIR / "modules"

# Complexity threshold — functions with CC ≥ this get a Prompt 4.3 explanation
COMPLEX_THRESHOLD = 5


# ============================================================================
# LANGCHAIN LLM CLIENT  (shared by all prompt agents)
# ============================================================================

class GroqLLMClient:
    """LangChain ChatGroq client for fast LLM inference."""

    def __init__(self, api_key: str, model_id: str = MODEL_ID):
        if not api_key:
            raise ValueError(
                "Groq API key is missing.\n"
                "Set GROQ_API_KEY in your .env or environment."
            )
        self.model_id = model_id
        self.client = ChatGroq(
            model=model_id,
            api_key=api_key,
            temperature=0.2,
            max_tokens=2048,
            model_kwargs={"top_p": 0.9},
        )
        logger.info("LangChain ChatGroq client ready → %s", model_id)

    @staticmethod
    def _coerce_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return self._coerce_text(response.content)


# ============================================================================
# HELPER – read source code
# ============================================================================

def _read_source(repo_path: Path, rel_path: str) -> str:
    abs_path = repo_path / rel_path
    try:
        raw = abs_path.read_bytes()
        try:
            import chardet
            enc = chardet.detect(raw[:8192]).get("encoding") or "utf-8"
        except ImportError:
            enc = "utf-8"
        return raw.decode(enc, errors="replace")
    except Exception:
        return ""


def _extract_function_source(full_source: str, lineno: int,
                              end_lineno: Optional[int]) -> str:
    """Extract function source code from full file by line numbers."""
    lines = full_source.splitlines()
    start = max(0, lineno - 1)
    end   = end_lineno if end_lineno else min(start + 50, len(lines))
    return "\n".join(lines[start:end])


# ============================================================================
# PROMPT 4.1 – PYTHON DOCSTRING GENERATOR
# ============================================================================

class PythonDocstringAgent:
    """
    Generates Google-style docstrings for Python functions/methods.

    System prompt enforces:
        - Short summary
        - Detailed explanation if logic is complex
        - Args with types and descriptions
        - Returns
        - Raises (if applicable)
        - Example usage (if meaningful)
    """

    SYSTEM_PROMPT = """\
You are a Senior Python Engineer specialising in documentation.

Analyze the given Python function and generate a Google-style docstring.

Include:
- Short summary (one line)
- Detailed explanation (if the logic is complex)
- Args: each parameter with type and description
- Returns: what the function returns, with type
- Raises: any exceptions the function may raise (if applicable)
- Example: a short usage example (if meaningful)

Return ONLY the properly formatted docstring wrapped in triple quotes.
Do NOT include the function signature — only the docstring body.\
"""

    def __init__(self, llm: GroqLLMClient):
        self.llm = llm

    def generate_docstring(self, func_name: str, func_code: str,
                           module_context: str = "") -> str:
        user_prompt = f"""Generate a Google-style docstring for this Python function.

Module context: {module_context[:200] if module_context else "N/A"}

Function:
```python
{func_code[:MAX_CODE_CHARS]}
```

Return ONLY the docstring (triple-quoted). No other text."""

        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error(f"Docstring generation failed for {func_name}: {e}")
            return f'"""TODO: Documentation generation failed: {e}"""'


# ============================================================================
# PROMPT 4.2 – JAVA DOC GENERATOR
# ============================================================================

class JavaDocAgent:
    """
    Generates JavaDoc comments for Java methods.
    Included for completeness per the spec — activated only for .java files.
    """

    SYSTEM_PROMPT = """\
You are a Senior Java Engineer specialising in documentation.

Analyze the following Java method and generate a JavaDoc comment.

Include:
- Brief description
- @param tags for every parameter
- @return tag describing the return value
- @throws tags for checked and unchecked exceptions (if applicable)

Return ONLY the JavaDoc block (/** ... */). No other text.\
"""

    def __init__(self, llm: GroqLLMClient):
        self.llm = llm

    def generate_javadoc(self, method_name: str, method_code: str) -> str:
        user_prompt = f"""Generate a JavaDoc comment for this Java method.

Method:
```java
{method_code[:MAX_CODE_CHARS]}
```

Return ONLY the JavaDoc block."""

        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error(f"JavaDoc generation failed for {method_name}: {e}")
            return f"/** TODO: Documentation generation failed: {e} */"


# ============================================================================
# PROMPT 4.3 – COMPLEX LOGIC EXPLANATION
# ============================================================================

class ComplexLogicAgent:
    """
    Explains the core logic of complex functions in simple terms.

    Activated when cyclomatic complexity ≥ COMPLEX_THRESHOLD.
    Focuses on: algorithm, edge cases, conditional branches, time complexity.
    """

    SYSTEM_PROMPT = """\
You are a Senior Software Engineer explaining code to junior developers.

Explain the core logic of the given function in simple, clear terms.

Focus on:
- The algorithm or approach used
- Edge cases that are handled
- Important conditional branches and what they control
- Time complexity (if identifiable)
- Why the function is written this way

Keep the explanation concise but thorough. Use bullet points.
Do NOT reproduce the full code — explain it.\
"""

    def __init__(self, llm: GroqLLMClient):
        self.llm = llm

    def explain(self, func_name: str, func_code: str,
                complexity: int) -> str:
        user_prompt = f"""Explain the core logic of this function in simple terms.

Function name: `{func_name}`
Cyclomatic complexity: {complexity}

```python
{func_code[:MAX_CODE_CHARS]}
```

Provide a clear explanation a junior developer can understand."""

        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error(f"Logic explanation failed for {func_name}: {e}")
            return f"> ⚠️ Explanation generation failed: {e}"


# ============================================================================
# PHASE 4 ORCHESTRATOR
# ============================================================================

class Phase4:
    """
    Orchestrates Phase 4: Function-Level Documentation Generation.

    For each parsed module:
        1. Read source code
        2. For every function/method → Prompt 4.1 (docstring)
        3. For complex functions (CC ≥ threshold) → Prompt 4.3 (logic explanation)
        4. Save per-module report
    Then build MASTER_REPORT.md
    """

    def __init__(self, api_key: str):
        llm = GroqLLMClient(api_key)

        self.docstring_agent = PythonDocstringAgent(llm)
        self.javadoc_agent   = JavaDocAgent(llm)
        self.logic_agent     = ComplexLogicAgent(llm)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODULES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    def _load_analysis(self) -> Dict:
        if not ANALYSIS_FILE.exists():
            raise FileNotFoundError(
                f"analysis_results.json not found at {ANALYSIS_FILE}\n"
                "Run phase1.py, phase2.py, phase3.py first."
            )
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Process one function ─────────────────────────────────────────────
    def _process_function(
        self, func: Dict, full_source: str, module_docstring: str
    ) -> Dict:
        """
        Run Prompt 4.1 (and optionally 4.3) on a single function.

        Returns dict with: name, signature, docstring, explanation (if complex).
        """
        name       = func.get("name", "?")
        lineno     = func.get("lineno")
        end_lineno = func.get("end_lineno")
        complexity = func.get("complexity", 1)
        signature  = func.get("signature", name)

        func_code = _extract_function_source(full_source, lineno, end_lineno)

        # Prompt 4.1 — Google-style docstring
        docstring = self.docstring_agent.generate_docstring(
            name, func_code, module_docstring
        )
        time.sleep(0.8)  # rate-limit

        # Prompt 4.3 — Complex logic explanation (if CC ≥ threshold)
        explanation = ""
        if complexity >= COMPLEX_THRESHOLD:
            explanation = self.logic_agent.explain(name, func_code, complexity)
            time.sleep(0.8)

        return {
            "name":        name,
            "signature":   signature,
            "lineno":      lineno,
            "complexity":  complexity,
            "docstring":   docstring,
            "explanation": explanation,
        }

    # ── Process one module ───────────────────────────────────────────────
    def _process_module(
        self, file_path: str, module_data: Dict, repo_path: Path
    ) -> Dict:
        """
        Process all functions & class methods in one module.

        Returns dict with: file, functions[], classes[].
        """
        full_source     = _read_source(repo_path, file_path)
        module_docstring = module_data.get("docstring", "")

        # Standalone functions
        func_results = []
        for func in module_data.get("functions", []):
            result = self._process_function(func, full_source, module_docstring)
            func_results.append(result)

        # Class methods
        class_results = []
        for cls in module_data.get("classes", []):
            method_results = []
            for method in cls.get("methods", []):
                result = self._process_function(method, full_source, module_docstring)
                method_results.append(result)
            class_results.append({
                "name":    cls["name"],
                "bases":   cls.get("bases", []),
                "methods": method_results,
            })

        return {
            "file":      file_path,
            "functions": func_results,
            "classes":   class_results,
        }

    # ── Save module report ───────────────────────────────────────────────
    def _save_module_report(self, module_result: Dict) -> Path:
        file_path = module_result["file"]
        safe_name = file_path.replace("/", "_").replace("\\", "_") + ".md"
        out_path  = MODULES_DIR / safe_name

        lines = [
            f"# Phase 4: Function-Level Documentation",
            f"## Module: `{file_path}`",
            "",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            "---",
            "",
        ]

        # ── Standalone functions ─────────────────────────────────────────
        funcs = module_result.get("functions", [])
        if funcs:
            lines.append("## Functions\n")
            for f in funcs:
                lines.append(f"### `{f['signature']}`")
                lines.append(f"- **Line:** {f['lineno']}  |  **Complexity (CC):** {f['complexity']}")
                lines.append("")
                lines.append("**Generated Docstring:**")
                lines.append("```python")
                lines.append(f['docstring'])
                lines.append("```")
                lines.append("")
                if f.get("explanation"):
                    lines.append("**Complex Logic Explanation:**")
                    lines.append("")
                    lines.append(f["explanation"])
                    lines.append("")
                lines.append("---\n")

        # ── Classes ──────────────────────────────────────────────────────
        classes = module_result.get("classes", [])
        if classes:
            lines.append("## Classes\n")
            for cls in classes:
                bases = ", ".join(cls["bases"]) if cls["bases"] else "object"
                lines.append(f"### Class `{cls['name']}` (bases: {bases})\n")
                for m in cls.get("methods", []):
                    lines.append(f"#### `{m['signature']}`")
                    lines.append(f"- **Line:** {m['lineno']}  |  **Complexity (CC):** {m['complexity']}")
                    lines.append("")
                    lines.append("**Generated Docstring:**")
                    lines.append("```python")
                    lines.append(m["docstring"])
                    lines.append("```")
                    lines.append("")
                    if m.get("explanation"):
                        lines.append("**Complex Logic Explanation:**")
                        lines.append("")
                        lines.append(m["explanation"])
                        lines.append("")
                    lines.append("---\n")

        if not funcs and not classes:
            lines.append("_No functions or classes found in this module._\n")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    # ── Master Report ────────────────────────────────────────────────────
    def _build_master_report(
        self, analysis: Dict, module_summaries: List[Dict]
    ) -> str:
        metadata = analysis.get("metadata", {})
        cmplx    = analysis.get("phase2", {}).get("complexity_metrics", {}).get("project_level", {})
        graph_a  = analysis.get("phase2", {}).get("graph_analysis", {})
        arch     = analysis.get("phase2", {}).get("architecture", {})

        total      = len(module_summaries)
        processed  = sum(1 for m in module_summaries if m.get("processed"))
        total_fns  = sum(m.get("functions_documented", 0) for m in module_summaries)
        total_expl = sum(m.get("complex_explained", 0) for m in module_summaries)

        lines = [
            "# Code-to-Doc: Phase 4 — Master Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Model:** `{MODEL_ID}`  ",
            f"**Engine:** LangChain + Groq  ",
            "",
            "---",
            "",
            "## Repository Overview",
            "",
            "| Key | Value |",
            "|-----|-------|",
            f"| Project | {metadata.get('project_name', 'N/A')} |",
            f"| URL | {analysis.get('github_url', 'N/A')} |",
            f"| Language | {metadata.get('primary_language', 'N/A')} |",
            f"| Source Files | {metadata.get('total_source_files', 0)} |",
            f"| Lines of Code | {metadata.get('total_lines_of_code', 0):,} |",
            "",
            "## Phase 4 Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Modules Processed | {processed} / {total} |",
            f"| Functions Documented (Prompt 4.1) | {total_fns} |",
            f"| Complex Logic Explained (Prompt 4.3) | {total_expl} |",
            f"| Complexity Threshold for Explanation | CC ≥ {COMPLEX_THRESHOLD} |",
            "",
            "## High-Complexity Functions",
            "",
        ]

        hc = cmplx.get("high_complexity_functions", [])
        if hc:
            lines.append("| File | Function | CC |")
            lines.append("|------|----------|----|")
            for h in hc[:15]:
                fn = h.get("method", h.get("function", "?"))
                cls = h.get("class", "")
                name = f"{cls}.{fn}" if cls else fn
                lines.append(f"| `{h.get('file', '?')}` | `{name}` | {h.get('complexity', '?')} |")
        else:
            lines.append("_No high-complexity functions detected._")

        lines += [
            "",
            "## Circular Dependencies",
            "",
        ]
        circulars = graph_a.get("circular_dependencies", [])
        if circulars:
            for cyc in circulars[:10]:
                lines.append(f"- {' → '.join(cyc)}")
        else:
            lines.append("✅ No circular dependencies detected.")

        lines += [
            "",
            "## Core Modules (most imported)",
            "",
            "| Module | Imported By # |",
            "|--------|---------------|",
        ]
        for cm in graph_a.get("core_modules", [])[:8]:
            lines.append(f"| `{cm['module']}` | {cm['imported_by_count']} |")

        lines += [
            "",
            "## Module Reports",
            "",
            "| # | Module | Functions | Complex Explained | Report |",
            "|---|--------|-----------|-------------------|--------|",
        ]
        for i, m in enumerate(module_summaries, 1):
            status = "✅" if m.get("processed") else "⏭"
            doc_name = m.get("report_name", "—")
            link = f"[{doc_name}](modules/{doc_name})" if m.get("processed") else "—"
            lines.append(
                f"| {i} | `{m['file']}` | {m.get('functions_documented', 0)} "
                f"| {m.get('complex_explained', 0)} | {status} {link} |"
            )

        lines += [
            "",
            "---",
            f"*Generated by Code-to-Doc Phase 4 on {datetime.now().strftime('%Y-%m-%d')}*",
        ]
        return "\n".join(lines)

    # ── Full run ─────────────────────────────────────────────────────────
    def run(self) -> Dict:
        console.print(Panel.fit(
            "[bold green]Code-to-Doc  ·  Phase 4[/]\n"
            "Function-Level Documentation Generation\n"
            f"Model: [cyan]{MODEL_ID}[/]  ·  Engine: [cyan]LangChain[/]\n"
            f"Prompts: 4.1 (Docstring) + 4.2 (JavaDoc) + 4.3 (Logic Explanation)",
            border_style="green",
        ))

        analysis       = self._load_analysis()
        repo_path      = Path(analysis["repo_path"])
        phase2         = analysis.get("phase2", {})
        parsed_modules = phase2.get("parsed_modules", {})

        if not parsed_modules:
            console.print("[red]No parsed_modules found. Run phase2.py first.[/]")
            return {}

        modules_to_process = list(parsed_modules.items())
        total = len(modules_to_process)

        console.print(f"\n[cyan]Processing {total} modules…[/]\n")

        module_summaries: List[Dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("Phase 4", total=total)

            for idx, (file_path, module_data) in enumerate(modules_to_process, 1):
                progress.update(task, description=f"[{idx}/{total}] {file_path}")

                module_result = self._process_module(
                    file_path, module_data, repo_path
                )

                report_path = self._save_module_report(module_result)
                safe_name = report_path.name

                fns_documented = len(module_result["functions"])
                for cls in module_result["classes"]:
                    fns_documented += len(cls.get("methods", []))

                complex_explained = sum(
                    1 for f in module_result["functions"] if f.get("explanation")
                )
                for cls in module_result["classes"]:
                    complex_explained += sum(
                        1 for m in cls.get("methods", []) if m.get("explanation")
                    )

                module_summaries.append({
                    "file":                file_path,
                    "processed":           True,
                    "report_name":         safe_name,
                    "functions_documented": fns_documented,
                    "complex_explained":   complex_explained,
                })

                progress.advance(task)

        # ── Master Report ────────────────────────────────────────────────
        master_md = self._build_master_report(analysis, module_summaries)
        master_path = OUTPUT_DIR / "MASTER_REPORT.md"
        master_path.write_text(master_md, encoding="utf-8")
        logger.info(f"Master report → {master_path}")

        # ── Update analysis_results.json ─────────────────────────────────
        analysis["phase"] = 4
        analysis["phase4"] = {
            "model":              MODEL_ID,
            "engine":             "LangChain + Groq",
            "prompts_used":       ["4.1_python_docstring", "4.2_javadoc", "4.3_complex_logic"],
            "complexity_threshold": COMPLEX_THRESHOLD,
            "modules_processed":  len(module_summaries),
            "total_functions_documented": sum(m["functions_documented"] for m in module_summaries),
            "total_complex_explained":    sum(m["complex_explained"] for m in module_summaries),
            "output_dir":         str(OUTPUT_DIR),
            "module_reports":     module_summaries,
            "completed_at":       datetime.now().isoformat(),
        }
        ANALYSIS_FILE.write_text(
            json.dumps(analysis, indent=2, default=str),
            encoding="utf-8",
        )

        # ── Summary table ────────────────────────────────────────────────
        total_fns  = sum(m["functions_documented"] for m in module_summaries)
        total_expl = sum(m["complex_explained"] for m in module_summaries)

        t = Table(title="Phase 4 — Function-Level Documentation", show_lines=True)
        t.add_column("Metric", style="cyan", min_width=30)
        t.add_column("Value",  style="green", justify="right")
        t.add_row("Model",                         MODEL_ID)
        t.add_row("Engine",                        "LangChain")
        t.add_row("Modules processed",             str(len(module_summaries)))
        t.add_row("Functions documented (4.1)",     str(total_fns))
        t.add_row("Complex logic explained (4.3)",  str(total_expl))
        t.add_row("Output directory",               str(OUTPUT_DIR))
        console.print(t)

        console.print(Panel.fit(
            f"[bold green]✓ Phase 4 complete[/]\n"
            f"Reports → [cyan]{OUTPUT_DIR}[/]\n"
            f"Master  → [cyan]{master_path}[/]",
            border_style="green",
        ))
        return analysis


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

    if not GROQ_API_KEY:
        console.print(Panel(
            "[red bold]Groq API key not found![/]\n\n"
            "Set [cyan]GROQ_API_KEY[/] in your [yellow].env[/] file or environment.\n"
            "Get a key at: https://console.groq.com/keys",
            title="❌ Missing API Key",
            border_style="red",
        ))
        raise SystemExit(1)

    phase4 = Phase4(api_key=GROQ_API_KEY)
    phase4.run()
