"""
Code-to-Doc: Phase 3 – Agent Configuration & Documentation Generation
======================================================================
Uses **LangChain** + Groq (Llama-3.3-70B)
to generate per-module Markdown documentation from analysis_results.json.

Master Agent Personality:
    You are a Senior Software Engineer reviewing legacy production code.
    - Explain purpose clearly.
    - Describe parameters precisely.
    - Mention return types.
    - Document raised exceptions.
    - Include short usage examples when helpful.
    - Do NOT hallucinate missing functionality.
    - Python → Google-style docstrings format.

Steps:
    1. Load enriched analysis_results.json  (Phase 1 + 2)
    2. For every parsed Python module → build a structured prompt
    3. Call Llama-3-8B via LangChain → receive Markdown documentation
    4. Save each module doc as  output/generated_docs/<module>.md
    5. Generate INDEX.md summary

Technology: LangChain, Groq, rich
Output    : output/generated_docs/
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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
load_dotenv()                          # loads .env from cwd or parent
# Also try project2's .env as a fallback (shared token)
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
MAX_CODE_CHARS = 12000                # Groq supports 128K context

BASE_DIR       = Path(__file__).parent
ANALYSIS_FILE  = BASE_DIR / "output" / "analysis_results.json"
OUTPUT_DIR     = BASE_DIR / "output" / "generated_docs"


# ============================================================================
# MASTER AGENT PERSONALITY  (System Prompt)
# ============================================================================

SYSTEM_PROMPT = """\
You are a Senior Software Engineer reviewing legacy production code.

Your task is to generate clear, professional, and accurate documentation \
in Markdown format.

Follow these rules strictly:
- Explain the purpose of the module clearly.
- Describe every parameter precisely (name, type, default, meaning).
- Mention return types and what the return value represents.
- Document raised exceptions (type + when raised).
- Include a short usage example when it helps understanding.
- Do NOT hallucinate missing functionality.
- If you are unsure about something, state your assumption clearly.
- Keep explanations concise but informative.
- Use Google-style docstring conventions for Python code.

Output ONLY valid Markdown. No surrounding commentary.\
"""


# ============================================================================
# LANGCHAIN LLM WRAPPER
# ============================================================================

class GroqLLMClient:
    """
    Wraps LangChain ChatGroq for fast LLM inference.
    """

    def __init__(self, api_key: str, model_id: str = MODEL_ID):
        if not api_key:
            raise ValueError(
                "Groq API key is missing.\n"
                "Set GROQ_API_KEY in your .env file or environment.\n"
                "Get a key at: https://console.groq.com/keys"
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
        """
        Call Groq chat completion with system and user messages.

        Args:
            system_prompt: The agent personality / instructions.
            user_prompt:   The documentation task content.

        Returns:
            Generated Markdown string.
        """
        response = self.client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return self._coerce_text(response.content)


# ============================================================================
# PROMPT BUILDER
# ============================================================================

class PromptBuilder:
    """Builds structured documentation prompts from parsed module data."""

    @staticmethod
    def build_module_prompt(
        file_path: str,
        module_data: Dict,
        dependencies: Dict,
        source_code: str,
    ) -> str:
        """
        Build a detailed documentation prompt for a single module.

        Args:
            file_path:    Relative path of the file.
            module_data:  Parsed module info (functions, classes, imports, etc.).
            dependencies: Dependency graph entry for this file.
            source_code:  Raw source code (will be truncated).

        Returns:
            Formatted prompt string.
        """
        # ── Functions summary ────────────────────────────────────────────
        func_lines = []
        for fn in module_data.get("functions", []):
            sig  = fn.get("signature", fn["name"])
            doc  = fn.get("docstring") or "No docstring."
            cplx = fn.get("complexity", "?")
            priv = "private" if fn.get("is_private") else "public"
            ret  = fn.get("return_type", "Any")
            func_lines.append(
                f"  - `{sig}` → {ret}  [{priv}, complexity={cplx}]\n"
                f"    {doc[:150]}"
            )

        # ── Classes summary ──────────────────────────────────────────────
        class_lines = []
        for cls in module_data.get("classes", []):
            bases = ", ".join(cls.get("bases", [])) or "object"
            class_lines.append(f"  - **{cls['name']}** (bases: {bases})")
            if cls.get("docstring"):
                class_lines.append(f"    {cls['docstring'][:120]}")
            for m in cls.get("methods", []):
                sig  = m.get("signature", m["name"])
                doc  = m.get("docstring") or ""
                cplx = m.get("complexity", "?")
                class_lines.append(
                    f"      · `{sig}` [CC={cplx}] {doc[:80]}"
                )

        # ── Constants ────────────────────────────────────────────────────
        const_lines = []
        for c in module_data.get("constants", []):
            const_lines.append(f"  - `{c['name']}` = {c.get('value', '?')}")

        # ── Dependency info ──────────────────────────────────────────────
        internal_deps = dependencies.get("imports", [])
        external_deps = dependencies.get("external_deps", [])

        # ── Build import list as strings ─────────────────────────────────
        import_strs = []
        for imp in module_data.get("imports", []):
            if imp["type"] == "from":
                import_strs.append(f"from {imp['module']} import {imp['name']}")
            else:
                import_strs.append(f"import {imp['module']}")

        # ── Assemble ─────────────────────────────────────────────────────
        prompt = f"""
## Task
Write complete, professional technical documentation for the Python module: `{file_path}`

## Module Docstring
{module_data.get("docstring") or "None provided."}

## Imports
{chr(10).join(import_strs[:30]) if import_strs else "None"}

## Internal Dependencies (project modules this file imports)
{", ".join(internal_deps) if internal_deps else "None"}

## External Libraries Used
{", ".join(external_deps) if external_deps else "None"}

## Constants
{chr(10).join(const_lines) if const_lines else "None"}

## Functions
{chr(10).join(func_lines) if func_lines else "None"}

## Classes
{chr(10).join(class_lines) if class_lines else "None"}

## Source Code (truncated to {MAX_CODE_CHARS} chars)
```python
{source_code[:MAX_CODE_CHARS]}
```

## Required Documentation Structure
1. `# <Module Name>` — Title derived from the file path.
2. `## Overview` — 3–4 sentences explaining what this module does and why it exists.
3. `## Dependencies` — Internal and external dependencies explained briefly.
4. `## Constants` — Module-level constants with descriptions.
5. `## Classes` — Each class with: purpose, constructor parameters, key methods
   (purpose, parameters with types, return value, exceptions raised).
6. `## Functions` — Each standalone function documented the same way.
7. `## Usage Example` — A short, realistic code snippet showing typical usage.
8. `## Notes` — Any assumptions, caveats, or edge-case behaviour.

Output ONLY valid Markdown.
""".strip()
        return prompt


# ============================================================================
# PHASE 3 DOCUMENTATION GENERATOR
# ============================================================================

class Phase3:
    """
    Orchestrates Phase 3: Documentation Generation.

    1. Load analysis_results.json (Phase 1 + 2)
    2. For each parsed module → build prompt → call LLM → save .md
    3. Generate INDEX.md
    """

    def __init__(self, api_key: str):
        self.llm         = GroqLLMClient(api_key)
        self.prompt_bld  = PromptBuilder()
        self.output_dir  = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    def _load_analysis(self) -> Dict:
        if not ANALYSIS_FILE.exists():
            raise FileNotFoundError(
                f"analysis_results.json not found at {ANALYSIS_FILE}\n"
                "Run phase1.py and phase2.py first."
            )
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded analysis → {ANALYSIS_FILE}")
        return data

    # ── Read source code ─────────────────────────────────────────────────
    @staticmethod
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
        except Exception as e:
            logger.warning(f"Could not read {rel_path}: {e}")
            return ""

    # ── Document one file ────────────────────────────────────────────────
    def _document_file(
        self,
        file_path: str,
        module_data: Dict,
        dep_graph: Dict,
        repo_path: Path,
    ) -> str:
        source      = self._read_source(repo_path, file_path)
        deps        = dep_graph.get(file_path, {})
        user_prompt = self.prompt_bld.build_module_prompt(
            file_path, module_data, deps, source
        )

        try:
            doc = self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            return doc
        except Exception as e:
            logger.error(f"LLM call failed for {file_path}: {e}")
            return (
                f"# {file_path}\n\n"
                f"> ⚠️ Documentation generation failed: {e}\n"
            )

    # ── Save one doc ─────────────────────────────────────────────────────
    def _save_doc(self, file_path: str, content: str) -> Path:
        safe_name = file_path.replace("/", "_").replace("\\", "_") + ".md"
        out_path  = self.output_dir / safe_name
        out_path.write_text(content, encoding="utf-8")
        return out_path

    # ── Full run ─────────────────────────────────────────────────────────
    def run(self) -> Dict:
        console.print(Panel.fit(
            "[bold green]Code-to-Doc  ·  Phase 3[/]\n"
            "Agent Configuration & Documentation Generation\n"
            f"Model: [cyan]{MODEL_ID}[/]  ·  Engine: [cyan]Groq[/]",
            border_style="green",
        ))

        analysis       = self._load_analysis()
        repo_path      = Path(analysis["repo_path"])
        phase2         = analysis.get("phase2", {})
        parsed_modules = phase2.get("parsed_modules", {})
        dep_graph      = phase2.get("dependency_graph", {})

        if not parsed_modules:
            console.print("[red]No parsed_modules found. Run phase2.py first.[/]")
            return {}

        # ── Filter: only document source files (skip test files) ─────────
        # We document both source & test — user can later filter.
        modules_to_doc = list(parsed_modules.items())
        total = len(modules_to_doc)

        console.print(f"\n[cyan]Documenting {total} modules…[/]\n")

        generated_index: List[Dict] = []
        success = 0
        failed  = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("Generating docs", total=total)

            for idx, (file_path, module_data) in enumerate(modules_to_doc, 1):
                progress.update(task, description=f"[{idx}/{total}] {file_path}")

                doc_content = self._document_file(
                    file_path, module_data, dep_graph, repo_path,
                )

                saved_path = self._save_doc(file_path, doc_content)
                generated_index.append({
                    "file":     file_path,
                    "doc_path": str(saved_path.name),
                    "defs":     module_data.get("total_defs", 0),
                })
                success += 1

                progress.advance(task)

                # Gentle rate-limit to avoid hammering the free API
                time.sleep(1.0)

        # ── Generate INDEX.md ────────────────────────────────────────────
        self._save_index(generated_index, analysis)

        # ── Update analysis_results.json ─────────────────────────────────
        analysis["phase"] = 3
        analysis["phase3"] = {
            "model":         MODEL_ID,
            "engine":        "Groq API",
            "documented":    success,
            "failed":        failed,
            "output_dir":    str(self.output_dir),
            "generated_docs": generated_index,
            "completed_at":  datetime.now().isoformat(),
        }
        ANALYSIS_FILE.write_text(
            json.dumps(analysis, indent=2, default=str),
            encoding="utf-8",
        )

        # ── Summary ──────────────────────────────────────────────────────
        t = Table(title="Phase 3 — Documentation Generation", show_lines=True)
        t.add_column("Metric", style="cyan", min_width=25)
        t.add_column("Value",  style="green", justify="right")
        t.add_row("Model",           MODEL_ID)
        t.add_row("Engine",          "Groq")
        t.add_row("Modules documented", str(success))
        t.add_row("Failed",          str(failed))
        t.add_row("Output",          str(self.output_dir))
        console.print(t)

        console.print(Panel.fit(
            f"[bold green]✓ Phase 3 complete[/]\n"
            f"Docs saved → [cyan]{self.output_dir}[/]",
            border_style="green",
        ))
        return analysis

    # ── INDEX.md ─────────────────────────────────────────────────────────
    def _save_index(self, index: List[Dict], analysis: Dict):
        metadata = analysis.get("metadata", {})
        arch     = analysis.get("phase2", {}).get("architecture", {})
        cmplx    = analysis.get("phase2", {}).get("complexity_metrics", {}).get("project_level", {})

        lines = [
            "# Code-to-Doc: Documentation Index",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Model:** `{MODEL_ID}`  ",
            f"**Engine:** Groq API",
            "",
            "---",
            "",
            "## Repository Info",
            "",
            "| Key | Value |",
            "|-----|-------|",
            f"| Project | {metadata.get('project_name', 'N/A')} |",
            f"| URL | {analysis.get('github_url', 'N/A')} |",
            f"| Language | {metadata.get('primary_language', 'N/A')} |",
            f"| Source Files | {metadata.get('total_source_files', 0)} |",
            f"| Lines of Code | {metadata.get('total_lines_of_code', 0):,} |",
            "",
            "## Codebase Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Parsed Modules | {cmplx.get('total_source_files', 0)} |",
            f"| Total LOC | {cmplx.get('total_loc', 0):,} |",
            f"| Avg Complexity (CC) | {cmplx.get('average_complexity', 0)} |",
            f"| Max Complexity (CC) | {cmplx.get('max_complexity', 0)} |",
            "",
            "## Documented Modules",
            "",
            "| # | File | Definitions | Documentation |",
            "|---|------|-------------|---------------|",
        ]

        for i, entry in enumerate(index, 1):
            doc_name = entry["doc_path"]
            lines.append(
                f"| {i} | `{entry['file']}` | {entry['defs']} "
                f"| [{doc_name}]({doc_name}) |"
            )

        lines.append("")
        lines.append("---")
        lines.append(f"*Generated by Code-to-Doc Phase 3 on {datetime.now().strftime('%Y-%m-%d')}*")

        index_path = self.output_dir / "INDEX.md"
        index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Index saved → {index_path}")


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

    phase3 = Phase3(api_key=GROQ_API_KEY)
    phase3.run()
