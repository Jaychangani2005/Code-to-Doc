"""
Code-to-Doc: Phase 5 – Professional README Generation
=======================================================
Uses **LangChain** + Groq (Llama-3.3-70B)
to generate a production-ready README.md for the analysed codebase.

README Sections (Prompt 5.1):
    1. Project Overview
    2. Architecture Overview (with ASCII diagram)
    3. Core Components
    4. Dependency Summary
    5. Installation Guide
    6. Usage Example
    7. Module Guide
    8. Known Limitations
    9. Development Notes

Inputs  : analysis_results.json (Phase 1-4 data)
Output  : output/README.md

Technology: LangChain, Groq, rich
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
from rich.progress import Progress, SpinnerColumn, TextColumn

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
MODEL_ID      = "llama-3.3-70b-versatile"
BASE_DIR      = Path(__file__).parent
ANALYSIS_FILE = BASE_DIR / "output" / "analysis_results.json"
OUTPUT_DIR    = BASE_DIR / "output"


# ============================================================================
# SYSTEM PROMPT – README SPECIALIST
# ============================================================================

SYSTEM_PROMPT = """\
You are a Senior Technical Writer and Software Architect.

Your task is to generate a professional, production-ready README.md \
for an open-source project based on the technical data provided.

Follow these rules strictly:
- Write clear, concise, and professional prose.
- Use proper Markdown formatting (headings, lists, code blocks, tables).
- Include an ASCII architecture diagram when describing the architecture.
- Do NOT hallucinate features or files that are not mentioned in the data.
- If data is missing for a section, state it honestly (e.g. "Not available").
- Keep the tone welcoming for new contributors.
- Be specific — reference actual module names, class names, and function counts.

Output ONLY valid Markdown. No surrounding commentary.\
"""


# ============================================================================
# LANGCHAIN LLM WRAPPER
# ============================================================================

class GroqLLMClient:
    """
    Uses LangChain ChatGroq for fast LLM inference.
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
            temperature=0.4,
            max_tokens=4096,
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
        """Send a system + user message through the Groq API."""
        response = self.client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return self._coerce_text(response.content)


# ============================================================================
# DATA COLLECTOR – Gather all project intelligence from analysis_results.json
# ============================================================================

class ProjectDataCollector:
    """
    Reads analysis_results.json and builds structured summaries
    suitable for the README generation prompt.
    """

    def __init__(self, data: Dict):
        self.data = data
        self.metadata = data.get("metadata", {})
        self.file_inventory = data.get("file_inventory", {})
        self.phase2 = data.get("phase2", {})
        self.phase3 = data.get("phase3", {})
        self.phase4 = data.get("phase4", {})

    # ── 1. Project Overview ──────────────────────────────────────────────────
    def project_overview(self) -> str:
        m = self.metadata
        lines = [
            f"Project Name   : {m.get('project_name', 'Unknown')}",
            f"GitHub URL     : {self.data.get('github_url', 'N/A')}",
            f"Primary Language: {m.get('primary_language', 'Unknown')}",
            f"Total Source Files: {m.get('total_source_files', 'N/A')}",
            f"Total Lines of Code: {m.get('total_lines_of_code', 'N/A')}",
        ]
        # Language breakdown
        lb = m.get("language_breakdown", {})
        if lb:
            lines.append(f"Language Breakdown: {json.dumps(lb)}")
        # Git metadata
        git = m.get("git_metadata", {})
        if git:
            lines.append(f"Active Branch: {git.get('active_branch', 'N/A')}")
            lines.append(f"Total Commits: {git.get('total_commits', 'N/A')}")
            lc = git.get("last_commit", {})
            if lc:
                lines.append(f"Last Commit: {lc.get('message', '').strip()[:80]} "
                             f"by {lc.get('author', 'N/A')} on {lc.get('date', 'N/A')}")
        # Readme preview
        rp = m.get("readme_preview", "")
        if rp:
            lines.append(f"\nOriginal README excerpt:\n{rp[:500]}")
        return "\n".join(lines)

    # ── 2. Architecture ─────────────────────────────────────────────────────
    def architecture_summary(self) -> str:
        arch = self.phase2.get("architecture", {})
        lines = []

        # Design patterns
        dp = arch.get("design_patterns", {})
        if dp:
            lines.append("Design Patterns Detected:")
            for pattern, usages in dp.items():
                lines.append(f"  - {pattern}: {', '.join(usages)}")

        # Layers
        layers = arch.get("layers", {})
        if layers:
            lines.append("\nArchitectural Layers:")
            for layer, modules in layers.items():
                lines.append(f"  {layer}: {', '.join(modules)}")

        # Data flow
        df = arch.get("data_flow", [])
        if df:
            lines.append("\nData Flow:")
            for edge in df[:15]:
                lines.append(f"  {edge.get('from', '?')} → {edge.get('to', '?')}")

        # Entry points
        ep = arch.get("entry_points", [])
        if ep:
            lines.append("\nEntry Points:")
            for item in ep:
                if isinstance(item, dict):
                    lines.append(f"  {item.get('file', '?')} ({item.get('type', 'unknown')})")
                else:
                    lines.append(f"  {item}")

        # Config handling
        ch = arch.get("config_handling", {})
        if ch:
            lines.append("\nConfig Handling:")
            if isinstance(ch, dict):
                for key, vals in ch.items():
                    if vals:
                        lines.append(f"  {key}: {', '.join(vals) if isinstance(vals, list) else vals}")
            elif isinstance(ch, list):
                lines.append(f"  {', '.join(str(c) for c in ch[:5])}")

        # Integrations
        integ = arch.get("integrations", {})
        if integ:
            lines.append("\nExternal Integrations:")
            if isinstance(integ, dict):
                for key, vals in integ.items():
                    if vals:
                        lines.append(f"  {key}: {', '.join(vals) if isinstance(vals, list) else vals}")
            elif isinstance(integ, list):
                lines.append(f"  {', '.join(str(i) for i in integ[:8])}")

        return "\n".join(lines) if lines else "No architecture data available."

    # ── 3. Core Components ──────────────────────────────────────────────────
    def core_components_summary(self) -> str:
        arch = self.phase2.get("architecture", {})
        comps = arch.get("core_components", [])
        lines = []
        for c in comps:
            mod = c.get("module", "?")
            defs = c.get("total_defs", 0)
            classes = c.get("classes", [])
            funcs = c.get("top_functions", [])
            parts = [f"Module: {mod} — {defs} definitions"]
            if classes:
                parts.append(f"  Classes: {', '.join(classes)}")
            if funcs:
                parts.append(f"  Functions: {', '.join(funcs)}")
            lines.append("\n".join(parts))
        return "\n".join(lines) if lines else "No component data."

    # ── 4. Dependency Summary ───────────────────────────────────────────────
    def dependency_summary(self) -> str:
        ga = self.phase2.get("graph_analysis", {})
        ext = ga.get("external_dependencies", {})
        lines = []

        # stdlib
        stdlib = ext.get("stdlib", {})
        if stdlib:
            top_std = sorted(stdlib.items(), key=lambda x: -x[1])[:10]
            lines.append("Standard Library (top usage):")
            for name, count in top_std:
                lines.append(f"  {name}: {count} imports")

        # third party
        tp = ext.get("third_party", {})
        if tp:
            top_tp = sorted(tp.items(), key=lambda x: -x[1])[:10]
            lines.append("\nThird-Party Dependencies:")
            for name, count in top_tp:
                lines.append(f"  {name}: {count} imports")

        # Internal dependency graph highlights
        core = ga.get("core_modules", [])
        if core:
            lines.append("\nCore Modules (by import count):")
            for m in core[:5]:
                lines.append(f"  {m.get('module', '?')}: imported by {m.get('imported_by_count', 0)} modules")

        coupled = ga.get("highly_coupled_modules", [])
        if coupled:
            lines.append("\nHighly Coupled Modules:")
            for m in coupled:
                lines.append(f"  {m.get('module', '?')}: fan_in={m.get('fan_in', 0)}, "
                             f"fan_out={m.get('fan_out', 0)}, coupling_score={m.get('coupling_score', 0)}")

        circ = ga.get("circular_dependencies", [])
        if circ:
            lines.append(f"\nCircular Dependencies: {len(circ)} cycle(s) detected")
            for cycle in circ[:5]:
                lines.append(f"  {' → '.join(cycle)}")
        else:
            lines.append("\nCircular Dependencies: None detected ✓")

        return "\n".join(lines) if lines else "No dependency data."

    # ── 5. File Statistics ──────────────────────────────────────────────────
    def file_statistics(self) -> str:
        inv = self.file_inventory.get("summary", {})
        lines = []
        if inv:
            lines.append(f"Total files scanned: {inv.get('total', 'N/A')}")
            cats = inv.get("categories", {})
            for cat, count in cats.items():
                lines.append(f"  {cat}: {count}")

        # Complexity metrics
        cm = self.phase2.get("complexity_metrics", {})
        pl = cm.get("project_level", {})
        if pl:
            lines.append(f"\nComplexity Metrics:")
            lines.append(f"  Average Cyclomatic Complexity: {pl.get('average_complexity', 'N/A')}")
            lines.append(f"  Max Complexity: {pl.get('max_complexity', 'N/A')}")
            hcf = pl.get("high_complexity_functions", [])
            if hcf:
                lines.append("  High-Complexity Functions:")
                for f in hcf:
                    label = f"{f.get('file', '?')}::{f.get('class', '')}.{f.get('method', f.get('function', ''))}"
                    lines.append(f"    {label} (CC={f.get('complexity', '?')})")

        # Per-file breakdown
        pf = cm.get("per_file", [])
        if pf:
            lines.append("\nPer-File Breakdown:")
            for entry in pf:
                lines.append(
                    f"  {entry.get('file', '?')}: "
                    f"{entry.get('loc', 0)} LOC, "
                    f"{entry.get('total_defs', 0)} defs, "
                    f"avg CC={entry.get('avg_complexity', 0)}, "
                    f"MI={entry.get('maintainability_index', 'N/A')}"
                )

        return "\n".join(lines) if lines else "No file statistics available."

    # ── 6. Module hierarchy ─────────────────────────────────────────────────
    def module_hierarchy(self) -> str:
        ga = self.phase2.get("graph_analysis", {})
        hier = ga.get("module_hierarchy", {})
        if not hier:
            return "No module hierarchy data."

        lines = []
        def _walk(tree, indent=0):
            for name, subtree in sorted(tree.items()):
                prefix = "  " * indent
                if isinstance(subtree, dict) and subtree:
                    lines.append(f"{prefix}{name}/")
                    _walk(subtree, indent + 1)
                else:
                    lines.append(f"{prefix}{name}")
        _walk(hier)
        return "\n".join(lines) if lines else "No module hierarchy."

    # ── 7. Phase 3 – Module-Level Docs Summary ──────────────────────────────
    def phase3_docs_summary(self) -> str:
        """Summarise Phase 3 generated module-level documentation."""
        p3 = self.phase3
        gen = p3.get("generated_docs", [])
        if not gen:
            return "No Phase 3 module documentation available."

        lines = [
            f"Phase 3 generated module-level docs for {p3.get('documented', len(gen))} modules.",
            f"Model: {p3.get('model', 'N/A')}  |  Engine: {p3.get('engine', 'N/A')}",
            "",
        ]

        # Read first ~300 chars of each generated doc as a preview
        docs_dir = BASE_DIR / "output" / "generated_docs"
        for entry in gen:
            fname = entry.get("doc_path", "")
            defs = entry.get("defs", 0)
            fpath = docs_dir / fname
            preview = ""
            if fpath.exists():
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
                # Grab first meaningful content (skip leading #/blank lines)
                content_lines = [l for l in raw.splitlines() if l.strip() and not l.startswith("---")]
                preview = "\n".join(content_lines[:8])  # first 8 lines
            lines.append(f"Module: {entry.get('file', fname)} ({defs} defs)")
            if preview:
                lines.append(f"  Preview: {preview[:300]}")
            lines.append("")

        return "\n".join(lines)

    # ── 8. Phase 4 – Function-Level Docs Summary ────────────────────────────
    def phase4_docs_summary(self) -> str:
        """Summarise Phase 4 function-level documentation reports."""
        p4 = self.phase4
        if not p4:
            return "No Phase 4 function-level documentation available."

        lines = [
            f"Phase 4 generated function-level docs for {p4.get('modules_processed', 0)} modules.",
            f"Total functions documented (Prompt 4.1 Docstrings): {p4.get('total_functions_documented', 0)}",
            f"Complex logic explanations (Prompt 4.3, CC≥{p4.get('complexity_threshold', 5)}): {p4.get('total_complex_explained', 0)}",
            f"Model: {p4.get('model', 'N/A')}  |  Engine: {p4.get('engine', 'N/A')}",
            "",
        ]

        # Read first ~300 chars of each module report as a preview
        reports_dir = BASE_DIR / "output" / "phase4_reports" / "modules"
        for entry in p4.get("module_reports", []):
            fname = entry.get("report_name", "")
            fdoc = entry.get("functions_documented", 0)
            fcplx = entry.get("complex_explained", 0)
            fpath = reports_dir / fname
            preview = ""
            if fpath.exists():
                raw = fpath.read_text(encoding="utf-8", errors="ignore")
                content_lines = [l for l in raw.splitlines() if l.strip() and not l.startswith("---")]
                preview = "\n".join(content_lines[:8])
            lines.append(f"Module: {entry.get('file', fname)} — {fdoc} funcs documented, {fcplx} complex explained")
            if preview:
                lines.append(f"  Preview: {preview[:300]}")
            lines.append("")

        return "\n".join(lines)

    # ── Build full context blob ─────────────────────────────────────────────
    def build_full_context(self) -> str:
        """Assemble all data sections into a single context string for the LLM."""
        sections = [
            ("PROJECT OVERVIEW", self.project_overview()),
            ("ARCHITECTURE ANALYSIS", self.architecture_summary()),
            ("CORE COMPONENTS", self.core_components_summary()),
            ("DEPENDENCY SUMMARY", self.dependency_summary()),
            ("FILE STATISTICS & COMPLEXITY", self.file_statistics()),
            ("MODULE HIERARCHY", self.module_hierarchy()),
            ("PHASE 3 – MODULE-LEVEL DOCUMENTATION", self.phase3_docs_summary()),
            ("PHASE 4 – FUNCTION-LEVEL DOCUMENTATION", self.phase4_docs_summary()),
        ]
        parts = []
        for title, content in sections:
            parts.append(f"=== {title} ===\n{content}")
        return "\n\n".join(parts)


# ============================================================================
# PROMPT BUILDER
# ============================================================================

README_USER_PROMPT = """\
Based on the following technical analysis data, generate a professional README.md.

Use this exact structure:
1. **Project Overview** — What the project does, its purpose, language, stats
2. **Architecture Overview** — High-level design, include an ASCII diagram showing module relationships
3. **Core Components** — List and describe each major module/class
4. **Dependency Summary** — External/internal dependencies, key libraries
5. **Installation Guide** — Step-by-step setup instructions (clone, install, etc.)
6. **Usage Example** — A realistic code example showing how to use the project
7. **Module Guide** — Brief description of each module file and what it contains
8. **Known Limitations** — Current issues, missing features, or areas needing improvement
9. **Development Notes** — How to contribute, run tests, code style, etc.

Technical Data:
{context}

Generate the full README.md now. Be thorough but concise.\
"""


# ============================================================================
# PHASE 5 ORCHESTRATOR
# ============================================================================

class Phase5:
    """
    Phase 5 – Professional README Generation

    Gathers all project intelligence from analysis_results.json,
    builds a comprehensive context, and uses the LLM to generate
    a production-ready README.md.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_analysis(self) -> Dict:
        """Load analysis_results.json."""
        if not ANALYSIS_FILE.exists():
            raise FileNotFoundError(
                f"analysis_results.json not found at {ANALYSIS_FILE}\n"
                "Run Phases 1-4 first."
            )
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> None:
        """Execute Phase 5."""
        console.print(Panel(
            "[bold cyan]Code-to-Doc  ·  Phase 5[/bold cyan]\n"
            "[white]Professional README Generation[/white]\n"
            f"[dim]Model: {MODEL_ID}  ·  Engine: Groq[/dim]",
            expand=False,
        ))

        # 1. Load data
        data = self._load_analysis()
        project_name = data.get("metadata", {}).get("project_name", "Unknown")
        console.print(f"\n📂 Project: [bold]{project_name}[/bold]")

        # 2. Collect all project intelligence
        collector = ProjectDataCollector(data)
        context = collector.build_full_context()
        console.print(f"📊 Context assembled: {len(context):,} characters")

        # Groq supports 128K context — much more room than HuggingFace 8B
        max_ctx = 60000
        if len(context) > max_ctx:
            context = context[:max_ctx] + "\n\n[... truncated for model context window ...]"
            console.print(f"[yellow]⚠ Context truncated to {max_ctx:,} chars[/yellow]")

        # 3. Initialize LLM
        llm = GroqLLMClient(self.api_key)

        # 4. Generate README
        console.print("\n🤖 Generating README.md via LLM…")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Calling LLM…", total=None)

            start = time.time()
            user_prompt = README_USER_PROMPT.format(context=context)
            readme_content = llm.generate(SYSTEM_PROMPT, user_prompt)
            elapsed = time.time() - start

            progress.update(task, description=f"Done in {elapsed:.1f}s")

        # 5. Post-process: ensure it starts with a heading
        readme_content = readme_content.strip()
        if not readme_content.startswith("#"):
            readme_content = f"# {project_name}\n\n{readme_content}"

        # 6. Save README.md
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        logger.info("README saved → %s", readme_path)

        # 7. Update analysis_results.json
        data["phase"] = 5
        data["phase5"] = {
            "model": MODEL_ID,
            "engine": "Groq API",
            "readme_path": str(readme_path),
            "context_length": len(context),
            "generation_time_s": round(elapsed, 2),
            "completed_at": datetime.now().isoformat(),
        }
        with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        # 8. Summary table
        table = Table(title="Phase 5 — README Generation")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Model", MODEL_ID)
        table.add_row("Engine", "Groq")
        table.add_row("Project", project_name)
        table.add_row("Context sent", f"{len(context):,} chars")
        table.add_row("Generation time", f"{elapsed:.1f}s")
        table.add_row("Output", str(readme_path))
        console.print(table)

        console.print(Panel(
            f"[green]✓ Phase 5 complete[/green]\n"
            f"README → {readme_path}",
            expand=False,
        ))


# ============================================================================
# MAIN
# ============================================================================

def main():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        console.print("[red]ERROR: GROQ_API_KEY not set. Add it to .env or environment.[/red]")
        raise SystemExit(1)

    phase5 = Phase5(api_key)
    phase5.run()


if __name__ == "__main__":
    main()
