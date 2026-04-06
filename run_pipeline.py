"""
Code-to-Doc: Full Pipeline Runner
===================================
Runs all phases sequentially from Phase 1 through inject_comments.

Usage:
    python run_pipeline.py <github_url>
    python run_pipeline.py                   # interactive prompt for URL

Phases:
    1. Clone & Inventory      → cloned_repo/, output/analysis_results.json
    2. AST & Architecture     → enriched analysis_results.json
    3. Module-Level Docs      → output/generated_docs/
    4. Function-Level Docs    → output/phase4_reports/
    5. README Generation      → output/README.md
    6. QA & Validation        → output/phase6_qa/
    7. Inject Comments        → output/documented_code/

Cleanup:
        - If a new repo URL is provided, old cloned_repo/ and output/
            are deleted before starting.
        - If the same repo URL is used again, cloned_repo/ is reused
            and only output/ is refreshed.
"""

import os
import sys
import json
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STATE_FILE = BASE_DIR / ".pipeline_state.json"
console = Console()


# ============================================================================
# CLEANUP – Remove old data before a fresh run
# ============================================================================

def cleanup(
    base_dir: Path,
    clean_clone: bool = True,
    clean_output: bool = True,
) -> None:
    """
    Remove old cloned_repo/ and/or output/ directories.

    Args:
        base_dir: workspace base directory.
        clean_clone: whether to delete cloned_repo/.
        clean_output: whether to delete output/.
    """
    dirs_to_clean = []
    if clean_clone:
        dirs_to_clean.append(base_dir / "cloned_repo")
    if clean_output:
        dirs_to_clean.append(base_dir / "output")

    cleaned = []
    for d in dirs_to_clean:
        if d.exists():
            try:
                shutil.rmtree(d)
                cleaned.append(str(d.name))
            except Exception as e:
                console.print(f"[yellow]⚠ Could not remove {d.name}: {e}[/yellow]")

    if cleaned:
        console.print(f"[dim]🗑  Cleaned: {', '.join(cleaned)}[/dim]")
    else:
        console.print("[dim]🗑  Nothing to clean — fresh workspace[/dim]")


def _normalize_repo_url(url: str) -> str:
    """Normalize repo URL for stable comparisons."""
    normalized = (url or "").strip().lower().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _load_last_repo_url(state_file: Path = STATE_FILE) -> str:
    """Load last successful repo URL from state file."""
    if not state_file.exists():
        return ""
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return str(data.get("last_repo_url", "")).strip()
    except Exception:
        return ""


def _save_last_repo_url(github_url: str, state_file: Path = STATE_FILE) -> None:
    """Persist last successful repo URL for next run cleanup decisions."""
    payload = {
        "last_repo_url": github_url.strip(),
        "updated_at": datetime.now().isoformat(),
    }
    state_file.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def prepare_workspace(base_dir: Path, github_url: str) -> None:
    """
    Clean workspace based on whether repo URL changed.

    - New repo URL: delete cloned_repo/ + output/
    - Same repo URL: keep cloned_repo/, delete output/
    """
    last_repo_url = _load_last_repo_url()
    last_norm = _normalize_repo_url(last_repo_url)
    current_norm = _normalize_repo_url(github_url)

    if last_repo_url and last_norm != current_norm:
        console.print("[yellow]🔁 New repository detected — removing previous clone and output.[/yellow]")
        cleanup(base_dir, clean_clone=True, clean_output=True)
        return

    if last_repo_url and last_norm == current_norm:
        console.print("[dim]🔄 Same repository detected — keeping cloned_repo/ and refreshing output/.[/dim]")
        cleanup(base_dir, clean_clone=False, clean_output=True)
        return

    # First run or missing state: full cleanup to avoid stale leftovers.
    cleanup(base_dir, clean_clone=True, clean_output=True)


# ============================================================================
# PHASE RUNNERS – Import and run each phase
# ============================================================================

def run_phase1(github_url: str) -> bool:
    """Phase 1: Clone repo & build file inventory."""
    from phase1 import Phase1
    try:
        github_token = os.environ.get("GITHUB_TOKEN")
        phase = Phase1(github_url, github_token)
        phase.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Phase 1 failed: {e}[/red]")
        traceback.print_exc()
        return False


def run_phase2() -> bool:
    """Phase 2: AST analysis, dependency graph, complexity metrics."""
    from phase2 import Phase2
    try:
        phase = Phase2()
        phase.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Phase 2 failed: {e}[/red]")
        traceback.print_exc()
        return False


def run_phase3() -> bool:
    """Phase 3: Module-level documentation (LLM)."""
    from phase3 import Phase3
    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            console.print("[red]✗ Phase 3: GROQ_API_KEY not set in environment or .env[/red]")
            return False
        phase = Phase3(api_key)
        phase.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Phase 3 failed: {e}[/red]")
        traceback.print_exc()
        return False


def run_phase4() -> bool:
    """Phase 4: Function-level documentation (LLM)."""
    from phase4 import Phase4
    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            console.print("[red]✗ Phase 4: GROQ_API_KEY not set in environment or .env[/red]")
            return False
        phase = Phase4(api_key)
        phase.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Phase 4 failed: {e}[/red]")
        traceback.print_exc()
        return False


def run_phase5() -> bool:
    """Phase 5: README generation (LLM)."""
    from phase5 import Phase5
    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            console.print("[red]✗ Phase 5: GROQ_API_KEY not set in environment or .env[/red]")
            return False
        phase = Phase5(api_key)
        phase.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Phase 5 failed: {e}[/red]")
        traceback.print_exc()
        return False


def run_phase6() -> bool:
    """Phase 6: QA & validation (LLM)."""
    from phase6 import Phase6
    try:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            console.print("[red]✗ Phase 6: GROQ_API_KEY not set in environment or .env[/red]")
            return False
        phase = Phase6(api_key)
        phase.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Phase 6 failed: {e}[/red]")
        traceback.print_exc()
        return False


def run_inject_comments() -> bool:
    """Inject generated docstrings + section comments into source code."""
    from inject_comments import InjectComments
    try:
        injector = InjectComments()
        injector.run()
        return True
    except Exception as e:
        console.print(f"[red]✗ Inject Comments failed: {e}[/red]")
        traceback.print_exc()
        return False


# ============================================================================
# PIPELINE DEFINITION
# ============================================================================

PHASES = [
    {
        "name": "Phase 1 — Clone & Inventory",
        "runner": None,  # special: needs github_url arg
        "needs_llm": False,
        "critical": True,   # pipeline cannot continue without this
    },
    {
        "name": "Phase 2 — AST & Architecture",
        "runner": run_phase2,
        "needs_llm": False,
        "critical": True,
    },
    {
        "name": "Phase 3 — Module-Level Docs",
        "runner": run_phase3,
        "needs_llm": True,
        "critical": False,   # Phase 4 can still run if Phase 3 fails
    },
    {
        "name": "Phase 4 — Function-Level Docs",
        "runner": run_phase4,
        "needs_llm": True,
        "critical": True,    # inject_comments needs Phase 4 output
    },
    {
        "name": "Phase 5 — README Generation",
        "runner": run_phase5,
        "needs_llm": True,
        "critical": False,
    },
    {
        "name": "Phase 6 — QA & Validation",
        "runner": run_phase6,
        "needs_llm": True,
        "critical": False,
    },
    {
        "name": "Inject Comments",
        "runner": run_inject_comments,
        "needs_llm": False,
        "critical": False,
    },
]


class PipelineState(TypedDict):
    github_url: str
    results: List[Dict]
    stop: bool


def _run_single_phase_graph(
    state: PipelineState,
    step_num: int,
    phase_name: str,
    runner,
    critical: bool,
) -> PipelineState:
    console.print(f"\n{'='*60}")
    console.print(f"[bold cyan]Step {step_num}/{len(PHASES)}: {phase_name}[/bold cyan]")
    console.print(f"{'='*60}\n")

    start = time.time()
    success = runner()
    elapsed = time.time() - start

    state["results"].append({
        "step": step_num,
        "name": phase_name,
        "success": success,
        "elapsed": elapsed,
        "critical": critical,
    })

    if success:
        console.print(f"\n[green]✓ {phase_name} completed in {elapsed:.1f}s[/green]")
    else:
        console.print(f"\n[red]✗ {phase_name} failed after {elapsed:.1f}s[/red]")
        if critical:
            console.print("[red bold]⚠ Critical phase failed — stopping pipeline.[/red bold]")
            state["stop"] = True
        else:
            console.print("[yellow]⚠ Non-critical — continuing to next phase…[/yellow]")

    return state


def _build_graph_app():
    graph = StateGraph(PipelineState)

    def phase1_node(state: PipelineState) -> PipelineState:
        def _runner() -> bool:
            ok = run_phase1(state["github_url"])
            if ok:
                _save_last_repo_url(state["github_url"])
            return ok

        return _run_single_phase_graph(
            state=state,
            step_num=1,
            phase_name=PHASES[0]["name"],
            runner=_runner,
            critical=PHASES[0]["critical"],
        )

    graph.add_node("phase1", phase1_node)

    node_names = ["phase1"]
    for idx, phase in enumerate(PHASES[1:], start=2):
        node_name = f"phase{idx}"

        def _make_node(step_num: int, phase_def: Dict):
            def _node(state: PipelineState) -> PipelineState:
                return _run_single_phase_graph(
                    state=state,
                    step_num=step_num,
                    phase_name=phase_def["name"],
                    runner=phase_def["runner"],
                    critical=phase_def["critical"],
                )

            return _node

        graph.add_node(node_name, _make_node(idx, phase))
        node_names.append(node_name)

    def _route(state: PipelineState) -> str:
        return "end" if state.get("stop") else "next"

    for idx, current in enumerate(node_names):
        mapping = {"end": END}
        if idx + 1 < len(node_names):
            mapping["next"] = node_names[idx + 1]
        else:
            mapping["next"] = END
        graph.add_conditional_edges(current, _route, mapping)

    graph.set_entry_point("phase1")
    return graph.compile()


def run_langgraph_pipeline(github_url: str) -> int:
    console.print(Panel(
        "[bold cyan]Code-to-Doc  ·  LangGraph Mode[/bold cyan]\n"
        "[white]Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Inject Comments[/white]\n"
        f"[dim]Working dir: {BASE_DIR}[/dim]",
        expand=False,
    ))

    console.print(f"\n🔗 Repository: [bold]{github_url}[/bold]")
    console.print("\n[bold]Step 0: Workspace Prep[/bold]")
    prepare_workspace(BASE_DIR, github_url)

    app = _build_graph_app()
    initial_state: PipelineState = {
        "github_url": github_url,
        "results": [],
        "stop": False,
    }

    start = time.time()
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        console.print(f"[red]✗ LangGraph execution failed: {e}[/red]")
        traceback.print_exc()
        return 1

    results = final_state["results"]
    total_elapsed = time.time() - start

    console.print(f"\n{'='*60}")
    console.print("[bold cyan]Pipeline Summary[/bold cyan]")
    console.print(f"{'='*60}\n")

    table = Table(title=f"Code-to-Doc Pipeline — {github_url.split('/')[-1]}")
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Phase", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for r in results:
        status = "[green]✓ Pass[/green]" if r["success"] else "[red]✗ Fail[/red]"
        table.add_row(
            str(r["step"]),
            r["name"],
            status,
            f"{r['elapsed']:.1f}s",
        )

    console.print(table)
    console.print(f"\n⏱  Total time: [bold]{total_elapsed:.1f}s[/bold]")

    all_passed = all(r["success"] for r in results)
    if all_passed:
        console.print(Panel(
            "[bold green]✓ Pipeline completed successfully![/bold green]\n"
            f"Repository: {github_url}\n"
            f"Total time: {total_elapsed:.1f}s\n"
            f"Output: {BASE_DIR / 'output'}",
            expand=False,
        ))
        return 0

    failed = [r["name"] for r in results if not r["success"]]
    console.print(Panel(
        f"[bold yellow]⚠ Pipeline completed with errors[/bold yellow]\n"
        f"Failed: {', '.join(failed)}\n"
        f"Total time: {total_elapsed:.1f}s",
        expand=False,
    ))
    return 1


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full Code-to-Doc pipeline."""

    # Load .env for GROQ_API_KEY
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        # Also try project2 .env as fallback
        p2_env = BASE_DIR.parent / "project2" / ".env"
        if p2_env.exists():
            load_dotenv(p2_env)
    except ImportError:
        pass

    # ── Banner ──────────────────────────────────────────────────────
    console.print(Panel(
        "[bold cyan]Code-to-Doc  ·  Full Pipeline[/bold cyan]\n"
        "[white]Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Inject Comments[/white]\n"
        f"[dim]Working dir: {BASE_DIR}[/dim]",
        expand=False,
    ))

    # ── Get GitHub URL ──────────────────────────────────────────────
    if len(sys.argv) > 1:
        github_url = sys.argv[1].strip()
    else:
        github_url = console.input("\n[bold]Enter GitHub repository URL:[/bold] ").strip()

    if not github_url:
        console.print("[red]✗ No GitHub URL provided. Exiting.[/red]")
        sys.exit(1)

    if not github_url.startswith("http"):
        console.print("[red]✗ Invalid URL. Please provide a full GitHub URL (https://github.com/...)[/red]")
        sys.exit(1)

    if "--graph" in sys.argv:
        sys.exit(run_langgraph_pipeline(github_url))

    console.print(f"\n🔗 Repository: [bold]{github_url}[/bold]")

    # ── Cleanup old data ────────────────────────────────────────────
    console.print("\n[bold]Step 0: Workspace Prep[/bold]")
    prepare_workspace(BASE_DIR, github_url)
    console.print()

    # ── Run all phases ──────────────────────────────────────────────
    results = []
    pipeline_start = time.time()

    for i, phase in enumerate(PHASES):
        phase_name = phase["name"]
        step_num = i + 1

        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]Step {step_num}/{len(PHASES)}: {phase_name}[/bold cyan]")
        console.print(f"{'='*60}\n")

        start = time.time()

        # Phase 1 is special — needs the URL argument
        if i == 0:
            success = run_phase1(github_url)
            if success:
                _save_last_repo_url(github_url)
        else:
            success = phase["runner"]()

        elapsed = time.time() - start

        results.append({
            "step": step_num,
            "name": phase_name,
            "success": success,
            "elapsed": elapsed,
            "critical": phase["critical"],
        })

        if success:
            console.print(f"\n[green]✓ {phase_name} completed in {elapsed:.1f}s[/green]")
        else:
            console.print(f"\n[red]✗ {phase_name} failed after {elapsed:.1f}s[/red]")
            if phase["critical"]:
                console.print("[red bold]⚠ Critical phase failed — stopping pipeline.[/red bold]")
                break
            else:
                console.print("[yellow]⚠ Non-critical — continuing to next phase…[/yellow]")

    total_elapsed = time.time() - pipeline_start

    # ── Summary ─────────────────────────────────────────────────────
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]Pipeline Summary[/bold cyan]")
    console.print(f"{'='*60}\n")

    table = Table(title=f"Code-to-Doc Pipeline — {github_url.split('/')[-1]}")
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Phase", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for r in results:
        status = "[green]✓ Pass[/green]" if r["success"] else "[red]✗ Fail[/red]"
        table.add_row(
            str(r["step"]),
            r["name"],
            status,
            f"{r['elapsed']:.1f}s",
        )

    console.print(table)
    console.print(f"\n⏱  Total time: [bold]{total_elapsed:.1f}s[/bold]")

    # ── Output summary ──────────────────────────────────────────────
    output_dir = BASE_DIR / "output"
    if output_dir.exists():
        console.print("\n[bold]Output Files:[/bold]")
        key_outputs = [
            ("analysis_results.json", "Full analysis data (Phases 1-6)"),
            ("generated_docs/", "Module-level docs (Phase 3)"),
            ("phase4_reports/", "Function-level docs (Phase 4)"),
            ("README.md", "Generated README (Phase 5)"),
            ("phase6_qa/", "QA report (Phase 6)"),
            ("documented_code/", "Source code with injected comments"),
        ]
        for name, desc in key_outputs:
            path = output_dir / name
            if path.exists():
                console.print(f"  [green]✓[/green] output/{name:30s} {desc}")
            else:
                console.print(f"  [dim]·[/dim] output/{name:30s} [dim](not generated)[/dim]")

    # ── Final status ────────────────────────────────────────────────
    all_passed = all(r["success"] for r in results)
    if all_passed:
        console.print(Panel(
            "[bold green]✓ Pipeline completed successfully![/bold green]\n"
            f"Repository: {github_url}\n"
            f"Total time: {total_elapsed:.1f}s\n"
            f"Output: {output_dir}",
            expand=False,
        ))
    else:
        failed = [r["name"] for r in results if not r["success"]]
        console.print(Panel(
            f"[bold yellow]⚠ Pipeline completed with errors[/bold yellow]\n"
            f"Failed: {', '.join(failed)}\n"
            f"Total time: {total_elapsed:.1f}s",
            expand=False,
        ))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
