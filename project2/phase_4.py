"""
Code-to-Doc: Phase 4 - Multi-Agent Code Analysis Pipeline
Uses Meta-Llama-3-8B-Instruct via Hugging Face Inference API.

Agents:
  1. RefactoringAgent        – Suggests code improvements
  2. CodeSmellDetector       – Identifies anti-patterns
  3. BestPracticesAnalyzer   – Checks standards & security
  4. ConsolidationAgent      – Merges findings, builds final report

Input  : output/analysis_results.json  (from Phase 1+2)
Output : output/phase4_reports/        (per-module .md + master report)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_ID       = "meta-llama/Meta-Llama-3-8B-Instruct"
ANALYSIS_FILE  = Path(__file__).parent / "output" / "analysis_results.json"
OUTPUT_DIR     = Path(__file__).parent / "output" / "phase4_reports"
MAX_CODE_CHARS = 5000   # keep prompts within context window


# ============================================================================
# SHARED: LLM CLIENT  (same as Phase 3)
# ============================================================================

class LlamaClient:
    """Wraps Hugging Face InferenceClient for Llama-3-8B-Instruct."""

    def __init__(self, hf_token: str):
        if not hf_token:
            raise ValueError(
                "Hugging Face API token is missing.\n"
                "Set HF_TOKEN in your environment or .env file."
            )
        self.client = InferenceClient(model=MODEL_ID, token=hf_token)
        logger.info(f"LlamaClient ready → {MODEL_ID}")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024) -> str:
        """Call the model and return the generated text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ============================================================================
# AGENT 1: REFACTORING AGENT
# ============================================================================

class RefactoringAgent:
    """
    Analyzes functions and classes for refactoring opportunities.
    Suggests cleaner, more Pythonic, or more maintainable equivalents.
    """

    SYSTEM_PROMPT = (
        "You are an expert Python refactoring engineer. "
        "You review code and provide concrete, actionable refactoring suggestions. "
        "Be specific: name the problem, explain why it is a problem, and show "
        "improved code. Output in Markdown only."
    )

    def __init__(self, llm: LlamaClient):
        self.llm = llm

    def analyze(self, file_path: str, module_data: Dict, source_code: str) -> str:
        """
        Run refactoring analysis on one module.

        Args:
            file_path:   Relative file path (used in headings).
            module_data: Parsed AST data (functions, classes, complexity).
            source_code: Raw source code (truncated).

        Returns:
            Markdown string with refactoring suggestions.
        """
        logger.info(f"    [RefactoringAgent] → {file_path}")

        # Summarize high-complexity functions for the prompt
        complex_fns = [
            f"- `{fn.get('signature', fn['name'])}` "
            f"(complexity={fn.get('complexity', '?')}, "
            f"lines {fn.get('lineno', '?')}–{fn.get('end_lineno', '?')})"
            for fn in module_data.get("functions", [])
            if (fn.get("complexity") or 0) >= 3
        ]

        # Also include class methods with high complexity
        for cls in module_data.get("classes", []):
            for m in cls.get("methods", []):
                if (m.get("complexity") or 0) >= 3:
                    complex_fns.append(
                        f"- `{cls['name']}.{m.get('signature', m['name'])}` "
                        f"(complexity={m.get('complexity', '?')})"
                    )

        user_prompt = f"""## Refactoring Review for `{file_path}`

### High-Complexity Functions / Methods Detected
{chr(10).join(complex_fns) if complex_fns else "- None above threshold"}

### Source Code (truncated to {MAX_CODE_CHARS} chars)
```python
{source_code[:MAX_CODE_CHARS]}
```

### Your Task
1. Identify the top 3-5 refactoring opportunities (e.g., long methods, repeated
   logic, poor naming, overly complex conditionals, missing abstractions).
2. For each issue:
   - **Problem**: clearly state what is wrong.
   - **Location**: function / class name + approximate line number.
   - **Suggestion**: describe the improvement.
   - **Improved Code**: show a short before/after snippet.
3. Assign a priority: 🔴 High / 🟡 Medium / 🟢 Low.

Output clean Markdown only.
"""
        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 1200)
        except Exception as e:
            logger.error(f"    RefactoringAgent failed for {file_path}: {e}")
            return f"## Refactoring Analysis\n\n> ⚠️ Analysis failed: {e}\n"


# ============================================================================
# AGENT 2: CODE SMELL DETECTOR
# ============================================================================

class CodeSmellDetector:
    """
    Detects code smells: anti-patterns, structural problems and
    quality red-flags that make code harder to maintain.
    """

    SYSTEM_PROMPT = (
        "You are a code quality expert specialising in code smell detection. "
        "Detect anti-patterns and structural problems in the given code. "
        "Be precise: name each smell, explain the symptoms, quantify where "
        "possible, and recommend a fix. Output in Markdown only."
    )

    # Rules applied before sending to LLM (fast heuristics)
    MAX_PARAMS      = 5
    MAX_COMPLEXITY  = 10
    MAX_LINES       = 60

    def __init__(self, llm: LlamaClient):
        self.llm = llm

    def _heuristic_smells(self, module_data: Dict) -> List[Dict]:
        """
        Fast rule-based smell detection (no LLM token cost).
        Returns list of smell dicts: {type, severity, location, detail}
        """
        smells: List[Dict] = []

        all_functions = list(module_data.get("functions", []))
        for cls in module_data.get("classes", []):
            for m in cls.get("methods", []):
                all_functions.append({**m, "_in_class": cls["name"]})

        for fn in all_functions:
            name     = fn.get("name", "?")
            loc      = f"{fn.get('_in_class', '')}.{name}" if fn.get('_in_class') else name
            params   = len(fn.get("parameters", []))
            cplx     = fn.get("complexity") or 0
            start    = fn.get("lineno") or 0
            end      = fn.get("end_lineno") or 0
            fn_lines = (end - start) if (start and end) else 0

            if params > self.MAX_PARAMS:
                smells.append({
                    "type": "Too Many Parameters",
                    "severity": "HIGH",
                    "location": loc,
                    "detail": f"{params} parameters (max recommended: {self.MAX_PARAMS})",
                })

            if cplx > self.MAX_COMPLEXITY:
                smells.append({
                    "type": "High Cyclomatic Complexity",
                    "severity": "HIGH",
                    "location": loc,
                    "detail": f"Complexity = {cplx} (max recommended: {self.MAX_COMPLEXITY})",
                })

            if fn_lines > self.MAX_LINES:
                smells.append({
                    "type": "Long Method",
                    "severity": "MEDIUM",
                    "location": loc,
                    "detail": f"{fn_lines} lines (max recommended: {self.MAX_LINES})",
                })

            if fn.get("is_private") is False and not fn.get("docstring"):
                smells.append({
                    "type": "Missing Docstring",
                    "severity": "LOW",
                    "location": loc,
                    "detail": "Public function/method lacks a docstring.",
                })

        # God class check
        for cls in module_data.get("classes", []):
            if len(cls.get("methods", [])) > 15:
                smells.append({
                    "type": "God Class",
                    "severity": "HIGH",
                    "location": cls["name"],
                    "detail": f"{len(cls['methods'])} methods — class has too many responsibilities.",
                })

        return smells

    def analyze(self, file_path: str, module_data: Dict, source_code: str) -> str:
        """
        Run code smell detection on one module.

        Returns:
            Markdown report combining heuristic + LLM analysis.
        """
        logger.info(f"    [CodeSmellDetector] → {file_path}")

        heuristic = self._heuristic_smells(module_data)

        # Format heuristic findings for the prompt
        h_lines = []
        for s in heuristic:
            icon = "🔴" if s["severity"] == "HIGH" else "🟡" if s["severity"] == "MEDIUM" else "🟢"
            h_lines.append(
                f"- {icon} **{s['type']}** in `{s['location']}`: {s['detail']}"
            )

        user_prompt = f"""## Code Smell Detection for `{file_path}`

### Heuristic Pre-Analysis (rule-based)
{chr(10).join(h_lines) if h_lines else "- No rule-based smells detected."}

### Source Code (truncated to {MAX_CODE_CHARS} chars)
```python
{source_code[:MAX_CODE_CHARS]}
```

### Your Task
Review the code carefully and identify ALL code smells, including:
- **Long methods / Large classes** (too many lines / methods)
- **Duplicate code / Copy-paste patterns**
- **Deep nesting** (>3 levels of if/for/while)
- **Magic numbers / Hard-coded strings**
- **Dead code** (unreachable or never-called)
- **Excessive comments** (comment explaining obvious things)
- **Feature envy** (method uses data from another class more than its own)
- **Data clumps** (groups of data always passed together)

For each smell:
| Field      | Content |
|------------|---------|
| Smell Type | name    |
| Severity   | 🔴/🟡/🟢 |
| Location   | function/class/line |
| Description | symptoms observed |
| Fix        | recommended approach |

Then provide an **Overall Code Quality Score** (1-10) with a brief justification.

Output clean Markdown only.
"""
        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 1200)
        except Exception as e:
            logger.error(f"    CodeSmellDetector failed for {file_path}: {e}")
            return f"## Code Smell Analysis\n\n> ⚠️ Analysis failed: {e}\n"


# ============================================================================
# AGENT 3: BEST PRACTICES ANALYZER
# ============================================================================

class BestPracticesAnalyzer:
    """
    Checks adherence to coding standards:
    type hints, docstrings, error handling, logging,
    security vulnerabilities, and performance issues.
    """

    SYSTEM_PROMPT = (
        "You are a senior code reviewer specialising in Python best practices, "
        "security auditing, and performance optimisation. "
        "Review the given code and produce a detailed checklist-style report. "
        "Flag CRITICAL security issues prominently. Output in Markdown only."
    )

    def __init__(self, llm: LlamaClient):
        self.llm = llm

    def _quick_checks(self, module_data: Dict, source_code: str) -> List[str]:
        """Fast checks that don't require LLM."""
        findings: List[str] = []
        source_lower = source_code.lower()

        # Check type hint coverage
        all_fns = list(module_data.get("functions", []))
        for cls in module_data.get("classes", []):
            all_fns.extend(cls.get("methods", []))

        untyped = [
            fn["name"] for fn in all_fns
            if not fn.get("is_private")
            and any(
                p.get("type") in (None, "Any")
                for p in fn.get("parameters", [])
                if p["name"] not in ("self", "cls")
            )
        ]
        if untyped:
            findings.append(
                f"🟡 **Missing Type Hints** on {len(untyped)} public function(s): "
                + ", ".join(f"`{n}`" for n in untyped[:5])
                + ("…" if len(untyped) > 5 else "")
            )

        # Bare except
        if "except:" in source_code:
            findings.append(
                "🔴 **Bare `except:`** detected — catches ALL exceptions including "
                "`SystemExit` and `KeyboardInterrupt`. Use specific exception types."
            )

        # print() usage
        import re
        print_calls = len(re.findall(r'\bprint\s*\(', source_code))
        if print_calls > 0:
            findings.append(
                f"🟢 **{print_calls}× `print()` call(s)** — replace with `logging` "
                "for production code."
            )

        # SQL injection pattern
        if re.search(r'execute\s*\(\s*f["\']|execute\s*\(\s*["\'].*%\s*', source_code):
            findings.append(
                "🔴 **Potential SQL Injection** — string-formatted SQL query detected. "
                "Use parameterised queries instead."
            )

        # Hard-coded secrets
        if re.search(r'(password|secret|api_key|token)\s*=\s*["\'][^"\']{4,}', source_lower):
            findings.append(
                "🔴 **Hard-coded Credential** detected — move secrets to environment "
                "variables or a secrets manager."
            )

        return findings

    def analyze(self, file_path: str, module_data: Dict, source_code: str) -> str:
        """
        Run best-practices analysis on one module.

        Returns:
            Markdown report with best-practice findings.
        """
        logger.info(f"    [BestPracticesAnalyzer] → {file_path}")

        quick = self._quick_checks(module_data, source_code)

        user_prompt = f"""## Best Practices Review for `{file_path}`

### Quick-Scan Findings (automated checks)
{chr(10).join(quick) if quick else "- No automated flags raised."}

### Source Code (truncated to {MAX_CODE_CHARS} chars)
```python
{source_code[:MAX_CODE_CHARS]}
```

### Your Task — Full Best-Practices Audit

Check the code against these categories and report findings for each:

#### 1. Code Style & Readability
- PEP 8 compliance (naming conventions, line length, spacing)
- Meaningful variable/function names
- Unnecessary complexity

#### 2. Documentation
- Module, class and function docstrings (present and informative?)
- Inline comments (accurate and non-obvious?)

#### 3. Type Safety
- Type annotations on function parameters and return types
- Use of Optional, Union, etc. where appropriate

#### 4. Error Handling
- Specific vs bare except clauses
- Proper exception propagation
- Resource cleanup (context managers / try-finally)

#### 5. Security
- Input validation
- SQL injection / command injection risks
- Hard-coded secrets or credentials
- Insecure deserialization

#### 6. Performance
- Inefficient algorithms (O(n²) where O(n) is possible)
- Redundant computations inside loops
- Memory leaks or large object retention

#### 7. Testability
- Pure functions vs side-effect-laden functions
- Global state usage
- Mocking difficulty

For each issue, assign:  🔴 Critical | 🟡 Medium | 🟢 Low

End with a **Best-Practices Compliance Score (1-10)** and a 2-sentence verdict.

Output clean Markdown only.
"""
        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 1400)
        except Exception as e:
            logger.error(f"    BestPracticesAnalyzer failed for {file_path}: {e}")
            return f"## Best Practices Analysis\n\n> ⚠️ Analysis failed: {e}\n"


# ============================================================================
# AGENT 4: CONSOLIDATION AGENT
# ============================================================================

class ConsolidationAgent:
    """
    Combines outputs from Agents 1-3 into a single, prioritised
    module report and contributes to the master codebase report.
    """

    SYSTEM_PROMPT = (
        "You are a Principal Engineer conducting a final code-review synthesis. "
        "You receive three analysis reports for the same file "
        "(Refactoring, Code Smells, Best Practices) and must produce a single "
        "consolidated report. Eliminate duplicates, prioritise by impact, and "
        "create an actionable improvement plan. Output in Markdown only."
    )

    def __init__(self, llm: LlamaClient):
        self.llm = llm

    def consolidate(
        self,
        file_path: str,
        refactoring_md: str,
        smells_md: str,
        practices_md: str,
    ) -> str:
        """
        Merge the three agent reports into one cohesive module report.

        Returns:
            Consolidated Markdown report.
        """
        logger.info(f"    [ConsolidationAgent] → {file_path}")

        user_prompt = f"""## Consolidation Task for `{file_path}`

You have received three separate analysis reports. Your job is to synthesise them.

---
### REPORT A — Refactoring Suggestions
{refactoring_md[:2000]}

---
### REPORT B — Code Smell Detection
{smells_md[:2000]}

---
### REPORT C — Best Practices Audit
{practices_md[:2000]}

---

### Your Consolidated Output Must Include:

#### 1. Executive Summary (3-4 sentences)
What is the overall state of this file? What are the most pressing concerns?

#### 2. Critical Issues Table (immediate action required)
| # | Issue | Source | Severity | Effort to Fix |
|---|-------|--------|----------|---------------|

#### 3. High Priority Improvements
Numbered list with description and recommended fix.

#### 4. Medium / Low Priority Improvements
Brief bullet list.

#### 5. Estimated Technical Debt
- Estimated fix time: X hours / days
- Code Quality Score: X/10
- Maintainability Index: Low / Medium / High

#### 6. Recommended Action Plan
Week-by-week improvement roadmap (max 4 weeks).

Output clean Markdown only.
"""
        try:
            return self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 1400)
        except Exception as e:
            logger.error(f"    ConsolidationAgent failed for {file_path}: {e}")
            return f"## Consolidated Report\n\n> ⚠️ Consolidation failed: {e}\n"


# ============================================================================
# PHASE 4 ORCHESTRATOR
# ============================================================================

class Phase4MultiAgentOrchestrator:
    """
    Drives all 4 agents across every module in analysis_results.json.

    Workflow per module:
        1. RefactoringAgent    → refactoring_md
        2. CodeSmellDetector   → smells_md
        3. BestPracticesAnalyzer → practices_md
        4. ConsolidationAgent  → consolidated_md  (saved as module report)

    Then generates:
        - Per-module Markdown reports in output/phase4_reports/modules/
        - A master codebase report   in output/phase4_reports/MASTER_REPORT.md
    """

    def __init__(
        self,
        hf_token: str,
        analysis_file: Path = ANALYSIS_FILE,
        output_dir: Path = OUTPUT_DIR,
    ):
        llm = LlamaClient(hf_token)

        self.refactoring_agent   = RefactoringAgent(llm)
        self.smell_detector      = CodeSmellDetector(llm)
        self.practices_analyzer  = BestPracticesAnalyzer(llm)
        self.consolidation_agent = ConsolidationAgent(llm)

        self.analysis_file = analysis_file
        self.output_dir    = output_dir
        self.modules_dir   = output_dir / "modules"
        self.modules_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _load_analysis(self) -> Dict:
        """Load and validate analysis_results.json."""
        if not self.analysis_file.exists():
            raise FileNotFoundError(
                f"analysis_results.json not found at {self.analysis_file}\n"
                "Run phase_1_2_claude.py first."
            )
        with open(self.analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded analysis from {self.analysis_file}")
        return data

    def _read_source(self, absolute_path: str) -> str:
        """Read raw source code from disk."""
        try:
            with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read {absolute_path}: {e}")
            return ""

    def _save(self, path: Path, content: str):
        """Write text content to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"    Saved → {path}")

    # ── Single module ─────────────────────────────────────────────────────────

    def _analyze_module(
        self,
        file_path: str,
        module_data: Dict,
        absolute_path: str,
    ) -> Tuple[str, str, str, str]:
        """
        Run all 4 agents on one module.

        Returns:
            (refactoring_md, smells_md, practices_md, consolidated_md)
        """
        source = self._read_source(absolute_path)

        refactoring_md  = self.refactoring_agent.analyze(file_path, module_data, source)
        smells_md       = self.smell_detector.analyze(file_path, module_data, source)
        practices_md    = self.practices_analyzer.analyze(file_path, module_data, source)
        consolidated_md = self.consolidation_agent.consolidate(
            file_path, refactoring_md, smells_md, practices_md
        )

        return refactoring_md, smells_md, practices_md, consolidated_md

    def _save_module_report(
        self,
        file_path: str,
        refactoring_md: str,
        smells_md: str,
        practices_md: str,
        consolidated_md: str,
    ) -> Path:
        """Save full module report (all 4 sections) to a single .md file."""
        safe_name = file_path.replace("/", "_").replace("\\", "_") + ".md"
        out_path  = self.modules_dir / safe_name

        content = "\n\n---\n\n".join([
            f"# Phase 4 Analysis: `{file_path}`\n\n"
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            f"## 🔧 Refactoring Suggestions\n\n{refactoring_md}",
            f"## 👃 Code Smell Detection\n\n{smells_md}",
            f"## ✅ Best Practices Audit\n\n{practices_md}",
            f"## 📊 Consolidated Report\n\n{consolidated_md}",
        ])

        self._save(out_path, content)
        return out_path

    # ── Master report ─────────────────────────────────────────────────────────

    def _build_master_report(
        self,
        analysis: Dict,
        module_summaries: List[Dict],
    ) -> str:
        """Build the codebase-wide MASTER_REPORT.md."""
        phase1   = analysis.get("phase1", {})
        metadata = phase1.get("metadata", {})
        metrics  = analysis.get("phase2", {}).get("complexity_metrics", {})
        arch     = analysis.get("phase2", {}).get("architecture", {})

        total_modules = len(module_summaries)
        processed     = sum(1 for m in module_summaries if m.get("processed"))
        skipped       = total_modules - processed

        lines = [
            "# Code-to-Doc: Phase 4 Master Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Model:** `{MODEL_ID}`  ",
            f"**Modules Analysed:** {processed} / {total_modules}  ",
            "",
            "---",
            "",
            "## Repository Overview",
            "",
            "| Key         | Value |",
            "|-------------|-------|",
            f"| Name        | {metadata.get('name', 'N/A')} |",
            f"| URL         | {metadata.get('url', 'N/A')} |",
            f"| Branch      | {metadata.get('branch', 'N/A')} |",
            f"| Commit      | {metadata.get('commit', 'N/A')} |",
            f"| Last Commit | {metadata.get('last_commit', 'N/A')} |",
            "",
            "## Codebase Metrics",
            "",
            "| Metric          | Value |",
            "|-----------------|-------|",
            f"| Total Files     | {metrics.get('total_files', 0)} |",
            f"| Total Lines     | {metrics.get('total_lines', 0)} |",
            f"| Avg File Size   | {metrics.get('average_file_size', 0):.0f} bytes |",
            f"| Circular Deps   | {len(metrics.get('circular_dependencies', []))} |",
            "",
            "## Architecture",
            f"- **Entry Points:** {', '.join(arch.get('entry_points', [])) or 'N/A'}",
            f"- **Core Modules:** {', '.join(arch.get('core_modules', [])[:5]) or 'N/A'}",
            "",
            "---",
            "",
            "## Module Analysis Index",
            "",
            "| # | Module | Language | Lines | Report |",
            "|---|--------|----------|-------|--------|",
        ]

        for i, m in enumerate(module_summaries, 1):
            status   = "✅" if m.get("processed") else "⏭ skipped"
            doc_name = Path(m["report_path"]).name if m.get("report_path") else "—"
            link     = f"[{doc_name}](modules/{doc_name})" if m.get("report_path") else "—"
            lines.append(
                f"| {i} | `{m['file']}` | {m.get('language','?')} "
                f"| {m.get('lines', 0)} | {link} |"
            )

        lines += [
            "",
            "---",
            "",
            "## Circular Dependencies Detected",
            "",
        ]

        circulars = metrics.get("circular_dependencies", [])
        if circulars:
            for cycle in circulars:
                lines.append(f"- {' → '.join(cycle)}")
        else:
            lines.append("✅ No circular dependencies detected.")

        lines += [
            "",
            "---",
            "",
            "## Design Patterns Detected",
            "",
        ]

        patterns = arch.get("design_patterns", {})
        for pattern, files in patterns.items():
            if files:
                lines.append(f"- **{pattern.title()}**: {', '.join(files)}")
        if not any(patterns.values()):
            lines.append("- No named patterns detected via heuristics.")

        lines += [
            "",
            "---",
            "",
            "## Architectural Layers",
            "",
            "| Layer        | Modules |",
            "|--------------|---------|",
        ]
        for layer, mods in arch.get("layers", {}).items():
            lines.append(
                f"| {layer.title()} | {', '.join(f'`{m}`' for m in mods) or '—'} |"
            )

        lines += [
            "",
            "---",
            "",
            f"_End of Phase 4 Master Report — {total_modules} modules processed_",
        ]

        return "\n".join(lines)

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self):
        """
        Full Phase 4 execution:
        1. Load analysis_results.json
        2. For each module → run 4 agents → save module report
        3. Build and save MASTER_REPORT.md
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: MULTI-AGENT CODE ANALYSIS PIPELINE")
        logger.info(f"Model : {MODEL_ID}")
        logger.info("=" * 60)

        analysis = self._load_analysis()

        phase2           = analysis.get("phase2", {})
        parsed_modules   = phase2.get("parsed_modules", {})
        code_files_meta  = {
            f["path"]: f
            for lang_files in phase2.get("code_files", {}).values()
            for f in lang_files
        }

        if not parsed_modules:
            logger.error(
                "No parsed_modules in analysis_results.json. "
                "Re-run phase_1_2_claude.py first."
            )
            return

        total   = len(parsed_modules)
        success = 0
        skipped = 0
        module_summaries: List[Dict] = []

        logger.info(f"Found {total} modules to analyse.\n")

        for idx, (file_path, module_data) in enumerate(parsed_modules.items(), 1):
            logger.info(f"[{idx}/{total}] {file_path}")

            meta          = code_files_meta.get(file_path, {})
            absolute_path = meta.get("absolute_path", "")

            if not absolute_path or not Path(absolute_path).exists():
                logger.warning("    Source file not found — skipping.")
                module_summaries.append({
                    "file":        file_path,
                    "language":    meta.get("language", "?"),
                    "lines":       meta.get("lines", 0),
                    "processed":   False,
                    "report_path": None,
                })
                skipped += 1
                continue

            refactoring_md, smells_md, practices_md, consolidated_md = \
                self._analyze_module(file_path, module_data, absolute_path)

            report_path = self._save_module_report(
                file_path, refactoring_md, smells_md, practices_md, consolidated_md
            )

            module_summaries.append({
                "file":        file_path,
                "language":    meta.get("language", "?"),
                "lines":       meta.get("lines", 0),
                "processed":   True,
                "report_path": str(report_path),
            })
            success += 1

        # ── Master Report ────────────────────────────────────────────────────
        master_md   = self._build_master_report(analysis, module_summaries)
        master_path = self.output_dir / "MASTER_REPORT.md"
        self._save(master_path, master_md)

        logger.info("=" * 60)
        logger.info("PHASE 4 COMPLETE")
        logger.info(f"  ✅ Analysed : {success} modules")
        logger.info(f"  ⏭  Skipped  : {skipped} modules")
        logger.info(f"  📁 Reports  : {self.output_dir}")
        logger.info(f"  📋 Master   : {master_path}")
        logger.info("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

def _load_env(env_path: str = ".env"):
    """Simple .env loader — no extra dependencies needed."""
    env_file = Path(__file__).parent / env_path
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


if __name__ == "__main__":
    _load_env()

    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not HF_TOKEN:
        print(
            "\n❌  Hugging Face token not found!\n"
            "    Set  HF_TOKEN=hf_xxxx  in your .env file or environment.\n"
            "    Get your token at: https://huggingface.co/settings/tokens\n"
        )
        raise SystemExit(1)

    orchestrator = Phase4MultiAgentOrchestrator(hf_token=HF_TOKEN)
    orchestrator.run()
