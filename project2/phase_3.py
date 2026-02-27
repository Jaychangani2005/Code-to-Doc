"""
Code-to-Doc: Phase 3 - AI Documentation Generator
Uses Meta-Llama-3-8B-Instruct via Hugging Face Inference API
to generate Markdown documentation from analysis_results.json
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_ID        = "meta-llama/Meta-Llama-3-8B-Instruct"
ANALYSIS_FILE   = Path(__file__).parent / "output" / "analysis_results.json"
OUTPUT_DIR      = Path(__file__).parent / "output" / "generated_docs"
MAX_CODE_CHARS  = 6000    # Limit to avoid exceeding context window


# ============================================================================
# PHASE 3: LLM CLIENT
# ============================================================================

class LlamaClient:
    """Wraps Hugging Face InferenceClient for Llama-3-8B-Instruct."""

    def __init__(self, hf_token: str):
        if not hf_token:
            raise ValueError(
                "Hugging Face API token is missing.\n"
                "Set HF_TOKEN in your environment or .env file."
            )
        self.client = InferenceClient(
            model=MODEL_ID,
            token=hf_token,
        )
        logger.info(f"LlamaClient ready → {MODEL_ID}")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024) -> str:
        """
        Call Llama-3-8B-Instruct with a system + user message.

        Args:
            system_prompt: Instructions for the model role.
            user_prompt:   The actual content/task.
            max_new_tokens: How many tokens to generate.

        Returns:
            Generated text string.
        """
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ]

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ============================================================================
# PHASE 3: PROMPT BUILDER
# ============================================================================

class PromptBuilder:
    """Builds structured prompts from parsed module data."""

    SYSTEM_PROMPT = (
        "You are a Senior Software Engineer Documentation Expert. "
        "Your job is to write clear, accurate, and well-structured technical "
        "documentation in Markdown format. Be concise but thorough. "
        "Output ONLY the Markdown — no extra commentary."
    )

    def build_module_prompt(
        self,
        file_path: str,
        module_data: Dict,
        dependencies: Dict,
        source_code: str,
    ) -> str:
        """
        Build a documentation prompt for a single module.

        Args:
            file_path:    Relative path of the file.
            module_data:  Parsed module info (functions, classes, etc.).
            dependencies: dependency_graph entry for this file.
            source_code:  Raw source code (truncated).

        Returns:
            Formatted prompt string.
        """
        # ── Functions summary ───────────────────────────────────────────────
        func_lines = []
        for fn in module_data.get("functions", []):
            sig  = fn.get("signature", fn["name"])
            doc  = fn.get("docstring") or "No docstring."
            cplx = fn.get("complexity", "?")
            priv = "private" if fn.get("is_private") else "public"
            func_lines.append(
                f"  - `{sig}` [{priv}, complexity={cplx}]\n    {doc[:120]}"
            )

        # ── Classes summary ─────────────────────────────────────────────────
        class_lines = []
        for cls in module_data.get("classes", []):
            class_lines.append(f"  - `{cls['name']}` (bases: {cls.get('bases', [])})")
            for m in cls.get("methods", []):
                sig  = m.get("signature", m["name"])
                doc  = m.get("docstring") or ""
                cplx = m.get("complexity", "?")
                class_lines.append(
                    f"      · `{sig}` [complexity={cplx}] {doc[:80]}"
                )

        # ── Dependency info ──────────────────────────────────────────────────
        internal_deps = dependencies.get("imports", [])
        external_deps = dependencies.get("external_deps", [])

        # ── Assemble prompt ──────────────────────────────────────────────────
        prompt = f"""
## Task
Write complete technical documentation for the file: `{file_path}`

## Module Docstring
{module_data.get("docstring") or "None"}

## Imports
{chr(10).join(module_data.get("imports", [])) or "None"}

## Internal Dependencies (files this module imports internally)
{", ".join(internal_deps) if internal_deps else "None"}

## External Libraries Used
{", ".join(external_deps) if external_deps else "None"}

## Functions Detected
{chr(10).join(func_lines) if func_lines else "None"}

## Classes Detected
{chr(10).join(class_lines) if class_lines else "None"}

## Source Code (truncated to {MAX_CODE_CHARS} chars)
```python
{source_code[:MAX_CODE_CHARS]}
```

## Documentation Format Required
1. `# <Module Name>` — Title
2. `## Overview` — 3-4 sentence summary of what this module does.
3. `## Dependencies` — Internal and external dependencies explained.
4. `## Classes` — Each class with description and its key methods.
5. `## Functions` — Each function with: purpose, parameters, return value.
6. `## Usage Example` — A short code snippet showing how to use this module.

Output Markdown only.
""".strip()
        return prompt


# ============================================================================
# PHASE 3: DOCUMENTATION GENERATOR
# ============================================================================

class Phase3DocumentationGenerator:
    """
    Reads analysis_results.json from Phase 1+2 and generates
    per-file Markdown documentation using Llama-3-8B-Instruct.
    """

    def __init__(self, hf_token: str,
                 analysis_file: Path = ANALYSIS_FILE,
                 output_dir: Path = OUTPUT_DIR):
        self.llama       = LlamaClient(hf_token)
        self.prompt_bld  = PromptBuilder()
        self.analysis_file = analysis_file
        self.output_dir  = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load_analysis(self) -> Dict:
        """Load analysis_results.json produced by Phase 1+2."""
        if not self.analysis_file.exists():
            raise FileNotFoundError(
                f"analysis_results.json not found at {self.analysis_file}\n"
                "Run phase_1_2_claude.py first."
            )
        with open(self.analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded analysis from {self.analysis_file}")
        return data

    # ── Source code reader ───────────────────────────────────────────────────

    def _read_source(self, absolute_path: str) -> str:
        """Read raw source code for a file."""
        try:
            with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read source for {absolute_path}: {e}")
            return ""

    # ── Single file ──────────────────────────────────────────────────────────

    def document_file(
        self,
        file_path: str,
        module_data: Dict,
        dependency_graph: Dict,
        absolute_path: str,
    ) -> str:
        """
        Generate documentation for a single file.

        Args:
            file_path:        Relative path (key in parsed_modules).
            module_data:      Parsed info (functions, classes, …).
            dependency_graph: Full dependency graph from Phase 2.
            absolute_path:    Full disk path to read source code.

        Returns:
            Markdown documentation string.
        """
        logger.info(f"  Documenting → {file_path}")

        source_code  = self._read_source(absolute_path)
        dependencies = dependency_graph.get(file_path, {})

        user_prompt = self.prompt_bld.build_module_prompt(
            file_path, module_data, dependencies, source_code
        )

        try:
            doc = self.llama.generate(
                system_prompt = PromptBuilder.SYSTEM_PROMPT,
                user_prompt   = user_prompt,
                max_new_tokens = 1200,
            )
            return doc
        except Exception as e:
            logger.error(f"    LLM call failed for {file_path}: {e}")
            return f"# {file_path}\n\n> ⚠️ Documentation generation failed: {e}\n"

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save_doc(self, file_path: str, content: str):
        """Save generated Markdown to output directory."""
        safe_name = file_path.replace("/", "_").replace("\\", "_") + ".md"
        out_path  = self.output_dir / safe_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"    Saved → {out_path}")
        return out_path

    # ── Full run ─────────────────────────────────────────────────────────────

    def run(self):
        """
        Full Phase 3 execution:
        1. Load analysis_results.json
        2. For each parsed module → call Llama → save .md
        3. Generate an index summary
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: DOCUMENTATION GENERATION")
        logger.info(f"Model : {MODEL_ID}")
        logger.info("=" * 60)

        analysis = self.load_analysis()

        phase2          = analysis.get("phase2", {})
        parsed_modules  = phase2.get("parsed_modules", {})
        dependency_graph = phase2.get("dependency_graph", {})
        code_files_meta = {
            f["path"]: f
            for lang_files in phase2.get("code_files", {}).values()
            for f in lang_files
        }

        if not parsed_modules:
            logger.error("No parsed_modules found in analysis_results.json. "
                         "Re-run phase_1_2_claude.py.")
            return

        total   = len(parsed_modules)
        success = 0
        skipped = 0

        logger.info(f"Found {total} modules to document.\n")

        generated_index = []

        for idx, (file_path, module_data) in enumerate(parsed_modules.items(), 1):
            logger.info(f"[{idx}/{total}] {file_path}")

            # Get absolute path from code_files metadata
            meta = code_files_meta.get(file_path, {})
            absolute_path = meta.get("absolute_path", "")

            if not absolute_path or not Path(absolute_path).exists():
                logger.warning(f"    Source file not found, skipping.")
                skipped += 1
                continue

            doc_content = self.document_file(
                file_path, module_data, dependency_graph, absolute_path
            )

            saved_path = self._save_doc(file_path, doc_content)
            generated_index.append({
                "file":     file_path,
                "doc_path": str(saved_path),
                "lines":    meta.get("lines", 0),
                "language": meta.get("language", "unknown"),
            })
            success += 1

        # ── Index file ───────────────────────────────────────────────────────
        self._save_index(generated_index, analysis)

        logger.info("=" * 60)
        logger.info(f"PHASE 3 COMPLETE")
        logger.info(f"  ✅ Documented : {success} files")
        logger.info(f"  ⏭  Skipped   : {skipped} files")
        logger.info(f"  📁 Output dir: {self.output_dir}")
        logger.info("=" * 60)

    # ── Index ────────────────────────────────────────────────────────────────

    def _save_index(self, index: List[Dict], analysis: Dict):
        """Generate and save an INDEX.md summary of all documented files."""
        phase1   = analysis.get("phase1", {})
        metadata = phase1.get("metadata", {})
        metrics  = analysis.get("phase2", {}).get("complexity_metrics", {})
        arch     = analysis.get("phase2", {}).get("architecture", {})

        lines = [
            "# Code-to-Doc: Documentation Index",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model:** `{MODEL_ID}`",
            "",
            "---",
            "",
            "## Repository Info",
            f"| Key | Value |",
            f"|-----|-------|",
            f"| Name   | {metadata.get('name', 'N/A')} |",
            f"| URL    | {metadata.get('url', 'N/A')} |",
            f"| Branch | {metadata.get('branch', 'N/A')} |",
            f"| Commit | {metadata.get('commit', 'N/A')} |",
            f"| Last Commit | {metadata.get('last_commit', 'N/A')} |",
            "",
            "## Codebase Stats",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Files    | {metrics.get('total_files', 0)} |",
            f"| Total Lines    | {metrics.get('total_lines', 0)} |",
            f"| Avg File Size  | {metrics.get('average_file_size', 0):.0f} bytes |",
            f"| Circular Deps  | {len(metrics.get('circular_dependencies', []))} |",
            "",
            "## Architecture",
            f"- **Entry Points:** {', '.join(arch.get('entry_points', [])) or 'N/A'}",
            f"- **Core Modules:** {', '.join(arch.get('core_modules', [])[:5]) or 'N/A'}",
            "",
            "## Documented Files",
            "",
            "| # | File | Language | Lines | Documentation |",
            "|---|------|----------|-------|---------------|",
        ]

        for i, entry in enumerate(index, 1):
            doc_name = Path(entry["doc_path"]).name
            lines.append(
                f"| {i} | `{entry['file']}` | {entry['language']} "
                f"| {entry['lines']} | [{doc_name}]({doc_name}) |"
            )

        index_path = self.output_dir / "INDEX.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Index saved → {index_path}")


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
    # Load .env if present
    _load_env()

    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not HF_TOKEN:
        print(
            "\n❌  Hugging Face token not found!\n"
            "    Set  HF_TOKEN=hf_xxxx  in your .env file or environment.\n"
            "    Get your token at: https://huggingface.co/settings/tokens\n"
        )
        raise SystemExit(1)

    generator = Phase3DocumentationGenerator(hf_token=HF_TOKEN)
    generator.run()