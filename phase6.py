"""
Code-to-Doc: Phase 6 – Documentation Validation & QA
======================================================
Validates and scores the generated docstrings from Phase 4.

Prompt 6.1 – Docstring Quality Review:
    Evaluate each docstring on:
        1. Completeness
        2. Clarity
        3. Parameter coverage
        4. Return description accuracy
        5. Formatting correctness
    Score out of 10.
    Suggest improvements if score < 8.

Prompt 6.2 – Consistency Validator:
    Compare function signature with docstring to verify:
        - All parameters are documented
        - Return type is documented
        - No undocumented exceptions
        - No placeholder text

Additional Validation:
    - ast.parse to verify Python docstrings are syntactically valid
    - Check docstring structure (Args, Returns, Raises sections)

Inputs  : Phase 4 module reports + analysis_results.json
Output  : output/phase6_qa/QA_REPORT.md

Technology: LangChain, ast.parse, regex, Groq, rich
"""

import os
import re
import ast
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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
BASE_DIR       = Path(__file__).parent
ANALYSIS_FILE  = BASE_DIR / "output" / "analysis_results.json"
PHASE4_DIR     = BASE_DIR / "output" / "phase4_reports" / "modules"
OUTPUT_DIR     = BASE_DIR / "output" / "phase6_qa"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DocstringEntry:
    """Represents a single extracted docstring from Phase 4 reports."""
    module: str
    function_name: str
    line_no: int
    complexity: int
    docstring: str
    signature: str = ""  # Function signature from source code
    # Validation results (filled in later)
    syntax_valid: bool = True
    syntax_error: str = ""
    has_args: bool = False
    has_returns: bool = False
    has_raises: bool = False
    has_example: bool = False


@dataclass
class ConsistencyResult:
    """Result of Prompt 6.2 consistency validation."""
    function_name: str
    module: str
    is_consistent: bool = True
    all_params_documented: bool = True
    return_documented: bool = True
    no_undocumented_exceptions: bool = True
    no_placeholder_text: bool = True
    missing_params: List[str] = field(default_factory=list)
    undocumented_exceptions: List[str] = field(default_factory=list)
    placeholders_found: List[str] = field(default_factory=list)
    suggestions: str = ""


@dataclass
class QAResult:
    """Result of LLM quality review for a docstring."""
    function_name: str
    module: str
    score: int = 0
    completeness: str = ""
    clarity: str = ""
    param_coverage: str = ""
    return_accuracy: str = ""
    formatting: str = ""
    suggestions: str = ""
    raw_response: str = ""


# ============================================================================
# SYSTEM PROMPT – DOCSTRING QUALITY REVIEWER
# ============================================================================

SYSTEM_PROMPT = """\
You are a documentation reviewer specializing in Python docstrings.

Evaluate each docstring based on these 5 criteria:
1. **Completeness** – Does it cover purpose, all parameters, return value, and exceptions?
2. **Clarity** – Is it easy to understand? Is the language precise and unambiguous?
3. **Parameter coverage** – Are all parameters listed with types and descriptions?
4. **Return description accuracy** – Is the return value clearly described with type?
5. **Formatting correctness** – Does it follow Google-style docstring conventions?

Respond in this EXACT format (no extra text):
```
SCORE: <number 1-10>
COMPLETENESS: <brief assessment>
CLARITY: <brief assessment>
PARAM_COVERAGE: <brief assessment>
RETURN_ACCURACY: <brief assessment>
FORMATTING: <brief assessment>
SUGGESTIONS: <if score < 8, list specific improvements; else "None">
```
"""

USER_PROMPT_TEMPLATE = """\
Review this Python docstring for the function `{function_name}`:

```python
{docstring}
```

Evaluate it on the 5 criteria and provide a score out of 10.
"""

# ============================================================================
# SYSTEM PROMPT 6.2 – CONSISTENCY VALIDATOR
# ============================================================================

CONSISTENCY_SYSTEM_PROMPT = """\
You are a documentation consistency checker.

Compare the function signature with its docstring and verify:
1. **All parameters documented** – Every parameter in the signature must appear in Args section
2. **Return type documented** – If function returns something, Returns section must exist
3. **No undocumented exceptions** – Any raised exceptions should be in Raises section
4. **No placeholder text** – No TODO, FIXME, TBD, "insert here", or similar placeholders

Respond in this EXACT format (no extra text):
```
CONSISTENT: <YES or NO>
ALL_PARAMS_DOCUMENTED: <YES or NO>
MISSING_PARAMS: <comma-separated list or "None">
RETURN_DOCUMENTED: <YES or NO>
UNDOCUMENTED_EXCEPTIONS: <comma-separated list or "None">
PLACEHOLDERS_FOUND: <comma-separated list or "None">
SUGGESTIONS: <specific fixes needed or "None">
```
"""

CONSISTENCY_USER_PROMPT = """\
Compare this function signature with its docstring:

**Function Signature:**
```python
{signature}
```

**Docstring:**
```python
{docstring}
```

Verify consistency between the signature and documentation.
"""


# ============================================================================
# LANGCHAIN LLM WRAPPER
# ============================================================================

class GroqLLMClient:
    """LangChain ChatGroq client for fast LLM inference."""

    def __init__(self, api_key: str, model_id: str = MODEL_ID):
        if not api_key:
            raise ValueError(
                "Groq API key is missing.\n"
                "Set GROQ_API_KEY in your .env file or environment."
            )
        self.model_id = model_id
        self.client = ChatGroq(
            model=model_id,
            api_key=api_key,
            temperature=0.3,
            max_tokens=1024,
            model_kwargs={"top_p": 0.9},
        )
        logger.info("LangChain ChatGroq QA reviewer ready → %s", model_id)

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
# DOCSTRING EXTRACTOR – Parse Phase 4 reports
# ============================================================================

class DocstringExtractor:
    """
    Extracts generated docstrings from Phase 4 module reports.
    """

    # Regex to match docstring blocks in Phase 4 markdown
    # Pattern: **Generated Docstring:** followed by ```python ... ```
    DOCSTRING_PATTERN = re.compile(
        r"####\s+`([^`]+)`.*?"                          # function name
        r"\*\*Line:\*\*\s*(\d+)\s*\|\s*"                # line number
        r"\*\*Complexity\s*\(CC\):\*\*\s*(\d+).*?"      # complexity
        r"\*\*Generated Docstring:\*\*\s*"              # marker
        r"```python\s*\n(.*?)```",                      # docstring content
        re.DOTALL | re.IGNORECASE
    )

    def __init__(self, phase4_dir: Path):
        self.phase4_dir = phase4_dir

    def extract_all(self) -> List[DocstringEntry]:
        """Extract all docstrings from all Phase 4 module reports."""
        entries: List[DocstringEntry] = []

        if not self.phase4_dir.exists():
            logger.warning("Phase 4 reports directory not found: %s", self.phase4_dir)
            return entries

        for md_file in self.phase4_dir.glob("*.md"):
            module_entries = self._extract_from_file(md_file)
            entries.extend(module_entries)
            logger.info("Extracted %d docstrings from %s", len(module_entries), md_file.name)

        return entries

    def _extract_from_file(self, md_file: Path) -> List[DocstringEntry]:
        """Extract docstrings from a single Phase 4 module report."""
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        entries: List[DocstringEntry] = []

        # Get module name from the file header
        module_match = re.search(r"## Module:\s*`([^`]+)`", content)
        module_name = module_match.group(1) if module_match else md_file.stem

        # Find all docstring entries
        for match in self.DOCSTRING_PATTERN.finditer(content):
            func_name = match.group(1).strip()
            line_no = int(match.group(2))
            complexity = int(match.group(3))
            docstring = match.group(4).strip()

            # Skip if docstring is empty or contains error message
            if not docstring or "Error" in docstring or "failed" in docstring.lower():
                continue

            entry = DocstringEntry(
                module=module_name,
                function_name=func_name,
                line_no=line_no,
                complexity=complexity,
                docstring=docstring,
            )

            # Static validation
            self._validate_syntax(entry)
            self._check_sections(entry)

            entries.append(entry)

        return entries

    def _validate_syntax(self, entry: DocstringEntry) -> None:
        """Validate docstring syntax using ast.parse."""
        # The extracted docstring may already include triple quotes - strip them
        doc = entry.docstring.strip()
        if doc.startswith('"""') and doc.endswith('"""'):
            doc = doc[3:-3]
        elif doc.startswith("'''") and doc.endswith("'''"):
            doc = doc[3:-3]
        
        # Wrap docstring in a minimal function to check syntax
        test_code = f'def _test_():\n    """{doc}"""\n    pass'
        try:
            ast.parse(test_code)
            entry.syntax_valid = True
            entry.syntax_error = ""
        except SyntaxError as e:
            entry.syntax_valid = False
            entry.syntax_error = str(e)

    def _check_sections(self, entry: DocstringEntry) -> None:
        """Check for presence of standard docstring sections."""
        doc_lower = entry.docstring.lower()
        entry.has_args = "args:" in doc_lower or "parameters:" in doc_lower
        entry.has_returns = "returns:" in doc_lower or "return:" in doc_lower
        entry.has_raises = "raises:" in doc_lower or "raise:" in doc_lower
        entry.has_example = "example:" in doc_lower or "examples:" in doc_lower or ">>>" in entry.docstring


# ============================================================================
# QA REVIEWER – LLM-based quality scoring
# ============================================================================

class QAReviewer:
    """
    Uses LLM to evaluate docstring quality based on Prompt 6.1 criteria.
    """

    def __init__(self, llm: GroqLLMClient):
        self.llm = llm

    def review(self, entry: DocstringEntry) -> QAResult:
        """Review a single docstring and return QA result."""
        result = QAResult(
            function_name=entry.function_name,
            module=entry.module,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            function_name=entry.function_name,
            docstring=entry.docstring,
        )

        try:
            response = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            result.raw_response = response
            self._parse_response(response, result)
        except Exception as e:
            logger.error("LLM review failed for %s: %s", entry.function_name, e)
            result.raw_response = f"Error: {e}"
            result.score = 0

        return result

    def _parse_response(self, response: str, result: QAResult) -> None:
        """Parse the structured LLM response into QAResult fields."""
        # Extract score
        score_match = re.search(r"SCORE:\s*(\d+)", response, re.IGNORECASE)
        if score_match:
            result.score = min(10, max(1, int(score_match.group(1))))

        # Extract other fields
        patterns = {
            "completeness": r"COMPLETENESS:\s*(.+?)(?=\n[A-Z_]+:|$)",
            "clarity": r"CLARITY:\s*(.+?)(?=\n[A-Z_]+:|$)",
            "param_coverage": r"PARAM_COVERAGE:\s*(.+?)(?=\n[A-Z_]+:|$)",
            "return_accuracy": r"RETURN_ACCURACY:\s*(.+?)(?=\n[A-Z_]+:|$)",
            "formatting": r"FORMATTING:\s*(.+?)(?=\n[A-Z_]+:|$)",
            "suggestions": r"SUGGESTIONS:\s*(.+?)(?=\n[A-Z_]+:|$)",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                setattr(result, field, match.group(1).strip())


# ============================================================================
# SIGNATURE EXTRACTOR – Get function signatures from source code
# ============================================================================

class SignatureExtractor:
    """
    Extracts function signatures from the original source code using AST.
    """

    def __init__(self, analysis_data: Dict):
        self.analysis_data = analysis_data
        self.repo_path = Path(analysis_data.get("repo_path", ""))

    def get_signature(self, module_rel_path: str, func_name: str, line_no: int) -> str:
        """Get the function signature for a specific function."""
        # Normalize path separators
        module_rel_path = module_rel_path.replace("\\", "/")
        
        # Try to find the source file
        source_file = self.repo_path / module_rel_path.replace("/", os.sep)
        if not source_file.exists():
            # Try cloned_repo subdirectory
            for subdir in ["cloned_repo", "s-tool", "s_tool"]:
                alt_path = self.repo_path / subdir / module_rel_path.replace("/", os.sep)
                if alt_path.exists():
                    source_file = alt_path
                    break
        
        if not source_file.exists():
            return f"def {func_name}(...):"

        try:
            source = source_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Match by name and approximate line number
                    if node.name == func_name.split("(")[0] or abs(node.lineno - line_no) < 3:
                        return self._format_signature(node)
                        
        except Exception as e:
            logger.debug("Could not extract signature for %s: %s", func_name, e)
        
        return f"def {func_name}(...):"

    def _format_signature(self, node: ast.FunctionDef) -> str:
        """Format an AST function definition as a signature string."""
        # Get function name
        name = node.name
        
        # Get arguments
        args = node.args
        arg_parts = []
        
        # Regular positional args
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        
        for i, arg in enumerate(args.args):
            arg_str = arg.arg
            # Add type annotation if present
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            # Add default value if present
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0:
                try:
                    arg_str += f" = {ast.unparse(args.defaults[default_idx])}"
                except:
                    arg_str += " = ..."
            arg_parts.append(arg_str)
        
        # *args
        if args.vararg:
            vararg = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                try:
                    vararg += f": {ast.unparse(args.vararg.annotation)}"
                except:
                    pass
            arg_parts.append(vararg)
        
        # **kwargs
        if args.kwarg:
            kwarg = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                try:
                    kwarg += f": {ast.unparse(args.kwarg.annotation)}"
                except:
                    pass
            arg_parts.append(kwarg)
        
        # Return type
        returns = ""
        if node.returns:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except:
                pass
        
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {name}({', '.join(arg_parts)}){returns}:"


# ============================================================================
# CONSISTENCY VALIDATOR – Prompt 6.2
# ============================================================================

class ConsistencyValidator:
    """
    Uses LLM to validate consistency between function signature and docstring.
    Implements Prompt 6.2.
    """

    # Placeholder patterns to check
    PLACEHOLDER_PATTERNS = [
        r'\bTODO\b', r'\bFIXME\b', r'\bTBD\b', r'\bXXX\b',
        r'\binsert\s+here\b', r'\bfill\s+in\b', r'\badd\s+description\b',
        r'\bNOT\s+IMPLEMENTED\b', r'\bplaceholder\b',
    ]

    def __init__(self, llm: GroqLLMClient):
        self.llm = llm
        self.placeholder_re = re.compile(
            '|'.join(self.PLACEHOLDER_PATTERNS),
            re.IGNORECASE
        )

    def validate(self, entry: DocstringEntry) -> ConsistencyResult:
        """Validate consistency between signature and docstring."""
        result = ConsistencyResult(
            function_name=entry.function_name,
            module=entry.module,
        )

        # Static checks first
        self._check_placeholders(entry, result)
        self._check_params_basic(entry, result)

        # If no signature, skip LLM validation
        if not entry.signature or entry.signature.endswith("(...):"):
            return result

        # LLM validation
        user_prompt = CONSISTENCY_USER_PROMPT.format(
            signature=entry.signature,
            docstring=entry.docstring,
        )

        try:
            response = self.llm.generate(CONSISTENCY_SYSTEM_PROMPT, user_prompt)
            self._parse_response(response, result)
        except Exception as e:
            logger.error("Consistency check failed for %s: %s", entry.function_name, e)

        # Determine overall consistency
        result.is_consistent = (
            result.all_params_documented and
            result.return_documented and
            result.no_undocumented_exceptions and
            result.no_placeholder_text
        )

        return result

    def _check_placeholders(self, entry: DocstringEntry, result: ConsistencyResult) -> None:
        """Check for placeholder text in docstring."""
        matches = self.placeholder_re.findall(entry.docstring)
        if matches:
            result.no_placeholder_text = False
            result.placeholders_found = list(set(matches))

    def _check_params_basic(self, entry: DocstringEntry, result: ConsistencyResult) -> None:
        """Basic parameter check - extract params from signature and check docstring."""
        if not entry.signature:
            return

        # Extract parameter names from signature
        sig_match = re.search(r'\(([^)]*)\)', entry.signature)
        if not sig_match:
            return

        params_str = sig_match.group(1)
        # Parse parameters (simplified)
        params = []
        for part in params_str.split(','):
            part = part.strip()
            if not part or part == 'self' or part == 'cls':
                continue
            # Extract parameter name (before : or =)
            param_name = re.split(r'[:\s=]', part)[0].strip()
            if param_name and not param_name.startswith('*'):
                params.append(param_name)
            elif param_name.startswith('**'):
                params.append(param_name[2:])
            elif param_name.startswith('*'):
                params.append(param_name[1:])

        # Check if each param is mentioned in docstring Args section
        doc_lower = entry.docstring.lower()
        missing = []
        for param in params:
            if param.lower() not in doc_lower:
                missing.append(param)

        if missing:
            result.all_params_documented = False
            result.missing_params = missing

    def _parse_response(self, response: str, result: ConsistencyResult) -> None:
        """Parse LLM consistency check response."""
        # All params documented
        match = re.search(r"ALL_PARAMS_DOCUMENTED:\s*(YES|NO)", response, re.IGNORECASE)
        if match:
            result.all_params_documented = match.group(1).upper() == "YES"

        # Missing params
        match = re.search(r"MISSING_PARAMS:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.IGNORECASE)
        if match and match.group(1).strip().lower() != "none":
            result.missing_params = [p.strip() for p in match.group(1).split(",") if p.strip()]

        # Return documented
        match = re.search(r"RETURN_DOCUMENTED:\s*(YES|NO)", response, re.IGNORECASE)
        if match:
            result.return_documented = match.group(1).upper() == "YES"

        # Undocumented exceptions
        match = re.search(r"UNDOCUMENTED_EXCEPTIONS:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.IGNORECASE)
        if match and match.group(1).strip().lower() != "none":
            result.no_undocumented_exceptions = False
            result.undocumented_exceptions = [e.strip() for e in match.group(1).split(",") if e.strip()]

        # Placeholders
        match = re.search(r"PLACEHOLDERS_FOUND:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.IGNORECASE)
        if match and match.group(1).strip().lower() != "none":
            result.no_placeholder_text = False
            result.placeholders_found.extend([p.strip() for p in match.group(1).split(",") if p.strip()])

        # Suggestions
        match = re.search(r"SUGGESTIONS:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).strip().lower() != "none":
            result.suggestions = match.group(1).strip()


# ============================================================================
# PHASE 6 ORCHESTRATOR
# ============================================================================

class Phase6:
    """
    Phase 6 – Documentation Validation & QA

    1. Extract all docstrings from Phase 4 reports
    2. Validate syntax using ast.parse
    3. LLM-evaluate each docstring on 5 quality criteria
    4. Generate comprehensive QA report
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_analysis(self) -> Dict:
        """Load analysis_results.json."""
        if not ANALYSIS_FILE.exists():
            raise FileNotFoundError(f"analysis_results.json not found at {ANALYSIS_FILE}")
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> None:
        """Execute Phase 6."""
        console.print(Panel(
            "[bold cyan]Code-to-Doc  ·  Phase 6[/bold cyan]\n"
            "[white]Documentation Validation & QA[/white]\n"
            f"[dim]Model: {MODEL_ID}  ·  Engine: Groq[/dim]\n"
            "[dim]Prompt 6.1: Quality Review  |  Prompt 6.2: Consistency Check[/dim]",
            expand=False,
        ))

        # 1. Load analysis data
        data = self._load_analysis()
        project_name = data.get("metadata", {}).get("project_name", "Unknown")
        console.print(f"\n📂 Project: [bold]{project_name}[/bold]")

        # 2. Extract docstrings from Phase 4 reports
        extractor = DocstringExtractor(PHASE4_DIR)
        entries = extractor.extract_all()
        console.print(f"📝 Extracted [bold]{len(entries)}[/bold] docstrings from Phase 4 reports")

        if not entries:
            console.print("[yellow]⚠ No docstrings found to review.[/yellow]")
            return

        # 2b. Extract function signatures from source code
        sig_extractor = SignatureExtractor(data)
        for entry in entries:
            entry.signature = sig_extractor.get_signature(
                entry.module, entry.function_name.split("(")[0], entry.line_no
            )
        console.print(f"🔍 Extracted function signatures from source code")

        # 3. Static validation summary
        syntax_valid = sum(1 for e in entries if e.syntax_valid)
        has_args = sum(1 for e in entries if e.has_args)
        has_returns = sum(1 for e in entries if e.has_returns)
        has_raises = sum(1 for e in entries if e.has_raises)
        has_example = sum(1 for e in entries if e.has_example)

        console.print(f"\n[bold]Static Validation:[/bold]")
        console.print(f"  Syntax valid (ast.parse): {syntax_valid}/{len(entries)}")
        console.print(f"  Has Args section: {has_args}/{len(entries)}")
        console.print(f"  Has Returns section: {has_returns}/{len(entries)}")
        console.print(f"  Has Raises section: {has_raises}/{len(entries)}")
        console.print(f"  Has Example: {has_example}/{len(entries)}")

        # 4. Initialize LLM and reviewers
        llm = GroqLLMClient(self.api_key)
        reviewer = QAReviewer(llm)
        consistency_validator = ConsistencyValidator(llm)

        qa_results: List[QAResult] = []
        consistency_results: List[ConsistencyResult] = []
        failed_reviews = 0

        console.print(f"\n🤖 Running Prompt 6.1 (Quality Review) + 6.2 (Consistency Check)…\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Validating…", total=len(entries))

            for i, entry in enumerate(entries):
                progress.update(task, description=f"[{i+1}/{len(entries)}] {entry.function_name[:30]}")

                # Prompt 6.1: Quality Review
                result = reviewer.review(entry)
                qa_results.append(result)

                if result.score == 0:
                    failed_reviews += 1

                # Prompt 6.2: Consistency Check
                cons_result = consistency_validator.validate(entry)
                consistency_results.append(cons_result)

                progress.advance(task)

                # Rate limiting
                time.sleep(0.3)

        # 5. Calculate statistics
        valid_results = [r for r in qa_results if r.score > 0]
        if valid_results:
            avg_score = sum(r.score for r in valid_results) / len(valid_results)
            min_score = min(r.score for r in valid_results)
            max_score = max(r.score for r in valid_results)
            below_8 = sum(1 for r in valid_results if r.score < 8)
        else:
            avg_score = min_score = max_score = 0
            below_8 = 0

        # Consistency stats
        consistent_count = sum(1 for r in consistency_results if r.is_consistent)
        missing_params_count = sum(1 for r in consistency_results if not r.all_params_documented)
        missing_returns_count = sum(1 for r in consistency_results if not r.return_documented)
        placeholder_count = sum(1 for r in consistency_results if not r.no_placeholder_text)

        # 6. Generate QA report
        report_path = self._generate_report(
            entries, qa_results, consistency_results, project_name,
            avg_score, min_score, max_score, below_8, failed_reviews,
            syntax_valid, has_args, has_returns, has_raises, has_example,
            consistent_count, missing_params_count, missing_returns_count, placeholder_count,
        )

        # 7. Update analysis_results.json
        data["phase"] = 6
        data["phase6"] = {
            "model": MODEL_ID,
            "engine": "LangChain",
            "prompts_used": ["6.1_quality_review", "6.2_consistency_check"],
            "docstrings_reviewed": len(entries),
            "reviews_completed": len(valid_results),
            "reviews_failed": failed_reviews,
            "average_score": round(avg_score, 2),
            "min_score": min_score,
            "max_score": max_score,
            "below_threshold": below_8,
            "syntax_valid_count": syntax_valid,
            "consistent_count": consistent_count,
            "missing_params_count": missing_params_count,
            "missing_returns_count": missing_returns_count,
            "placeholder_count": placeholder_count,
            "report_path": str(report_path),
            "completed_at": datetime.now().isoformat(),
        }
        with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        # 8. Summary table
        table = Table(title="Phase 6 — Documentation QA")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Model", MODEL_ID)
        table.add_row("Docstrings reviewed", str(len(entries)))
        table.add_row("LLM reviews completed", str(len(valid_results)))
        table.add_row("LLM reviews failed", str(failed_reviews))
        table.add_row("Average score (6.1)", f"{avg_score:.1f}/10")
        table.add_row("Min / Max score", f"{min_score} / {max_score}")
        table.add_row("Below 8 (need improvement)", str(below_8))
        table.add_row("Consistent (6.2)", f"{consistent_count}/{len(entries)}")
        table.add_row("Missing params", str(missing_params_count))
        table.add_row("Syntax valid (ast.parse)", f"{syntax_valid}/{len(entries)}")
        table.add_row("Report", str(report_path))
        console.print(table)

        console.print(Panel(
            f"[green]✓ Phase 6 complete[/green]\n"
            f"QA Report → {report_path}",
            expand=False,
        ))

    def _generate_report(
        self,
        entries: List[DocstringEntry],
        results: List[QAResult],
        consistency_results: List[ConsistencyResult],
        project_name: str,
        avg_score: float,
        min_score: int,
        max_score: int,
        below_8: int,
        failed_reviews: int,
        syntax_valid: int,
        has_args: int,
        has_returns: int,
        has_raises: int,
        has_example: int,
        consistent_count: int,
        missing_params_count: int,
        missing_returns_count: int,
        placeholder_count: int,
    ) -> Path:
        """Generate comprehensive QA report in Markdown."""
        report_path = self.output_dir / "QA_REPORT.md"

        lines = [
            f"# Phase 6: Documentation Quality Assurance Report",
            f"## Project: `{project_name}`",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Docstrings Reviewed | {len(entries)} |",
            f"| LLM Reviews Completed | {len(results) - failed_reviews} |",
            f"| LLM Reviews Failed | {failed_reviews} |",
            f"| **Average Score (6.1)** | **{avg_score:.1f}/10** |",
            f"| Min Score | {min_score} |",
            f"| Max Score | {max_score} |",
            f"| Below 8 (Need Improvement) | {below_8} |",
            "",
            "---",
            "",
            "## Prompt 6.2: Consistency Validation",
            "",
            f"| Check | Pass | Total | Percentage |",
            f"|-------|------|-------|------------|",
            f"| Fully Consistent | {consistent_count} | {len(entries)} | {100*consistent_count/len(entries):.0f}% |",
            f"| All Params Documented | {len(entries) - missing_params_count} | {len(entries)} | {100*(len(entries)-missing_params_count)/len(entries):.0f}% |",
            f"| Return Documented | {len(entries) - missing_returns_count} | {len(entries)} | {100*(len(entries)-missing_returns_count)/len(entries):.0f}% |",
            f"| No Placeholders | {len(entries) - placeholder_count} | {len(entries)} | {100*(len(entries)-placeholder_count)/len(entries):.0f}% |",
            "",
            "---",
            "",
            "## Static Validation (ast.parse)",
            "",
            f"| Check | Pass | Total | Percentage |",
            f"|-------|------|-------|------------|",
            f"| Syntax Valid | {syntax_valid} | {len(entries)} | {100*syntax_valid/len(entries):.0f}% |",
            f"| Has Args Section | {has_args} | {len(entries)} | {100*has_args/len(entries):.0f}% |",
            f"| Has Returns Section | {has_returns} | {len(entries)} | {100*has_returns/len(entries):.0f}% |",
            f"| Has Raises Section | {has_raises} | {len(entries)} | {100*has_raises/len(entries):.0f}% |",
            f"| Has Example | {has_example} | {len(entries)} | {100*has_example/len(entries):.0f}% |",
            "",
            "---",
            "",
            "## Score Distribution (Prompt 6.1)",
            "",
        ]

        # Score distribution
        score_dist = {i: 0 for i in range(1, 11)}
        for r in results:
            if r.score > 0:
                score_dist[r.score] += 1

        lines.append("| Score | Count |")
        lines.append("|-------|-------|")
        for score in range(10, 0, -1):
            count = score_dist[score]
            bar = "█" * count
            lines.append(f"| {score} | {count} {bar} |")
        lines.append("")

        # Consistency issues
        inconsistent = [(e, c) for e, c in zip(entries, consistency_results) if not c.is_consistent]
        if inconsistent:
            lines.extend([
                "---",
                "",
                "## Consistency Issues (Prompt 6.2)",
                "",
            ])
            for entry, cons in inconsistent:
                issues = []
                if not cons.all_params_documented:
                    issues.append(f"Missing params: {', '.join(cons.missing_params)}")
                if not cons.return_documented:
                    issues.append("Return not documented")
                if not cons.no_undocumented_exceptions:
                    issues.append(f"Undocumented exceptions: {', '.join(cons.undocumented_exceptions)}")
                if not cons.no_placeholder_text:
                    issues.append(f"Placeholders: {', '.join(cons.placeholders_found)}")
                
                lines.extend([
                    f"### `{entry.function_name}` ({entry.module})",
                    f"- **Signature:** `{entry.signature}`",
                    f"- **Issues:** {'; '.join(issues)}",
                ])
                if cons.suggestions:
                    lines.append(f"- **Suggestions:** {cons.suggestions}")
                lines.extend(["", "---", ""])

        # Docstrings needing improvement (score < 8)
        needs_improvement = [(e, r) for e, r in zip(entries, results) if 0 < r.score < 8]
        if needs_improvement:
            lines.extend([
                "",
                "## Docstrings Needing Improvement (Score < 8)",
                "",
            ])
            for entry, result in sorted(needs_improvement, key=lambda x: x[1].score):
                lines.extend([
                    f"### `{entry.function_name}` ({entry.module})",
                    f"- **Score:** {result.score}/10",
                    f"- **Line:** {entry.line_no}  |  **Complexity:** {entry.complexity}",
                    "",
                    "**Evaluation:**",
                    f"- Completeness: {result.completeness}",
                    f"- Clarity: {result.clarity}",
                    f"- Param Coverage: {result.param_coverage}",
                    f"- Return Accuracy: {result.return_accuracy}",
                    f"- Formatting: {result.formatting}",
                    "",
                    "**Suggestions:**",
                    f"> {result.suggestions}",
                    "",
                    "---",
                    "",
                ])

        # Top scoring docstrings
        top_scorers = [(e, r) for e, r in zip(entries, results) if r.score >= 9]
        if top_scorers:
            lines.extend([
                "## Top-Scoring Docstrings (Score ≥ 9)",
                "",
            ])
            for entry, result in sorted(top_scorers, key=lambda x: -x[1].score)[:10]:
                lines.append(f"- **{entry.function_name}** ({entry.module}) — Score: {result.score}/10")
            lines.append("")

        # All results table
        lines.extend([
            "---",
            "",
            "## All Reviewed Docstrings",
            "",
            "| Module | Function | Line | CC | Score | Args | Returns | Raises | Example |",
            "|--------|----------|------|-----|-------|------|---------|--------|---------|",
        ])
        for entry, result in zip(entries, results):
            check = "✓"
            cross = "✗"
            lines.append(
                f"| {entry.module} | `{entry.function_name}` | {entry.line_no} | {entry.complexity} | "
                f"{result.score}/10 | {check if entry.has_args else cross} | "
                f"{check if entry.has_returns else cross} | {check if entry.has_raises else cross} | "
                f"{check if entry.has_example else cross} |"
            )

        lines.extend([
            "",
            "---",
            "",
            f"_Report generated by Code-to-Doc Phase 6 using {MODEL_ID}_",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("QA report saved → %s", report_path)
        return report_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        console.print("[red]ERROR: GROQ_API_KEY not set. Add it to .env or environment.[/red]")
        raise SystemExit(1)

    phase6 = Phase6(api_key)
    phase6.run()


if __name__ == "__main__":
    main()
