"""
Code-to-Doc: Inject Comments into Source Code
================================================
Takes the generated docstrings from Phase 4 and injects them
into the actual source code files.

Output is saved to a separate folder so you can compare:
- Original code (cloned_repo/)
- Documented code (output/documented_code/)

This allows you to verify that comments are placed on the correct lines.
"""

import os
import re
import ast
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).parent
ANALYSIS_FILE  = BASE_DIR / "output" / "analysis_results.json"
PHASE4_DIR     = BASE_DIR / "output" / "phase4_reports" / "modules"
OUTPUT_DIR     = BASE_DIR / "output" / "documented_code"


def _backup_path(path: Path) -> Path:
    """Return backup path for a source file (e.g., file.py -> file.py.bak)."""
    if path.suffix:
        return path.with_suffix(path.suffix + ".bak")
    return path.with_name(path.name + ".bak")


def ensure_backup(path: Path) -> Optional[Path]:
    """
    Create a .bak copy once before in-place modifications.
    Returns backup path if created/existing, None on failure.
    """
    try:
        bak = _backup_path(path)
        if not bak.exists() and path.exists():
            shutil.copy2(path, bak)
        return bak
    except Exception as e:
        logger.warning("Failed to create backup for %s: %s", path, e)
        return None


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FunctionDoc:
    """Stores a function's generated documentation."""
    name: str
    line_no: int
    complexity: int
    docstring: str
    logic_explanation: str = ""


@dataclass
class ModuleDocs:
    """Stores all documentation for a single module."""
    module_path: str  # relative path like "s_tool/core.py"
    functions: List[FunctionDoc]


# ============================================================================
# DOCSTRING EXTRACTOR – Parse Phase 4 reports
# ============================================================================

class Phase4Extractor:
    """
    Extracts generated docstrings from Phase 4 module reports.
    """

    # Regex to match function entries in Phase 4 markdown
    # Handles both ### (top-level functions) and #### (class methods)
    FUNCTION_PATTERN = re.compile(
        r"###[#]?\s+`([^`]+)`.*?"                       # function name with signature (### or ####)
        r"\*\*Line:\*\*\s*(\d+)\s*\|\s*"                # line number
        r"\*\*Complexity\s*\(CC\):\*\*\s*(\d+).*?"      # complexity
        r"\*\*Generated Docstring:\*\*\s*"              # marker
        r"```python\s*\n(.*?)```",                      # docstring content
        re.DOTALL | re.IGNORECASE
    )

    # Regex for complex logic explanation
    LOGIC_PATTERN = re.compile(
        r"\*\*Complex Logic Explanation:\*\*\s*\n(.*?)(?=\n---|\n####|\Z)",
        re.DOTALL
    )

    def __init__(self, phase4_dir: Path):
        self.phase4_dir = phase4_dir

    def extract_all(self) -> List[ModuleDocs]:
        """Extract all documentation from all Phase 4 reports."""
        modules: List[ModuleDocs] = []

        if not self.phase4_dir.exists():
            logger.warning("Phase 4 reports directory not found: %s", self.phase4_dir)
            return modules

        for md_file in self.phase4_dir.glob("*.md"):
            module_docs = self._extract_from_file(md_file)
            if module_docs and module_docs.functions:
                modules.append(module_docs)
                logger.info("Extracted %d functions from %s", 
                           len(module_docs.functions), md_file.name)

        return modules

    def _extract_from_file(self, md_file: Path) -> Optional[ModuleDocs]:
        """Extract documentation from a single Phase 4 report."""
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        
        # Get module path from header
        module_match = re.search(r"## Module:\s*`([^`]+)`", content)
        if not module_match:
            return None
        
        module_path = module_match.group(1).replace("\\", "/")
        functions: List[FunctionDoc] = []

        # Find all function documentation
        for match in self.FUNCTION_PATTERN.finditer(content):
            func_signature = match.group(1).strip()
            # Extract just the function name
            func_name = func_signature.split("(")[0].strip()
            
            line_no = int(match.group(2))
            complexity = int(match.group(3))
            docstring = match.group(4).strip()

            # Skip if docstring contains error message
            if "Error" in docstring or "failed" in docstring.lower():
                continue

            # Clean up docstring - remove surrounding triple quotes if present
            docstring = self._clean_docstring(docstring)

            # Look for logic explanation after this function
            func_end = match.end()
            logic_match = self.LOGIC_PATTERN.search(content, func_end)
            logic_explanation = ""
            if logic_match and logic_match.start() < func_end + 500:
                logic_explanation = logic_match.group(1).strip()

            functions.append(FunctionDoc(
                name=func_name,
                line_no=line_no,
                complexity=complexity,
                docstring=docstring,
                logic_explanation=logic_explanation,
            ))

        return ModuleDocs(module_path=module_path, functions=functions)

    def _clean_docstring(self, docstring: str) -> str:
        """Clean up docstring - remove surrounding quotes if present."""
        doc = docstring.strip()
        # Remove surrounding triple quotes
        if doc.startswith('"""') and doc.endswith('"""'):
            doc = doc[3:-3].strip()
        elif doc.startswith("'''") and doc.endswith("'''"):
            doc = doc[3:-3].strip()
        return doc


# ============================================================================
# CODE INJECTOR – Insert docstrings into source code
# ============================================================================

class CodeInjector:
    """
    Injects generated docstrings into Python source code.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def inject_module(self, module_docs: ModuleDocs) -> Tuple[bool, str, int]:
        """
        Inject docstrings into a module in-place inside cloned repo.
        
        Returns: (success, target_path, num_injected)
        """
        # Find the source file
        source_file = self._find_source_file(module_docs.module_path)
        if not source_file or not source_file.exists():
            return False, f"Source file not found: {module_docs.module_path}", 0

        # Read original source
        try:
            original_code = source_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return False, f"Failed to read source: {e}", 0

        # Inject docstrings
        modified_code, num_injected = self._inject_docstrings(
            original_code, module_docs.functions
        )

        # In-place save in cloned repo (with backup only if changed)
        if num_injected > 0:
            ensure_backup(source_file)
            source_file.write_text(modified_code, encoding="utf-8")

        return True, str(source_file), num_injected

    def _find_source_file(self, module_path: str) -> Optional[Path]:
        """Find the source file in the repo."""
        # Try direct path
        direct = self.repo_path / module_path.replace("/", os.sep)
        if direct.exists():
            return direct

        # Try with cloned_repo subdirectory
        for subdir in ["", "cloned_repo", "s-tool", "speedtest-cli"]:
            candidate = self.repo_path / subdir / module_path.replace("/", os.sep)
            if candidate.exists():
                return candidate

        # Search for the file
        filename = Path(module_path).name
        for found in self.repo_path.rglob(filename):
            if found.is_file():
                return found

        return None

    def _inject_docstrings(
        self, 
        code: str, 
        functions: List[FunctionDoc]
    ) -> Tuple[str, int]:
        """
        Inject docstrings into the code.
        
        Returns: (modified_code, num_injected)
        """
        lines = code.split("\n")
        injections: List[Tuple[int, str, str]] = []  # (line_no, func_name, docstring)
        
        # Sort functions by line number (descending) to inject from bottom up
        # This prevents line number shifts
        sorted_funcs = sorted(functions, key=lambda f: f.line_no, reverse=True)

        for func in sorted_funcs:
            if not func.docstring:
                continue

            # Find the function definition line
            target_line = func.line_no - 1  # 0-indexed
            if target_line < 0 or target_line >= len(lines):
                continue

            # Check if this line is a function definition
            line = lines[target_line]
            if not self._is_function_def(line, func.name):
                # Search nearby lines
                target_line = self._find_function_line(lines, func.name, func.line_no)
                if target_line is None:
                    continue

            # Check if function already has a docstring
            if self._has_docstring(lines, target_line):
                continue

            injections.append((target_line, func.name, func.docstring))

        # Apply injections (from bottom to top)
        num_injected = 0
        for target_line, func_name, docstring in injections:
            # Get indentation of the function body
            func_line = lines[target_line]
            base_indent = len(func_line) - len(func_line.lstrip())
            body_indent = " " * (base_indent + 4)

            # Format docstring
            formatted_doc = self._format_docstring(docstring, body_indent)

            # Insert after the function definition line
            # Find the line with the colon (could be multi-line def)
            insert_line = target_line
            while insert_line < len(lines) and ":" not in lines[insert_line]:
                insert_line += 1
            insert_line += 1  # Insert after the colon line

            # Insert the docstring
            lines.insert(insert_line, formatted_doc)
            num_injected += 1

            logger.debug("Injected docstring for %s at line %d", func_name, insert_line + 1)

        return "\n".join(lines), num_injected

    def _is_function_def(self, line: str, func_name: str) -> bool:
        """Check if line is a function definition for the given name."""
        stripped = line.strip()
        patterns = [
            f"def {func_name}(",
            f"async def {func_name}(",
        ]
        return any(stripped.startswith(p) for p in patterns)

    def _find_function_line(
        self, 
        lines: List[str], 
        func_name: str, 
        approx_line: int
    ) -> Optional[int]:
        """Find the actual line number of a function definition."""
        # Search within ±10 lines of the approximate line
        start = max(0, approx_line - 10)
        end = min(len(lines), approx_line + 10)

        for i in range(start, end):
            if self._is_function_def(lines[i], func_name):
                return i

        return None

    def _has_docstring(self, lines: List[str], func_line: int) -> bool:
        """Check if the function already has a docstring."""
        # Find the line after the function definition (after the colon)
        check_line = func_line + 1
        while check_line < len(lines):
            line = lines[check_line].strip()
            if not line:
                check_line += 1
                continue
            # Check for docstring
            return line.startswith('"""') or line.startswith("'''")
        return False

    def _format_docstring(self, docstring: str, indent: str) -> str:
        """Format docstring with proper indentation."""
        doc_lines = docstring.split("\n")
        
        # Build formatted docstring
        formatted = [f'{indent}"""']
        for line in doc_lines:
            if line.strip():
                formatted.append(f"{indent}{line}")
            else:
                formatted.append("")
        formatted.append(f'{indent}"""')
        
        return "\n".join(formatted)


# ============================================================================
# SECTION COMMENTER – AST-based comments for classes, imports, configs
# ============================================================================

class SectionCommenter:
    """
    Uses AST analysis to add comments to every section of a Python file:
    - Module-level docstring (if missing)
    - Import section annotation
    - Class docstrings (with field descriptions)
    - Configuration/assignment annotations (e.g. Django settings, URL patterns)
    - Inline section separators
    """

    # Django/common patterns for smart annotations
    DJANGO_PATTERNS = {
        "models.Model": "Django ORM model — maps to a database table",
        "forms.ModelForm": "Django ModelForm — auto-generates form fields from a model",
        "forms.Form": "Django Form — manual form definition",
        "AppConfig": "Django app configuration class",
        "admin.ModelAdmin": "Django admin customisation for this model",
        "TestCase": "Django/unittest test case",
        "APIView": "Django REST Framework API view",
        "ViewSet": "Django REST Framework ViewSet",
        "Serializer": "DRF Serializer — converts model instances to/from JSON",
        "Migration": "Django database migration — auto-generated schema change",
    }

    MODULE_PURPOSE = {
        "models.py": "Database models — defines the data schema for this app.",
        "views.py": "View functions/classes — handle HTTP requests and return responses.",
        "forms.py": "Form definitions — validation and rendering of HTML forms.",
        "urls.py": "URL routing — maps URL patterns to view functions.",
        "admin.py": "Admin site configuration — registers models with Django admin.",
        "apps.py": "App configuration — metadata for the Django application.",
        "settings.py": "Project settings — configuration for the Django project.",
        "tests.py": "Test suite — unit and integration tests for this app.",
        "serializers.py": "DRF Serializers — data serialization/deserialization.",
        "manage.py": "Django management script — entry point for CLI commands.",
        "asgi.py": "ASGI entry point — for async-capable web servers.",
        "wsgi.py": "WSGI entry point — for traditional web servers.",
        "__init__.py": "Package initializer — marks this directory as a Python package.",
        "migrations": "Auto-generated database migration.",
    }

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def comment_file(self, source_file: Path, rel_path: str) -> Tuple[bool, int]:
        """
        Add section comments to a single Python file.
        Returns (success, num_comments_added).
        """
        try:
            code = source_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False, 0

        if not code.strip():
            # Empty file (e.g. __init__.py) — just add a module docstring
            purpose = self._guess_module_purpose(rel_path)
            ensure_backup(source_file)
            source_file.write_text(f'"""{purpose}"""\n', encoding="utf-8")
            return True, 1

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, 0

        lines = code.split("\n")
        insertions: List[Tuple[int, str]] = []  # (line_index, comment_text)

        # 1. Module-level docstring
        if not self._has_module_docstring(tree):
            purpose = self._guess_module_purpose(rel_path)
            insertions.append((0, f'"""\n{purpose}\n"""\n'))

        # 2. Import section comment
        import_lines = self._get_import_lines(tree)
        if import_lines:
            first_import = min(import_lines)
            insertions.append((first_import, "# --- Imports ---"))

        # 3. Class docstrings and field annotations
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_comments = self._build_class_comments(node, lines)
                for line_idx, comment in cls_comments:
                    insertions.append((line_idx, comment))

        # 4. Top-level assignments / config blocks
        config_comments = self._annotate_config(tree, lines, rel_path)
        for line_idx, comment in config_comments:
            insertions.append((line_idx, comment))

        # Apply insertions (sort by line, bottom-up to preserve indices)
        # Deduplicate by line number (keep first)
        seen_lines = set()
        unique_insertions = []
        for line_idx, comment in sorted(insertions, key=lambda x: x[0]):
            if line_idx not in seen_lines:
                seen_lines.add(line_idx)
                unique_insertions.append((line_idx, comment))

        # Insert from bottom to top
        num_added = 0
        for line_idx, comment in sorted(unique_insertions, key=lambda x: x[0], reverse=True):
            # Don't insert duplicate comments
            if line_idx < len(lines) and lines[line_idx].strip() == comment.strip():
                continue
            # Don't add if a comment already exists on the line above
            if line_idx > 0 and lines[line_idx - 1].strip().startswith("#") and comment.startswith("#"):
                continue
            lines.insert(line_idx, comment)
            num_added += 1

        # Save in-place only if comments were added
        if num_added > 0:
            ensure_backup(source_file)
            source_file.write_text("\n".join(lines), encoding="utf-8")

        return True, num_added

    def _has_module_docstring(self, tree: ast.Module) -> bool:
        """Check if the module already has a docstring."""
        if tree.body and isinstance(tree.body[0], ast.Expr):
            val = tree.body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return True
        return False

    def _guess_module_purpose(self, rel_path: str) -> str:
        """Guess the purpose of a module from its filename."""
        filename = Path(rel_path).name
        # Check exact filename match
        if filename in self.MODULE_PURPOSE:
            return self.MODULE_PURPOSE[filename]
        # Check if it's a migration
        if "migration" in rel_path.lower():
            return self.MODULE_PURPOSE["migrations"]
        return f"Module: {rel_path}"

    def _get_import_lines(self, tree: ast.Module) -> List[int]:
        """Get line numbers of all import statements."""
        import_lines = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno - 1)  # 0-indexed
        return import_lines

    def _build_class_comments(
        self, node: ast.ClassDef, lines: List[str]
    ) -> List[Tuple[int, str]]:
        """Build comments for a class: docstring + field annotations."""
        comments: List[Tuple[int, str]] = []
        class_line = node.lineno - 1  # 0-indexed

        # Determine base class description
        bases_str = ", ".join(
            ast.unparse(b) if hasattr(ast, "unparse") else "?"
            for b in node.bases
        )
        base_desc = ""
        for pattern, desc in self.DJANGO_PATTERNS.items():
            if pattern in bases_str:
                base_desc = f"  ({desc})"
                break

        # Check if class already has a docstring
        has_docstring = False
        if node.body and isinstance(node.body[0], ast.Expr):
            val = node.body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                has_docstring = True

        if not has_docstring:
            # Add a class-level comment above the class
            comment = f"# {node.name}{base_desc}"
            comments.append((class_line, comment))

            # Build class docstring content
            doc_parts = [f'Class {node.name}']
            if bases_str:
                doc_parts[0] += f" (inherits from {bases_str})"
            doc_parts[0] += "."
            if base_desc:
                doc_parts.append(base_desc.strip().strip("()"))

            # Gather field info from class body
            fields = self._extract_class_fields(node, lines)
            if fields:
                doc_parts.append("")
                doc_parts.append("Attributes:")
                for fname, ftype, fdesc in fields:
                    if ftype:
                        doc_parts.append(f"    {fname} ({ftype}): {fdesc}")
                    else:
                        doc_parts.append(f"    {fname}: {fdesc}")

            # Find inner Meta class
            for child in node.body:
                if isinstance(child, ast.ClassDef) and child.name == "Meta":
                    meta_info = self._describe_meta(child, lines)
                    if meta_info:
                        doc_parts.append("")
                        doc_parts.append(f"Meta: {meta_info}")

            # Insert docstring after class def line
            indent = " " * (node.col_offset + 4)
            docstring_lines = [f'{indent}"""']
            for part in doc_parts:
                if part:
                    docstring_lines.append(f"{indent}{part}")
                else:
                    docstring_lines.append("")
            docstring_lines.append(f'{indent}"""')
            docstring_text = "\n".join(docstring_lines)

            # Insert after the class definition line
            insert_at = class_line + 1
            comments.append((insert_at, docstring_text))

        return comments

    def _extract_class_fields(
        self, node: ast.ClassDef, lines: List[str]
    ) -> List[Tuple[str, str, str]]:
        """Extract field info from class body assignments."""
        fields = []
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        fname = target.id
                        # Try to describe the field from the source line
                        line = lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                        ftype, fdesc = self._describe_field(fname, line)
                        fields.append((fname, ftype, fdesc))
        return fields

    def _describe_field(self, name: str, line: str) -> Tuple[str, str]:
        """Describe a field from its assignment line."""
        # Django model fields
        django_fields = {
            "CharField": ("str", "Text field"),
            "TextField": ("str", "Long text field"),
            "IntegerField": ("int", "Integer field"),
            "FloatField": ("float", "Floating point field"),
            "BooleanField": ("bool", "Boolean field"),
            "DateTimeField": ("datetime", "Date and time field"),
            "DateField": ("date", "Date field"),
            "EmailField": ("str", "Email address field"),
            "URLField": ("str", "URL field"),
            "FileField": ("File", "File upload field"),
            "ImageField": ("Image", "Image upload field"),
            "ForeignKey": ("FK", "Foreign key relationship"),
            "ManyToManyField": ("M2M", "Many-to-many relationship"),
            "OneToOneField": ("1:1", "One-to-one relationship"),
            "BigAutoField": ("int", "Auto-incrementing primary key"),
            "DecimalField": ("Decimal", "Fixed-point decimal field"),
            "SlugField": ("str", "URL slug field"),
            "JSONField": ("dict", "JSON data field"),
        }

        for field_type, (type_str, desc) in django_fields.items():
            if field_type in line:
                # Extract max_length if present
                ml_match = re.search(r"max_length\s*=\s*(\d+)", line)
                if ml_match:
                    desc += f" (max_length={ml_match.group(1)})"
                return type_str, desc

        return "", "Configuration value"

    def _describe_meta(self, meta_node: ast.ClassDef, lines: List[str]) -> str:
        """Describe a Meta inner class."""
        parts = []
        for child in meta_node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        line = lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                        if target.id == "model":
                            parts.append(f"model={line.split('=')[-1].strip()}")
                        elif target.id == "fields":
                            parts.append(f"fields={line.split('=')[-1].strip()}")
                        elif target.id == "ordering":
                            parts.append(f"ordering={line.split('=')[-1].strip()}")
                        else:
                            parts.append(f"{target.id}={line.split('=')[-1].strip()}")
        return ", ".join(parts) if parts else ""

    def _annotate_config(
        self, tree: ast.Module, lines: List[str], rel_path: str
    ) -> List[Tuple[int, str]]:
        """Add comments for top-level config assignments."""
        comments: List[Tuple[int, str]] = []
        filename = Path(rel_path).name

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        varname = target.id
                        line_idx = node.lineno - 1

                        # Skip if there's already a comment on the line above
                        if line_idx > 0 and lines[line_idx - 1].strip().startswith("#"):
                            continue

                        desc = self._describe_config_var(varname, lines[line_idx], filename)
                        if desc:
                            comments.append((line_idx, f"# {desc}"))

            # URL patterns
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "urlpatterns":
                        line_idx = node.lineno - 1
                        if line_idx > 0 and lines[line_idx - 1].strip().startswith("#"):
                            continue
                        comments.append((line_idx, "# URL routing table — maps URL paths to view functions"))

        return comments

    def _describe_config_var(self, name: str, line: str, filename: str) -> str:
        """Generate a description for a config variable."""
        config_descriptions = {
            "SECRET_KEY": "Secret key for cryptographic signing (keep secret in production!)",
            "DEBUG": "Debug mode toggle — NEVER enable in production",
            "ALLOWED_HOSTS": "Hostnames/IPs this Django site can serve",
            "INSTALLED_APPS": "Registered Django applications and middleware",
            "MIDDLEWARE": "Request/response processing pipeline",
            "ROOT_URLCONF": "Root URL configuration module",
            "TEMPLATES": "Template engine configuration",
            "DATABASES": "Database connection settings",
            "AUTH_PASSWORD_VALIDATORS": "Password validation rules",
            "LANGUAGE_CODE": "Default language for the site",
            "TIME_ZONE": "Server timezone setting",
            "STATIC_URL": "URL prefix for static files (CSS, JS, images)",
            "STATIC_ROOT": "Directory where collectstatic gathers files",
            "STATICFILES_DIRS": "Additional directories for static files",
            "DEFAULT_AUTO_FIELD": "Default primary key field type for models",
            "BASE_DIR": "Base directory path for the project",
            "WSGI_APPLICATION": "WSGI application entry point",
            "ASGI_APPLICATION": "ASGI application entry point",
            "MEDIA_URL": "URL prefix for user-uploaded media files",
            "MEDIA_ROOT": "Directory for user-uploaded files",
            "LOGIN_URL": "URL for the login page",
            "LOGIN_REDIRECT_URL": "URL to redirect after login",
            "LOGOUT_REDIRECT_URL": "URL to redirect after logout",
        }

        # Exact match on common settings
        if name in config_descriptions:
            return config_descriptions[name]

        # Django admin site customisation
        if "site_header" in line:
            return "Admin site header text"
        if "site_title" in line:
            return "Admin site title (shown in browser tab)"
        if "index_title" in line:
            return "Admin index page title"

        # Generic top-level constants (ALL_CAPS)
        if name.isupper() and filename == "settings.py":
            return f"Django setting: {name}"

        return ""

    def process_all_files(self, file_inventory: List[Dict]) -> List[Dict]:
        """
        Process all Python files in the repo, adding section comments.
        Returns list of result dicts.
        """
        results = []

        for entry in file_inventory:
            rel_path = entry.get("file", "")
            if not rel_path.endswith(".py"):
                continue

            source_file = self.repo_path / rel_path.replace("/", os.sep)
            if not source_file.exists():
                continue

            success, num_added = self.comment_file(source_file, rel_path)

            results.append({
                "module": rel_path,
                "success": success,
                "comments_added": num_added,
            })

            if success and num_added > 0:
                logger.info("Added %d section comments to %s", num_added, rel_path)

        return results

    def _comment_existing(self, output_path: Path, code: str, rel_path: str) -> Tuple[bool, int]:
        """Add section comments to a file that was already processed by CodeInjector."""
        if not code.strip():
            return True, 0

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, 0

        lines = code.split("\n")
        insertions: List[Tuple[int, str]] = []

        # Module docstring
        if not self._has_module_docstring(tree):
            purpose = self._guess_module_purpose(rel_path)
            insertions.append((0, f'"""\n{purpose}\n"""\n'))

        # Import section
        import_lines = self._get_import_lines(tree)
        if import_lines:
            first_import = min(import_lines)
            insertions.append((first_import, "# --- Imports ---"))

        # Classes without docstrings
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_comments = self._build_class_comments(node, lines)
                for line_idx, comment in cls_comments:
                    insertions.append((line_idx, comment))

        # Config annotations
        config_comments = self._annotate_config(tree, lines, rel_path)
        for line_idx, comment in config_comments:
            insertions.append((line_idx, comment))

        # Deduplicate and apply
        seen_lines = set()
        unique_insertions = []
        for line_idx, comment in sorted(insertions, key=lambda x: x[0]):
            if line_idx not in seen_lines:
                seen_lines.add(line_idx)
                unique_insertions.append((line_idx, comment))

        num_added = 0
        for line_idx, comment in sorted(unique_insertions, key=lambda x: x[0], reverse=True):
            if line_idx < len(lines) and lines[line_idx].strip() == comment.strip():
                continue
            if line_idx > 0 and lines[line_idx - 1].strip().startswith("#") and comment.startswith("#"):
                continue
            lines.insert(line_idx, comment)
            num_added += 1

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return True, num_added


# ============================================================================
# LINE COMMENTER – LLM-based inline comments on every meaningful line
# ============================================================================

MODEL_ID = "llama-3.3-70b-versatile"

LINE_COMMENT_SYSTEM_PROMPT = """\
You are a Senior Software Engineer adding inline code comments.

Your job is to take a block of Python code and return the EXACT same code \
with helpful inline comments (# comments) added.

Rules:
- Add a short # comment ABOVE or at the END of lines that perform meaningful logic.
- Do NOT comment on obvious lines like blank lines, closing brackets, or simple pass statements.
- Group related lines and add a single comment above the group when appropriate.
- Keep comments SHORT (under 60 characters).
- Do NOT change, remove, or reorder ANY code. Return it exactly as given with comments added.
- Do NOT add triple-quote docstrings — ONLY # comments.
- Do NOT wrap output in markdown code fences or backticks.
- Output ONLY the commented code, nothing else.\
"""


class LineCommenter:
    """
    Uses an LLM to add inline # comments on each meaningful line of code.
    Processes files in cloned_repo/ in-place (after docstrings & section
    comments have already been injected).
    """

    MAX_CHUNK_CHARS = 5000   # max source chars per LLM call (Groq has 128K context)
    SLEEP_BETWEEN   = 0.3    # rate-limit

    def __init__(self, target_root: Path, api_key: str):
        self.target_root = target_root
        if not api_key:
            raise ValueError("GROQ_API_KEY required for LineCommenter")
        self.client = ChatGroq(
            model=MODEL_ID,
            api_key=api_key,
            temperature=0.2,
            max_tokens=4096,
            model_kwargs={"top_p": 0.9},
        )
        logger.info("LineCommenter ready — model %s", MODEL_ID)

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

    # ----- public API --------------------------------------------------------
    def process_all(self, file_inventory: List[Dict]) -> List[Dict]:
        """Add inline comments to every Python file in the target root."""
        results = []
        for entry in file_inventory:
            rel_path = entry.get("file", "")
            if not rel_path.endswith(".py"):
                continue
            out_file = self.target_root / rel_path.replace("/", os.sep)
            if not out_file.exists():
                continue

            success, n_comments = self._process_file(out_file, rel_path)
            results.append({
                "module": rel_path,
                "success": success,
                "inline_comments_added": n_comments,
            })
            if success and n_comments > 0:
                logger.info("Added ~%d inline comments → %s", n_comments, rel_path)
        return results

    # ----- per-file logic ----------------------------------------------------
    def _process_file(self, filepath: Path, rel_path: str) -> Tuple[bool, int]:
        """Read a file, send chunks to LLM, write back with inline comments."""
        try:
            code = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False, 0

        if not code.strip() or len(code) < 20:
            return True, 0

        # Split into logical chunks (function/class bodies, or top-level blocks)
        chunks = self._split_into_chunks(code)
        commented_chunks: List[str] = []
        total_added = 0

        for chunk in chunks:
            if self._is_trivial(chunk):
                # Don't waste an LLM call on blank / import-only blocks
                commented_chunks.append(chunk)
                continue

            commented, n = self._add_comments_to_chunk(chunk)
            commented_chunks.append(commented)
            total_added += n

        # Reassemble file
        final_code = "\n".join(commented_chunks)

        # Safety: verify it still parses
        try:
            ast.parse(final_code)
        except SyntaxError:
            # LLM broke the code — fall back to original
            logger.warning("LLM-commented code has syntax errors, keeping original: %s", rel_path)
            return False, 0

        if final_code != code:
            ensure_backup(filepath)
            filepath.write_text(final_code, encoding="utf-8")
        return True, total_added

    # ----- chunking ----------------------------------------------------------
    def _split_into_chunks(self, code: str) -> List[str]:
        """
        Split code into manageable chunks (≤ MAX_CHUNK_CHARS).
        Splits at top-level function/class boundaries when possible.
        """
        lines = code.split("\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for line in lines:
            is_boundary = (
                line.startswith("def ")
                or line.startswith("async def ")
                or line.startswith("class ")
                or line.startswith("@")  # decorator = new block coming
            )
            if is_boundary and current_len > 200:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            current.append(line)
            current_len += len(line) + 1

            if current_len >= self.MAX_CHUNK_CHARS:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

        if current:
            chunks.append("\n".join(current))

        return chunks

    def _is_trivial(self, chunk: str) -> bool:
        """Skip chunks that are only whitespace, comments, or imports."""
        meaningful = [
            l for l in chunk.split("\n")
            if l.strip()
            and not l.strip().startswith("#")
            and not l.strip().startswith("import ")
            and not l.strip().startswith("from ")
            and not l.strip().startswith('"""')
            and not l.strip().startswith("'''")
        ]
        return len(meaningful) < 3

    # ----- LLM call ----------------------------------------------------------
    def _add_comments_to_chunk(self, chunk: str) -> Tuple[str, int]:
        """Send a code chunk to the LLM and get back commented version."""
        user_prompt = (
            "Add inline # comments to the following Python code. "
            "Return ONLY the code with comments added, nothing else.\n\n"
            f"{chunk}"
        )
        try:
            time.sleep(self.SLEEP_BETWEEN)
            response = self.client.invoke([
                SystemMessage(content=LINE_COMMENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            result = self._coerce_text(response.content)

            # Strip markdown fences if LLM wrapped them
            if result.startswith("```"):
                result = re.sub(r"^```\w*\n?", "", result)
                result = re.sub(r"\n?```$", "", result)

            # Count added comment lines
            orig_comments = sum(1 for l in chunk.split("\n") if l.strip().startswith("#"))
            new_comments  = sum(1 for l in result.split("\n") if l.strip().startswith("#"))
            n_added = max(0, new_comments - orig_comments)

            return result, n_added

        except Exception as e:
            logger.warning("LLM call failed for chunk (%s), keeping original", e)
            return chunk, 0


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class InjectComments:
    """
    Main orchestrator for injecting comments into source code.
    """

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_analysis(self) -> Dict:
        """Load analysis_results.json."""
        if not ANALYSIS_FILE.exists():
            raise FileNotFoundError(f"analysis_results.json not found at {ANALYSIS_FILE}")
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> None:
        """Execute the injection process."""
        load_dotenv()
        _p2_env = BASE_DIR.parent / "project2" / ".env"
        if _p2_env.exists():
            load_dotenv(_p2_env)

        console.print(Panel(
            "[bold cyan]Code-to-Doc  ·  Inject Comments[/bold cyan]\n"
            "[white]Insert generated docstrings + inline comments into source code (in-place)[/white]\n"
            "[dim]Target: cloned_repo (with .bak backup files per changed file)[/dim]",
            expand=False,
        ))

        # 1. Load analysis data
        data = self._load_analysis()
        project_name = data.get("metadata", {}).get("project_name", "Unknown")
        repo_path = Path(data.get("repo_path", ""))
        
        console.print(f"\n📂 Project: [bold]{project_name}[/bold]")
        console.print(f"📁 Source: {repo_path}")

        # 2. Extract documentation from Phase 4 reports
        extractor = Phase4Extractor(PHASE4_DIR)
        modules = extractor.extract_all()
        
        total_functions = sum(len(m.functions) for m in modules)
        console.print(f"📝 Found [bold]{total_functions}[/bold] documented functions in [bold]{len(modules)}[/bold] modules")

        # 3. Inject Phase 4 docstrings into cloned repo source code (in-place)
        injector = CodeInjector(repo_path)
        
        results = []
        total_injected = 0

        if modules:
            console.print(f"\n🔧 Injecting function docstrings into source files…\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Injecting…", total=len(modules))

                for module in modules:
                    progress.update(task, description=f"{module.module_path}")
                    
                    success, output_path, num_injected = injector.inject_module(module)
                    results.append({
                        "module": module.module_path,
                        "success": success,
                        "output_path": output_path,
                        "functions_available": len(module.functions),
                        "injected": num_injected,
                    })
                    total_injected += num_injected

                    progress.advance(task)

        # 4. Section Commenter — add comments to ALL files (classes, imports, configs)
        file_inventory_raw = data.get("file_inventory", {})
        # Build flat list of file dicts from the nested inventory structure
        file_inventory = []
        if isinstance(file_inventory_raw, dict):
            inv = file_inventory_raw.get("inventory", {})
            if isinstance(inv, dict):
                for category_files in inv.values():
                    if isinstance(category_files, list):
                        for entry in category_files:
                            if isinstance(entry, dict) and entry.get("path", "").endswith(".py"):
                                file_inventory.append({"file": entry["path"].replace("\\", "/")})
            elif isinstance(inv, list):
                for entry in inv:
                    if isinstance(entry, dict) and entry.get("path", "").endswith(".py"):
                        file_inventory.append({"file": entry["path"].replace("\\", "/")})

        if not file_inventory:
            # Fallback: scan the repo directly
            for py in sorted(repo_path.rglob("*.py")):
                rel = str(py.relative_to(repo_path)).replace(os.sep, "/")
                file_inventory.append({"file": rel})

        console.print(f"\n📝 Adding section comments to all {len(file_inventory)} files…\n")

        commenter = SectionCommenter(repo_path)
        section_results = commenter.process_all_files(file_inventory)

        total_section_comments = sum(r["comments_added"] for r in section_results)
        files_commented = sum(1 for r in section_results if r["comments_added"] > 0)
        console.print(f"   Added [bold]{total_section_comments}[/bold] section comments across [bold]{files_commented}[/bold] files")

        # 5. Inline Line Comments (LLM-based) — add # comments on meaningful lines
        inline_results = []
        total_inline_comments = 0
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if groq_api_key:
            console.print(f"\n💬 Adding inline line-by-line comments via LLM…\n")
            try:
                line_commenter = LineCommenter(repo_path, groq_api_key)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=console,
                ) as progress:
                    py_files = [e for e in file_inventory if e["file"].endswith(".py")]
                    task = progress.add_task("Commenting…", total=len(py_files))
                    for entry in py_files:
                        rel = entry["file"]
                        progress.update(task, description=rel)
                        out_file = repo_path / rel.replace("/", os.sep)
                        if out_file.exists():
                            ok, n = line_commenter._process_file(out_file, rel)
                            inline_results.append({"module": rel, "success": ok, "inline_comments_added": n})
                            total_inline_comments += n
                        progress.advance(task)

                files_inline = sum(1 for r in inline_results if r["inline_comments_added"] > 0)
                console.print(f"   Added ~[bold]{total_inline_comments}[/bold] inline comments across [bold]{files_inline}[/bold] files")
            except Exception as e:
                console.print(f"   [yellow]⚠ Inline comments skipped: {e}[/yellow]")
        else:
            console.print("\n[yellow]⚠ GROQ_API_KEY not set — skipping inline line comments (LLM required)[/yellow]")

        # 6. Generate summary report
        self._generate_report(project_name, results, total_injected,
                              section_results, total_section_comments,
                              inline_results, total_inline_comments)

        # 7. Summary table
        table = Table(title="Injection Summary — Function Docstrings")
        table.add_column("Module", style="bold")
        table.add_column("Functions", justify="right")
        table.add_column("Injected", justify="right")
        table.add_column("Status")

        for r in results:
            status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
            table.add_row(
                r["module"],
                str(r["functions_available"]),
                str(r["injected"]),
                status,
            )

        console.print(table)

        # Section comments table
        table2 = Table(title="Section Comments — Classes, Imports, Configs")
        table2.add_column("Module", style="bold")
        table2.add_column("Comments Added", justify="right")
        table2.add_column("Status")

        for r in section_results:
            if r["comments_added"] > 0 or not r["success"]:
                status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
                table2.add_row(
                    r["module"],
                    str(r["comments_added"]),
                    status,
                )

        console.print(table2)

        # Inline comments table
        if inline_results:
            table3 = Table(title="Inline Line Comments (LLM)")
            table3.add_column("Module", style="bold")
            table3.add_column("Comments Added", justify="right")
            table3.add_column("Status")
            for r in inline_results:
                if r["inline_comments_added"] > 0 or not r["success"]:
                    status = "[green]✓[/green]" if r["success"] else "[red]✗[/red]"
                    table3.add_row(r["module"], str(r["inline_comments_added"]), status)
            console.print(table3)

        console.print(Panel(
            f"[green]✓ Injection complete[/green]\n"
            f"Target repo (in-place): {repo_path}\n"
            f"Function docstrings injected: {total_injected}\n"
            f"Section comments added: {total_section_comments}\n"
            f"Inline line comments added: ~{total_inline_comments}\n"
            "Backups: `.bak` files created for changed files",
            expand=False,
        ))

    def _generate_report(
        self, 
        project_name: str, 
        results: List[Dict],
        total_injected: int,
        section_results: List[Dict] = None,
        total_section_comments: int = 0,
        inline_results: List[Dict] = None,
        total_inline_comments: int = 0,
    ) -> None:
        """Generate a report of the injection process."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "INJECTION_REPORT.md"

        lines = [
            f"# Docstring Injection Report",
            f"## Project: `{project_name}`",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Modules with function docs | {len(results)} |",
            f"| Function docstrings injected | {total_injected} |",
            f"| Files with section comments | {sum(1 for r in (section_results or []) if r['comments_added'] > 0)} |",
            f"| Total section comments added | {total_section_comments} |",
            f"| Inline line comments added | ~{total_inline_comments} |",
            f"| Target directory | `cloned_repo` (in-place edits) |",
            f"| Backup files | `.bak` next to each changed file |",
            "",
            "---",
            "",
            "## Function Docstring Injection",
            "",
            "| Module | Available | Injected | Status |",
            "|--------|-----------|----------|--------|",
        ]

        for r in results:
            status = "✓ Success" if r["success"] else f"✗ {r['output_path']}"
            lines.append(
                f"| `{r['module']}` | {r['functions_available']} | {r['injected']} | {status} |"
            )

        # Section comments table
        if section_results:
            lines.extend([
                "",
                "---",
                "",
                "## Section Comments (Classes, Imports, Configs)",
                "",
                "| Module | Comments Added | Status |",
                "|--------|---------------|--------|",
            ])
            for r in section_results:
                if r["comments_added"] > 0:
                    status = "✓" if r["success"] else "✗"
                    lines.append(f"| `{r['module']}` | {r['comments_added']} | {status} |")

        # Inline comments table
        if inline_results:
            lines.extend([
                "",
                "---",
                "",
                "## Inline Line Comments (LLM-generated)",
                "",
                "| Module | Comments Added | Status |",
                "|--------|---------------|--------|",
            ])
            for r in inline_results:
                if r["inline_comments_added"] > 0:
                    status = "✓" if r["success"] else "✗"
                    lines.append(f"| `{r['module']}` | ~{r['inline_comments_added']} | {status} |")

        lines.extend([
            "",
            "---",
            "",
            "## How to Compare",
            "",
            "1. Open the changed file in `cloned_repo/`",
            "2. Open its backup file with the same name + `.bak`",
            "3. Use a diff tool to compare:",
            "   ```",
            "   # VS Code",
            "   code --diff file.py.bak file.py",
            "   ```",
            "",
            "The injected comments/docstrings are applied directly in cloned_repo files.",
            "",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Injection report saved → %s", report_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    injector = InjectComments()
    injector.run()


if __name__ == "__main__":
    main()
