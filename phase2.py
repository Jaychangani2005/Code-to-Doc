"""
Code-to-Doc: Phase 2 – Code Understanding & Dependency Mapping
===============================================================
Prompt 2.1 – Dependency Graph Analysis
Prompt 2.2 – Architecture Understanding

Steps:
  1. Load Phase 1 analysis_results.json
  2. AST-parse every Python source file (functions, classes, imports, constants)
  3. Build dependency graph (internal imports, external deps)
  4. Compute complexity metrics (cyclomatic via AST, LOC, maintainability)
  5. Detect circular dependencies
  6. Identify core modules, highly coupled modules, module hierarchy
  7. Detect design patterns (Factory, Singleton, Observer, Decorator, …)
  8. Analyse architecture: layers, data flow, entry points, config handling
  9. Merge everything back into output/analysis_results.json

Technology: AST, networkx, radon (optional), rich
Output    : output/analysis_results.json  (enriched with Phase 2 data)
"""

import os
import ast
import json
import math
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import chardet
except ImportError:
    chardet = None

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
OUTPUT_DIR    = BASE_DIR / "output"
ANALYSIS_FILE = OUTPUT_DIR / "analysis_results.json"


# ============================================================================
# HELPERS
# ============================================================================

def _read_file_text(path: Path) -> Optional[str]:
    """Read a text file, auto-detecting encoding."""
    try:
        raw = path.read_bytes()
        if chardet:
            enc = chardet.detect(raw[:8192]).get("encoding") or "utf-8"
        else:
            enc = "utf-8"
        return raw.decode(enc, errors="replace")
    except Exception:
        return None


def _safe_unparse(node) -> str:
    """ast.unparse with fallback."""
    try:
        return ast.unparse(node)
    except Exception:
        return "???"


# ============================================================================
# CODE PARSER  (AST-based, Python only)
# ============================================================================

class CodeParser:
    """
    Parses every Python source file via the AST and extracts:
      - imports (stdlib, internal, external)
      - functions (signature, params, return type, decorators, docstring, complexity)
      - classes   (bases, methods, attributes, docstring)
      - module-level constants
      - module docstring
    """

    def __init__(self, repo_path: Path, source_files: List[Dict]):
        self.repo_path = repo_path
        self.source_files = source_files        # list of dicts from Phase 1 inventory
        self.parsed_modules: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    def parse_all(self) -> Dict[str, Dict]:
        console.print("\n[bold cyan]── Parsing Python source files (AST) ──[/]")

        py_files = [f for f in self.source_files if f["extension"] == ".py"]

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as prog:
            task = prog.add_task("Parsing…", total=len(py_files))
            for file_info in py_files:
                prog.advance(task)
                rel = file_info["path"]
                abs_path = self.repo_path / rel
                if not abs_path.exists():
                    continue
                mod = self._parse_one(abs_path, rel)
                if mod:
                    self.parsed_modules[rel] = mod

        console.print(f"[green]✓ Parsed {len(self.parsed_modules)} Python modules[/]")
        return self.parsed_modules

    # ------------------------------------------------------------------
    def _parse_one(self, abs_path: Path, rel: str) -> Optional[Dict]:
        src = _read_file_text(abs_path)
        if src is None:
            return None
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as exc:
            logger.warning(f"SyntaxError in {rel}: {exc}")
            return None

        imports   = self._extract_imports(tree)
        functions = []
        classes   = []
        constants = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._extract_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))
            elif isinstance(node, ast.Assign):
                constants.extend(self._extract_constants(node))

        return {
            "file":       rel,
            "docstring":  ast.get_docstring(tree) or "",
            "imports":    imports,
            "functions":  functions,
            "classes":    classes,
            "constants":  constants,
            "has_main":   self._has_main_guard(tree),
            "total_defs": len(functions) + sum(len(c["methods"]) for c in classes),
        }

    # ── imports ──────────────────────────────────────────────────────
    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        imports: List[Dict] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module":    alias.name,
                        "name":      alias.asname or alias.name,
                        "type":      "import",
                        "lineno":    node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in (node.names or []):
                    imports.append({
                        "module":    mod,
                        "name":      alias.name,
                        "type":      "from",
                        "lineno":    node.lineno,
                    })
        return imports

    # ── functions ────────────────────────────────────────────────────
    def _extract_function(self, node) -> Dict:
        params = self._extract_params(node)
        return {
            "name":        node.name,
            "lineno":      node.lineno,
            "end_lineno":  getattr(node, "end_lineno", None),
            "docstring":   ast.get_docstring(node) or "",
            "signature":   self._build_signature(node),
            "parameters":  params,
            "return_type": self._annotation_str(node.returns),
            "decorators":  [self._decorator_name(d) for d in node.decorator_list],
            "is_async":    isinstance(node, ast.AsyncFunctionDef),
            "is_private":  node.name.startswith("_"),
            "complexity":  self._cyclomatic(node),
        }

    def _extract_params(self, node) -> List[Dict]:
        params = []
        args = node.args
        defaults_off = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            p = {
                "name": arg.arg,
                "type": self._annotation_str(arg.annotation),
                "default": None,
                "kind": "positional",
            }
            di = i - defaults_off
            if di >= 0:
                p["default"] = _safe_unparse(args.defaults[di])
            params.append(p)
        if args.vararg:
            params.append({"name": f"*{args.vararg.arg}", "type": self._annotation_str(args.vararg.annotation), "default": None, "kind": "var_positional"})
        if args.kwarg:
            params.append({"name": f"**{args.kwarg.arg}", "type": self._annotation_str(args.kwarg.annotation), "default": None, "kind": "var_keyword"})
        return params

    def _build_signature(self, node) -> str:
        try:
            parts = []
            args = node.args
            defaults_off = len(args.args) - len(args.defaults)
            for i, arg in enumerate(args.args):
                s = arg.arg
                if arg.annotation:
                    s += f": {_safe_unparse(arg.annotation)}"
                di = i - defaults_off
                if di >= 0:
                    s += f" = {_safe_unparse(args.defaults[di])}"
                parts.append(s)
            if args.vararg:
                parts.append(f"*{args.vararg.arg}")
            if args.kwarg:
                parts.append(f"**{args.kwarg.arg}")
            sig = f"{node.name}({', '.join(parts)})"
            if node.returns:
                sig += f" -> {_safe_unparse(node.returns)}"
            return sig
        except Exception:
            return f"{node.name}(…)"

    # ── classes ──────────────────────────────────────────────────────
    def _extract_class(self, node: ast.ClassDef) -> Dict:
        methods = []
        attrs = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item))
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        attrs.append({"name": t.id, "lineno": item.lineno})
        return {
            "name":       node.name,
            "lineno":     node.lineno,
            "end_lineno": getattr(node, "end_lineno", None),
            "docstring":  ast.get_docstring(node) or "",
            "bases":      [_safe_unparse(b) for b in node.bases],
            "decorators": [self._decorator_name(d) for d in node.decorator_list],
            "methods":    methods,
            "attributes": attrs,
            "is_private": node.name.startswith("_"),
        }

    # ── constants ────────────────────────────────────────────────────
    def _extract_constants(self, node: ast.Assign) -> List[Dict]:
        out = []
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.isupper():
                out.append({
                    "name":   t.id,
                    "lineno": node.lineno,
                    "value":  _safe_unparse(node.value)[:120],
                })
        return out

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _annotation_str(ann) -> str:
        if ann is None:
            return "Any"
        return _safe_unparse(ann)

    @staticmethod
    def _decorator_name(dec) -> str:
        try:
            if isinstance(dec, ast.Name):
                return dec.id
            if isinstance(dec, ast.Call):
                return _safe_unparse(dec.func)
            return _safe_unparse(dec)
        except Exception:
            return "unknown"

    @staticmethod
    def _cyclomatic(node) -> int:
        """McCabe cyclomatic complexity of a single function/method."""
        c = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                  ast.With, ast.Assert)):
                c += 1
            elif isinstance(child, ast.BoolOp):
                c += len(child.values) - 1
        return c

    @staticmethod
    def _has_main_guard(tree: ast.AST) -> bool:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.If):
                try:
                    test_src = _safe_unparse(node.test)
                    if "__name__" in test_src and "__main__" in test_src:
                        return True
                except Exception:
                    pass
        return False


# ============================================================================
# DEPENDENCY GRAPH BUILDER  (Prompt 2.1)
# ============================================================================

class DependencyGraphBuilder:
    """
    Builds a module-level dependency graph for every Python source file.

    For each module tracks:
        - internal imports   (imports referring to another project module)
        - external imports   (third-party / stdlib)
        - imported_by        (reverse edges)

    Uses networkx (if available) for cycle detection & centrality.
    """

    # Standard library top-level packages (Python 3.10+); subset is enough
    _STDLIB_TOP = {
        "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
        "asyncore", "atexit", "base64", "bdb", "binascii", "binhex",
        "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk",
        "cmath", "cmd", "code", "codecs", "codeop", "collections",
        "colorsys", "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
        "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
        "difflib", "dis", "distutils", "doctest", "email", "encodings",
        "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
        "fnmatch", "formatter", "fractions", "ftplib", "functools", "gc",
        "getopt", "getpass", "gettext", "glob", "grp", "gzip", "hashlib",
        "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
        "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
        "json", "keyword", "lib2to3", "linecache", "locale", "logging",
        "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
        "mmap", "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
        "numbers", "operator", "optparse", "os", "ossaudiodev", "parser",
        "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
        "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
        "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
        "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
        "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
        "selectors", "shelve", "shlex", "shutil", "signal", "site",
        "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "sqlite3",
        "ssl", "stat", "statistics", "string", "stringprep", "struct",
        "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog",
        "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "test",
        "textwrap", "threading", "time", "timeit", "tkinter", "token",
        "tokenize", "trace", "traceback", "tracemalloc", "tty", "turtle",
        "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
        "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
        "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
        "zipapp", "zipfile", "zipimport", "zlib", "_thread",
        "typing_extensions", "t", "annotations",
    }

    def __init__(self, repo_path: Path, source_files: List[Dict],
                 parsed_modules: Dict[str, Dict]):
        self.repo_path = repo_path
        self.source_files = source_files
        self.parsed_modules = parsed_modules

        # Build a lookup:  "click.core" → "src\\click\\core.py"  etc.
        self._module_lookup: Dict[str, str] = self._build_module_lookup()

        # adjacency
        self.graph: Dict[str, Dict] = {}
        for rel in parsed_modules:
            self.graph[rel] = {
                "imports":       [],
                "imported_by":   [],
                "external_deps": [],
            }

    # ------------------------------------------------------------------
    def build(self) -> Dict[str, Dict]:
        """Build the full dependency graph and return it."""
        console.print("\n[bold cyan]── Prompt 2.1: Dependency Graph Analysis ──[/]")

        for rel, mod in self.parsed_modules.items():
            ext_deps: List[str] = []
            int_deps: List[str] = []
            for imp in mod.get("imports", []):
                target = self._resolve_import(imp["module"], rel)
                if target:
                    int_deps.append(target)
                else:
                    top = imp["module"].split(".")[0]
                    ext_deps.append(imp["module"])
            self.graph[rel]["imports"]       = sorted(set(int_deps))
            self.graph[rel]["external_deps"] = sorted(set(ext_deps))

        # reverse edges
        for rel, data in self.graph.items():
            for dep in data["imports"]:
                if dep in self.graph:
                    self.graph[dep]["imported_by"].append(rel)
        for data in self.graph.values():
            data["imported_by"] = sorted(set(data["imported_by"]))

        self._display_graph_summary()
        return self.graph

    # ------------------------------------------------------------------
    def _build_module_lookup(self) -> Dict[str, str]:
        """Map dotted module name → relative file path."""
        lookup: Dict[str, str] = {}
        for rel in self.parsed_modules:
            # "src\\click\\core.py" → "src.click.core"
            dotted = rel.replace("\\", "/").replace("/", ".").removesuffix(".py")
            lookup[dotted] = rel
            # also store without the leading "src." prefix if present
            if dotted.startswith("src."):
                lookup[dotted[4:]] = rel
            # also store just the filename stem
            stem = Path(rel).stem
            if stem != "__init__":
                lookup[stem] = rel
        return lookup

    def _resolve_import(self, module_name: str, from_file: str) -> Optional[str]:
        """Try to resolve an import to an internal project file."""
        if not module_name:
            return None
        # Direct lookup
        if module_name in self._module_lookup:
            return self._module_lookup[module_name]
        # Try progressively shorter prefixes  click.core → click
        parts = module_name.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in self._module_lookup:
                return self._module_lookup[candidate]
        return None

    # ------------------------------------------------------------------
    def analyse_graph(self) -> Dict:
        """
        Prompt 2.1 analysis:
          1. Core modules (highest in-degree)
          2. Highly coupled modules
          3. Circular dependencies
          4. External dependencies summary
          5. Internal module hierarchy
        """
        core_modules   = self._core_modules()
        coupled        = self._highly_coupled()
        circular       = self._detect_cycles()
        ext_summary    = self._external_summary()
        hierarchy      = self._module_hierarchy()
        centrality     = self._betweenness_centrality()

        analysis = {
            "core_modules":           core_modules,
            "highly_coupled_modules": coupled,
            "circular_dependencies":  circular,
            "external_dependencies":  ext_summary,
            "module_hierarchy":       hierarchy,
            "centrality":             centrality,
        }
        self._display_analysis(analysis)
        return analysis

    # ── core modules (top-10 by in-degree) ──
    def _core_modules(self) -> List[Dict]:
        ranked = sorted(
            self.graph.items(),
            key=lambda kv: len(kv[1]["imported_by"]),
            reverse=True,
        )
        return [
            {"module": m, "imported_by_count": len(d["imported_by"])}
            for m, d in ranked[:10]
        ]

    # ── highly coupled (imports + imported_by both high) ──
    def _highly_coupled(self) -> List[Dict]:
        coupled = []
        for m, d in self.graph.items():
            fan_in  = len(d["imported_by"])
            fan_out = len(d["imports"])
            if fan_in + fan_out >= 4:                 # threshold
                coupled.append({
                    "module":  m,
                    "fan_in":  fan_in,
                    "fan_out": fan_out,
                    "coupling_score": fan_in + fan_out,
                })
        coupled.sort(key=lambda x: x["coupling_score"], reverse=True)
        return coupled[:15]

    # ── cycle detection ──
    def _detect_cycles(self) -> List[List[str]]:
        if HAS_NETWORKX:
            G = nx.DiGraph()
            for m, d in self.graph.items():
                for dep in d["imports"]:
                    G.add_edge(m, dep)
            try:
                cycles = list(nx.simple_cycles(G))
                return [list(c) for c in cycles[:20]]
            except Exception:
                pass
        # fallback DFS
        return self._dfs_cycles()

    def _dfs_cycles(self) -> List[List[str]]:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for nb in self.graph.get(node, {}).get("imports", []):
                if nb not in self.graph:
                    continue
                if nb not in visited:
                    dfs(nb, path[:])
                elif nb in rec_stack:
                    cycles.append(path + [nb])
            rec_stack.discard(node)

        for n in list(self.graph):
            if n not in visited:
                dfs(n, [])
        return cycles[:20]

    # ── external deps summary ──
    def _external_summary(self) -> Dict:
        counter: Counter = Counter()
        for d in self.graph.values():
            for e in d["external_deps"]:
                counter[e.split(".")[0]] += 1
        stdlib = {}
        third_party = {}
        for pkg, cnt in counter.most_common():
            if pkg in self._STDLIB_TOP:
                stdlib[pkg] = cnt
            else:
                third_party[pkg] = cnt
        return {"stdlib": stdlib, "third_party": third_party}

    # ── hierarchy (directory nesting) ──
    def _module_hierarchy(self) -> Dict:
        tree: Dict = {}
        for rel in self.graph:
            parts = Path(rel).parts
            node = tree
            for p in parts:
                node = node.setdefault(p, {})
        return tree

    # ── betweenness centrality via networkx ──
    def _betweenness_centrality(self) -> Dict[str, float]:
        if not HAS_NETWORKX:
            return {}
        G = nx.DiGraph()
        for m, d in self.graph.items():
            for dep in d["imports"]:
                G.add_edge(m, dep)
        if len(G) == 0:
            return {}
        bc = nx.betweenness_centrality(G)
        # return top-10
        top = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return {m: round(v, 4) for m, v in top}

    # ── display helpers ──
    def _display_graph_summary(self):
        total_internal = sum(len(d["imports"]) for d in self.graph.values())
        total_external = sum(len(d["external_deps"]) for d in self.graph.values())
        t = Table(title="Dependency Graph Summary", show_lines=True)
        t.add_column("Metric",  style="cyan",  min_width=30)
        t.add_column("Value",   style="green", justify="right")
        t.add_row("Modules in graph",      str(len(self.graph)))
        t.add_row("Internal dependency edges", str(total_internal))
        t.add_row("External dependency refs",  str(total_external))
        console.print(t)

    def _display_analysis(self, a: Dict):
        # core modules
        t = Table(title="Core Modules (most imported)", show_lines=True)
        t.add_column("Module", style="cyan")
        t.add_column("Imported By #", style="green", justify="right")
        for cm in a["core_modules"][:8]:
            t.add_row(cm["module"], str(cm["imported_by_count"]))
        console.print(t)

        # circular
        if a["circular_dependencies"]:
            console.print(f"[red]⚠  {len(a['circular_dependencies'])} circular dependency cycle(s) detected![/]")
            for cyc in a["circular_dependencies"][:5]:
                console.print(f"  [yellow]{'  →  '.join(cyc)}[/]")
        else:
            console.print("[green]✓ No circular dependencies detected[/]")

        # external deps
        tp = a["external_dependencies"].get("third_party", {})
        if tp:
            console.print(Panel(
                ", ".join(f"{k} ({v})" for k, v in list(tp.items())[:15]),
                title="Third-party Dependencies",
                border_style="yellow",
            ))


# ============================================================================
# COMPLEXITY METRICS
# ============================================================================

class ComplexityAnalyzer:
    """Compute file-level and project-level complexity metrics."""

    def __init__(self, parsed_modules: Dict[str, Dict], source_files: List[Dict]):
        self.parsed_modules = parsed_modules
        self.source_files = source_files

    def compute(self) -> Dict:
        console.print("\n[bold cyan]── Computing complexity metrics ──[/]")

        per_file: List[Dict] = []
        all_complexities: List[int] = []

        for rel, mod in self.parsed_modules.items():
            funcs = mod.get("functions", [])
            for cls in mod.get("classes", []):
                funcs = funcs + cls.get("methods", [])

            complexities = [f["complexity"] for f in funcs]
            avg_cc = (sum(complexities) / len(complexities)) if complexities else 0
            max_cc = max(complexities) if complexities else 0
            all_complexities.extend(complexities)

            # find matching source info for LOC
            loc = 0
            size_kb = 0.0
            for sf in self.source_files:
                if sf["path"] == rel:
                    loc = sf.get("lines", 0)
                    size_kb = sf.get("size_kb", 0.0)
                    break

            # Maintainability index (simplified Halstead-free formula)
            # MI = max(0, 171 − 5.2 × ln(V) − 0.23 × CC − 16.2 × ln(LOC)) scaled 0-100
            mi = self._maintainability_index(loc, avg_cc)

            per_file.append({
                "file":          rel,
                "loc":           loc,
                "size_kb":       size_kb,
                "num_functions":  len(mod.get("functions", [])),
                "num_classes":   len(mod.get("classes", [])),
                "total_defs":    mod.get("total_defs", 0),
                "avg_complexity": round(avg_cc, 2),
                "max_complexity": max_cc,
                "maintainability_index": mi,
            })

        # Project-level
        total_loc   = sum(f["loc"] for f in per_file)
        total_files = len(per_file)
        avg_cc_all  = (sum(all_complexities) / len(all_complexities)) if all_complexities else 0
        max_cc_all  = max(all_complexities) if all_complexities else 0

        result = {
            "project_level": {
                "total_source_files": total_files,
                "total_loc":          total_loc,
                "average_complexity": round(avg_cc_all, 2),
                "max_complexity":     max_cc_all,
                "high_complexity_functions": self._high_complexity_funcs(),
            },
            "per_file": sorted(per_file, key=lambda x: x["max_complexity"], reverse=True),
        }
        self._display(result)
        return result

    def _maintainability_index(self, loc: int, avg_cc: float) -> float:
        if loc <= 0:
            return 100.0
        try:
            mi = max(0, (171 - 5.2 * math.log(max(loc, 1)) - 0.23 * avg_cc - 16.2 * math.log(max(loc, 1))) * 100 / 171)
            return round(mi, 1)
        except Exception:
            return 0.0

    def _high_complexity_funcs(self) -> List[Dict]:
        """Functions with cyclomatic complexity ≥ 10."""
        out = []
        for rel, mod in self.parsed_modules.items():
            for f in mod.get("functions", []):
                if f["complexity"] >= 10:
                    out.append({"file": rel, "function": f["name"], "complexity": f["complexity"]})
            for cls in mod.get("classes", []):
                for m in cls.get("methods", []):
                    if m["complexity"] >= 10:
                        out.append({"file": rel, "class": cls["name"], "method": m["name"], "complexity": m["complexity"]})
        out.sort(key=lambda x: x["complexity"], reverse=True)
        return out[:20]

    def _display(self, r: Dict):
        p = r["project_level"]
        t = Table(title="Complexity Metrics", show_lines=True)
        t.add_column("Metric", style="cyan", min_width=30)
        t.add_column("Value",  style="green", justify="right")
        t.add_row("Source files parsed",   str(p["total_source_files"]))
        t.add_row("Total LOC",            f"{p['total_loc']:,}")
        t.add_row("Average CC",           str(p["average_complexity"]))
        t.add_row("Max CC",               str(p["max_complexity"]))
        t.add_row("High-CC functions (≥10)", str(len(p["high_complexity_functions"])))
        console.print(t)


# ============================================================================
# ARCHITECTURE ANALYSER  (Prompt 2.2)
# ============================================================================

class ArchitectureAnalyzer:
    """
    Prompt 2.2 – Architecture Understanding:
      1. Core architectural components
      2. Design patterns (MVC, Singleton, Factory, Observer, Decorator, …)
      3. Data flow between modules
      4. Entry points & execution flow
      5. Configuration handling
      6. Database / API integrations
    """

    def __init__(self, repo_path: Path, parsed_modules: Dict[str, Dict],
                 dep_graph: Dict[str, Dict]):
        self.repo_path = repo_path
        self.parsed   = parsed_modules
        self.graph    = dep_graph

    # ------------------------------------------------------------------
    def analyse(self) -> Dict:
        console.print("\n[bold cyan]── Prompt 2.2: Architecture Understanding ──[/]")

        result = {
            "core_components":    self._core_components(),
            "design_patterns":    self._detect_patterns(),
            "data_flow":          self._data_flow(),
            "entry_points":       self._entry_points(),
            "config_handling":    self._config_handling(),
            "integrations":       self._detect_integrations(),
            "layers":             self._identify_layers(),
        }
        self._display(result)
        return result

    # ── 1. core components ───────────────────────────────────────────
    def _core_components(self) -> List[Dict]:
        """Modules with most definitions (functions + classes)."""
        ranked = sorted(
            self.parsed.items(),
            key=lambda kv: kv[1].get("total_defs", 0),
            reverse=True,
        )
        return [
            {"module": m, "total_defs": d["total_defs"],
             "classes": [c["name"] for c in d.get("classes", [])],
             "top_functions": [f["name"] for f in d.get("functions", [])[:5]]}
            for m, d in ranked[:10]
        ]

    # ── 2. design patterns ──────────────────────────────────────────
    def _detect_patterns(self) -> Dict[str, List[str]]:
        patterns: Dict[str, List[str]] = {
            "singleton":  [],
            "factory":    [],
            "decorator":  [],
            "observer":   [],
            "strategy":   [],
            "template_method": [],
            "context_manager": [],
        }
        for rel, mod in self.parsed.items():
            for cls in mod.get("classes", []):
                name_low = cls["name"].lower()
                bases_low = " ".join(cls.get("bases", [])).lower()
                methods = {m["name"] for m in cls.get("methods", [])}
                decorators_all = set()
                for m in cls.get("methods", []):
                    decorators_all.update(m.get("decorators", []))

                # Singleton
                if "__new__" in methods or "__init__" in methods:
                    if "_instance" in " ".join(a["name"] for a in cls.get("attributes", [])):
                        patterns["singleton"].append(f"{rel}::{cls['name']}")

                # Factory
                if "factory" in name_low or "create" in name_low:
                    patterns["factory"].append(f"{rel}::{cls['name']}")
                if any("classmethod" in d for d in decorators_all):
                    for m in cls.get("methods", []):
                        if m["name"].startswith("create") or m["name"].startswith("from_"):
                            patterns["factory"].append(f"{rel}::{cls['name']}.{m['name']}")

                # Decorator pattern (wraps / __call__)
                if "__call__" in methods or "__enter__" in methods:
                    patterns["decorator"].append(f"{rel}::{cls['name']}")

                # Observer
                if any(kw in name_low for kw in ("listener", "handler", "observer", "event", "signal")):
                    patterns["observer"].append(f"{rel}::{cls['name']}")

                # Context Manager
                if "__enter__" in methods and "__exit__" in methods:
                    patterns["context_manager"].append(f"{rel}::{cls['name']}")

                # Strategy / Template Method
                if "abc.ABC" in bases_low or "ABC" in cls.get("bases", []):
                    patterns["strategy"].append(f"{rel}::{cls['name']}")

            # function-level decorator pattern
            for func in mod.get("functions", []):
                if any(d in ("contextmanager", "contextlib.contextmanager") for d in func.get("decorators", [])):
                    patterns["context_manager"].append(f"{rel}::{func['name']}")

        # remove empties
        return {k: v for k, v in patterns.items() if v}

    # ── 3. data flow ─────────────────────────────────────────────────
    def _data_flow(self) -> List[Dict]:
        """Simplified: module A → module B edges with annotation."""
        flows = []
        for m, d in self.graph.items():
            for dep in d.get("imports", []):
                flows.append({"from": m, "to": dep})
        return flows[:50]

    # ── 4. entry points ──────────────────────────────────────────────
    def _entry_points(self) -> List[Dict]:
        eps = []
        for rel, mod in self.parsed.items():
            if mod.get("has_main"):
                eps.append({"file": rel, "type": "if __name__ == '__main__'"})
            for func in mod.get("functions", []):
                if func["name"] in ("main", "cli", "run", "app"):
                    eps.append({"file": rel, "type": f"function:{func['name']}"})
            for cls in mod.get("classes", []):
                if cls["name"].lower() in ("app", "application", "cli"):
                    eps.append({"file": rel, "type": f"class:{cls['name']}"})
        # Also modules not imported by anyone
        for rel, d in self.graph.items():
            if not d.get("imported_by"):
                eps.append({"file": rel, "type": "leaf (not imported)"})
        return eps

    # ── 5. configuration handling ────────────────────────────────────
    def _config_handling(self) -> Dict:
        config_patterns = {
            "env_vars": [],
            "config_files": [],
            "arg_parsing": [],
        }
        for rel, mod in self.parsed.items():
            for imp in mod.get("imports", []):
                if imp["module"] in ("os.environ", "os", "dotenv", "python-dotenv"):
                    config_patterns["env_vars"].append(rel)
                if imp["module"] in ("configparser", "toml", "tomli", "yaml", "json"):
                    config_patterns["config_files"].append(rel)
                if imp["module"] in ("argparse", "click", "typer", "optparse", "sys"):
                    config_patterns["arg_parsing"].append(rel)
        # deduplicate
        for k in config_patterns:
            config_patterns[k] = sorted(set(config_patterns[k]))
        return config_patterns

    # ── 6. integrations ──────────────────────────────────────────────
    def _detect_integrations(self) -> Dict:
        db_libs    = {"sqlalchemy", "sqlite3", "pymongo", "psycopg2", "mysql",
                      "peewee", "tortoise", "databases", "asyncpg", "aiosqlite"}
        api_libs   = {"requests", "httpx", "aiohttp", "fastapi", "flask",
                      "django", "starlette", "bottle", "sanic", "tornado",
                      "urllib", "urllib3", "grpc"}
        found_db: List[str] = []
        found_api: List[str] = []
        for _m, mod in self.parsed.items():
            for imp in mod.get("imports", []):
                top = imp["module"].split(".")[0]
                if top in db_libs:
                    found_db.append(imp["module"])
                if top in api_libs:
                    found_api.append(imp["module"])
        return {
            "database": sorted(set(found_db)),
            "api_http": sorted(set(found_api)),
        }

    # ── 7. layers ────────────────────────────────────────────────────
    def _identify_layers(self) -> Dict[str, List[str]]:
        layers: Dict[str, List[str]] = {
            "presentation": [],
            "business_logic": [],
            "data_access": [],
            "utilities": [],
            "testing": [],
            "configuration": [],
        }
        for rel in self.graph:
            low = rel.lower()
            if any(k in low for k in ("view", "template", "ui", "render", "termui", "formatting")):
                layers["presentation"].append(rel)
            elif any(k in low for k in ("core", "command", "decorat", "shell_completion")):
                layers["business_logic"].append(rel)
            elif any(k in low for k in ("model", "db", "data", "store", "orm")):
                layers["data_access"].append(rel)
            elif any(k in low for k in ("test", "conftest", "fixture")):
                layers["testing"].append(rel)
            elif any(k in low for k in ("config", "setting", "env", "conf.py")):
                layers["configuration"].append(rel)
            else:
                layers["utilities"].append(rel)
        return {k: v for k, v in layers.items() if v}

    # ── display ──────────────────────────────────────────────────────
    def _display(self, a: Dict):
        # core components
        t = Table(title="Core Architectural Components", show_lines=True)
        t.add_column("Module", style="cyan", min_width=35)
        t.add_column("Defs",   style="green", justify="right")
        t.add_column("Classes", style="white")
        for cc in a["core_components"][:6]:
            t.add_row(cc["module"], str(cc["total_defs"]), ", ".join(cc["classes"][:3]))
        console.print(t)

        # patterns
        if a["design_patterns"]:
            tree = Tree("[bold]Design Patterns Detected")
            for pat, items in a["design_patterns"].items():
                branch = tree.add(f"[cyan]{pat}[/] ({len(items)})")
                for item in items[:5]:
                    branch.add(f"[dim]{item}[/]")
            console.print(tree)

        # integrations
        integ = a["integrations"]
        if integ.get("database") or integ.get("api_http"):
            console.print(Panel(
                f"Database: {', '.join(integ['database']) or 'None'}\n"
                f"API/HTTP: {', '.join(integ['api_http']) or 'None'}",
                title="Integrations", border_style="magenta",
            ))

        # entry points
        eps = a["entry_points"]
        if eps:
            t2 = Table(title="Entry Points", show_lines=True)
            t2.add_column("File", style="cyan")
            t2.add_column("Type", style="yellow")
            for ep in eps[:8]:
                t2.add_row(ep["file"], ep["type"])
            console.print(t2)


# ============================================================================
# PHASE 2 ORCHESTRATOR
# ============================================================================

class Phase2:
    """
    Orchestrates the full Phase 2 pipeline:
        1. Load Phase 1 results
        2. AST-parse source files           (CodeParser)
        3. Build dependency graph            (DependencyGraphBuilder)  – Prompt 2.1
        4. Analyse dependency graph          (DependencyGraphBuilder)  – Prompt 2.1
        5. Compute complexity metrics        (ComplexityAnalyzer)
        6. Architecture understanding        (ArchitectureAnalyzer)    – Prompt 2.2
        7. Merge & save enriched analysis_results.json
    """

    def __init__(self):
        if not ANALYSIS_FILE.exists():
            raise FileNotFoundError(
                f"Phase 1 output not found: {ANALYSIS_FILE}\n"
                "Run phase1.py first."
            )
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            self.phase1 = json.load(f)

        self.repo_path = Path(self.phase1["repo_path"])
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Cloned repo not found at {self.repo_path}")

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        console.print(Panel.fit(
            "[bold green]Code-to-Doc  ·  Phase 2[/]\n"
            "Code Understanding & Dependency Mapping",
            border_style="green",
        ))

        # Collect source files (source + tests from Phase 1 inventory)
        inv = self.phase1["file_inventory"]["inventory"]
        all_source = inv.get("source", []) + inv.get("tests", [])

        # Step 1 – Parse
        parser = CodeParser(self.repo_path, all_source)
        parsed = parser.parse_all()

        # Step 2 – Dependency graph (Prompt 2.1)
        builder = DependencyGraphBuilder(self.repo_path, all_source, parsed)
        dep_graph = builder.build()

        # Step 3 – Analyse the graph (Prompt 2.1)
        graph_analysis = builder.analyse_graph()

        # Step 4 – Complexity metrics
        complexity = ComplexityAnalyzer(parsed, all_source).compute()

        # Step 5 – Architecture understanding (Prompt 2.2)
        architecture = ArchitectureAnalyzer(self.repo_path, parsed, dep_graph).analyse()

        # Step 6 – Merge into Phase 1 results and save
        self.phase1["phase"] = 2
        self.phase1["phase2"] = {
            "parsed_modules":     parsed,
            "dependency_graph":   dep_graph,
            "graph_analysis":     graph_analysis,
            "complexity_metrics": complexity,
            "architecture":       architecture,
            "completed_at":       datetime.now().isoformat(),
        }

        self._save(self.phase1)

        console.print(Panel.fit(
            f"[bold green]✓ Phase 2 complete[/]\n"
            f"Results saved → [cyan]{ANALYSIS_FILE}[/]",
            border_style="green",
        ))
        return self.phase1

    @staticmethod
    def _save(data: Dict):
        ANALYSIS_FILE.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Saved → {ANALYSIS_FILE}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    Phase2().run()
