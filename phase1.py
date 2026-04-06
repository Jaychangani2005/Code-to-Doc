"""
Code-to-Doc: Phase 1 – Repository Setup & Code Scanning
========================================================
Prompt 1.1 – Repository Metadata Extraction
Prompt 1.2 – File Categorization

Steps:
  1. Accept GitHub URL from user
  2. Clone repository locally (gitpython)
  3. Extract structured metadata (name, language, LOC, entry points, purpose)
  4. Categorize every file (source / config / tests / docs / build / third-party)
  5. Filter noise (venv, node_modules, .git, binaries, dist)
  6. Save full inventory to output/analysis_results.json

Output : output/analysis_results.json
"""

import os
import ast
import json
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

import chardet
import git
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────
load_dotenv()
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
CLONE_DIR   = BASE_DIR / "cloned_repo"
OUTPUT_FILE = OUTPUT_DIR / "analysis_results.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Constants – language / category mappings
# ─────────────────────────────────────────────

# Map file extension → programming language
LANGUAGE_MAP: Dict[str, str] = {
    ".py":    "Python",
    ".js":    "JavaScript",
    ".ts":    "TypeScript",
    ".jsx":   "JavaScript (React)",
    ".tsx":   "TypeScript (React)",
    ".java":  "Java",
    ".kt":    "Kotlin",
    ".go":    "Go",
    ".rs":    "Rust",
    ".cpp":   "C++",
    ".c":     "C",
    ".cs":    "C#",
    ".rb":    "Ruby",
    ".php":   "PHP",
    ".swift": "Swift",
    ".scala": "Scala",
    ".r":     "R",
    ".m":     "MATLAB/Objective-C",
    ".sh":    "Shell",
    ".bash":  "Shell",
    ".ps1":   "PowerShell",
}

# Map extension → category
CATEGORY_MAP: Dict[str, str] = {
    # Source code
    ".py": "source", ".js": "source", ".ts": "source",
    ".jsx": "source", ".tsx": "source", ".java": "source",
    ".kt": "source", ".go": "source", ".rs": "source",
    ".cpp": "source", ".c": "source", ".cs": "source",
    ".rb": "source", ".php": "source", ".swift": "source",
    ".scala": "source", ".r": "source", ".m": "source",
    ".sh": "source", ".bash": "source", ".ps1": "source",
    # Configuration
    ".json": "config", ".yaml": "config", ".yml": "config",
    ".toml": "config", ".ini": "config", ".cfg": "config",
    ".env": "config", ".properties": "config", ".xml": "config",
    ".conf": "config", ".config": "config",
    # Documentation
    ".md": "docs", ".rst": "docs", ".txt": "docs",
    ".pdf": "docs", ".adoc": "docs", ".wiki": "docs",
    # Build artifacts / scripts
    ".gradle": "build", ".maven": "build", ".makefile": "build",
    # Web / templates
    ".html": "source", ".css": "source", ".scss": "source",
    ".jinja": "source", ".j2": "source",
}

# Entry-point filenames to look for
ENTRY_POINTS = {
    "main.py", "app.py", "run.py", "server.py", "manage.py",
    "cli.py", "index.js", "index.ts", "main.js", "app.js",
    "main.go", "main.rs", "Main.java", "Application.java",
    "main.c", "main.cpp", "Program.cs",
}

# Directories to skip unconditionally
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", ".env",
    "__pycache__", ".pytest_cache", "dist", "build", "target",
    ".tox", "site-packages", ".eggs", "*.egg-info",
    ".idea", ".vscode", ".DS_Store", "htmlcov", ".mypy_cache",
}

# Extensions that are certainly binary / non-text
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".class", ".jar", ".war", ".pyc", ".pyo",
    ".db", ".sqlite", ".sqlite3",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".lock",          # package-lock etc — not useful for docs
}


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def is_skip_dir(path: Path) -> bool:
    """Return True if any component of *path* is in the skip-list."""
    for part in path.parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    return False


def is_binary_file(path: Path) -> bool:
    """Return True for known binary extensions OR when chardet fails."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:          # null bytes → binary
            return True
        result = chardet.detect(raw[:8192])
        return result["encoding"] is None
    except Exception:
        return True


def count_lines(path: Path) -> int:
    """Count non-empty lines in a text file; return 0 on error."""
    try:
        raw   = path.read_bytes()
        det   = chardet.detect(raw[:8192])
        enc   = det.get("encoding") or "utf-8"
        lines = raw.decode(enc, errors="replace").splitlines()
        return sum(1 for ln in lines if ln.strip())
    except Exception:
        return 0


def detect_language(path: Path) -> str:
    """Return the primary language name for a given file path."""
    return LANGUAGE_MAP.get(path.suffix.lower(), "Unknown")


def get_category(path: Path, is_test: bool) -> str:
    """Return the category string for a file."""
    if is_test:
        return "tests"
    return CATEGORY_MAP.get(path.suffix.lower(), "other")


def is_test_file(path: Path) -> bool:
    """Heuristic: file lives in a tests/ folder or name starts with test_."""
    lower = path.name.lower()
    if lower.startswith("test_") or lower.endswith("_test.py"):
        return True
    for part in path.parts:
        if part.lower() in ("tests", "test", "spec", "__tests__"):
            return True
    return False


def read_readme(repo_path: Path) -> str:
    """Return the text of the first README found (any extension)."""
    for name in ("README.md", "README.rst", "README.txt", "Readme.md", "readme.md"):
        p = repo_path / name
        if p.exists():
            try:
                raw = p.read_bytes()
                enc = chardet.detect(raw[:8192]).get("encoding") or "utf-8"
                return raw.decode(enc, errors="replace")[:3000]
            except Exception:
                pass
    return ""


# ============================================================================
# PROMPT 1.1 – REPOSITORY METADATA EXTRACTION
# ============================================================================

class RepoMetadataExtractor:
    """
    Extracts structured metadata from a locally cloned repository.

    Covers:
        - Project name
        - Primary programming language
        - Total source files & LOC
        - High-level purpose from README
        - Entry point detection
        - Repo git history summary
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    # ------------------------------------------------------------------
    def extract(self) -> Dict:
        """Run all metadata extraction steps and return a unified dict."""
        console.print("\n[bold cyan]── Prompt 1.1: Repository Metadata Extraction ──[/]")

        name         = self._get_project_name()
        readme_text  = read_readme(self.repo_path)
        purpose      = self._infer_purpose(readme_text)
        lang_stats   = self._count_languages()
        primary_lang = max(lang_stats, key=lang_stats.get) if lang_stats else "Unknown"
        entry_points = self._find_entry_points()
        git_meta     = self._get_git_metadata()
        total_files, total_loc = self._count_files_and_loc()

        metadata = {
            "project_name":       name,
            "primary_language":   primary_lang,
            "language_breakdown": lang_stats,
            "total_source_files": total_files,
            "total_lines_of_code": total_loc,
            "project_purpose":    purpose,
            "entry_points":       entry_points,
            "git_metadata":       git_meta,
            "readme_preview":     readme_text[:500] if readme_text else "No README found",
            "extracted_at":       datetime.now().isoformat(),
        }

        self._display_metadata(metadata)
        return metadata

    # ------------------------------------------------------------------
    def _get_project_name(self) -> str:
        return self.repo_path.name

    def _infer_purpose(self, readme: str) -> str:
        """Extract the first meaningful paragraph from the README."""
        if not readme:
            return "Purpose not determined (no README found)"
        lines = [ln.strip() for ln in readme.splitlines() if ln.strip()]
        # Skip heading lines that are just the project name
        for line in lines:
            if line.startswith("#"):
                continue
            if len(line) > 30:
                return line[:300]
        return lines[0][:300] if lines else "Unable to determine purpose"

    def _count_languages(self) -> Dict[str, int]:
        """Count source-code files per language (skipping noise dirs)."""
        counts: Counter = Counter()
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if is_skip_dir(path.relative_to(self.repo_path)):
                continue
            lang = detect_language(path)
            if lang != "Unknown":
                counts[lang] += 1
        return dict(counts.most_common())

    def _find_entry_points(self) -> List[str]:
        """Return relative paths of recognised entry-point files."""
        found = []
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if is_skip_dir(path.relative_to(self.repo_path)):
                continue
            if path.name in ENTRY_POINTS:
                found.append(str(path.relative_to(self.repo_path)))
        return found

    def _get_git_metadata(self) -> Dict:
        """Pull basic git info (branch, commit count, last commit)."""
        try:
            repo         = git.Repo(self.repo_path)
            active_branch = repo.active_branch.name
            commit_count  = sum(1 for _ in repo.iter_commits())
            last_commit   = repo.head.commit
            return {
                "active_branch": active_branch,
                "total_commits": commit_count,
                "last_commit":   {
                    "hash":    str(last_commit.hexsha[:8]),
                    "message": last_commit.message.strip()[:100],
                    "author":  str(last_commit.author),
                    "date":    datetime.fromtimestamp(last_commit.committed_date).isoformat(),
                },
            }
        except Exception as exc:
            logger.warning(f"Git metadata unavailable: {exc}")
            return {"error": str(exc)}

    def _count_files_and_loc(self) -> Tuple[int, int]:
        """Return (file_count, total_loc) for all non-binary, non-skip files."""
        file_count = 0
        total_loc  = 0
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.repo_path)
            if is_skip_dir(rel):
                continue
            if is_binary_file(path):
                continue
            if path.suffix.lower() not in LANGUAGE_MAP:
                continue
            file_count += 1
            total_loc  += count_lines(path)
        return file_count, total_loc

    def _display_metadata(self, m: Dict) -> None:
        t = Table(title="Repository Metadata", show_lines=True)
        t.add_column("Field",  style="cyan",  min_width=25)
        t.add_column("Value",  style="white")
        t.add_row("Project Name",          m["project_name"])
        t.add_row("Primary Language",      m["primary_language"])
        t.add_row("Total Source Files",    str(m["total_source_files"]))
        t.add_row("Total Lines of Code",   f"{m['total_lines_of_code']:,}")
        t.add_row("Project Purpose",       m["project_purpose"][:80] + "…" if len(m["project_purpose"]) > 80 else m["project_purpose"])
        t.add_row("Entry Points",          ", ".join(m["entry_points"]) or "None detected")
        gm = m["git_metadata"]
        if "error" not in gm:
            t.add_row("Git Branch",        gm.get("active_branch", "N/A"))
            t.add_row("Total Commits",     str(gm.get("total_commits", "N/A")))
        console.print(t)


# ============================================================================
# PROMPT 1.2 – FILE CATEGORIZATION
# ============================================================================

class FileCategorizer:
    """
    Walks the repo and produces a clean, categorized inventory of every file.

    Categories:
        source         – documentable source code
        config         – configuration / environment files
        tests          – test files
        docs           – documentation files
        build          – build scripts / artifacts
        third_party    – vendored / generated files
        filtered_out   – noise (venv, .git, binaries, etc.)
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    # ------------------------------------------------------------------
    def categorize(self) -> Dict:
        """Walk the repo, classify every file, return full inventory."""
        console.print("\n[bold cyan]── Prompt 1.2: File Categorization ──[/]")

        inventory: Dict[str, List[Dict]] = {
            "source":      [],
            "config":      [],
            "tests":       [],
            "docs":        [],
            "build":       [],
            "third_party": [],
            "other":       [],
        }
        filtered_out: List[Dict] = []

        all_files = sorted(self.repo_path.rglob("*"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Scanning files…", total=len(all_files))

            for path in all_files:
                progress.advance(task)
                if not path.is_file():
                    continue

                rel = path.relative_to(self.repo_path)

                # ── Noise filter ─────────────────────────────────────
                reason = self._filter_reason(rel, path)
                if reason:
                    filtered_out.append({
                        "path":   str(rel),
                        "reason": reason,
                    })
                    continue

                # ── Classify ─────────────────────────────────────────
                is_test = is_test_file(rel)
                category = get_category(path, is_test)
                lang     = detect_language(path)
                loc      = count_lines(path)

                entry: Dict = {
                    "path":      str(rel),
                    "name":      path.name,
                    "extension": path.suffix.lower(),
                    "language":  lang,
                    "lines":     loc,
                    "size_kb":   round(path.stat().st_size / 1024, 2),
                }

                bucket = inventory.get(category, inventory["other"])
                bucket.append(entry)

        result = {
            "inventory":       inventory,
            "filtered_out":    filtered_out,
            "summary": {
                "source_files":      len(inventory["source"]),
                "config_files":      len(inventory["config"]),
                "test_files":        len(inventory["tests"]),
                "doc_files":         len(inventory["docs"]),
                "build_files":       len(inventory["build"]),
                "third_party_files": len(inventory["third_party"]),
                "other_files":       len(inventory["other"]),
                "filtered_out":      len(filtered_out),
            },
        }

        self._display_summary(result["summary"])
        return result

    # ------------------------------------------------------------------
    def _filter_reason(self, rel: Path, abs_path: Path) -> Optional[str]:
        """Return a reason string if this file should be filtered, else None."""
        if is_skip_dir(rel):
            return "in excluded directory"
        if is_binary_file(abs_path):
            return "binary file"
        # Third-party / vendored heuristics
        parts_lower = [p.lower() for p in rel.parts]
        if any(p in ("vendor", "vendors", "third_party", "thirdparty", "extern") for p in parts_lower):
            return "third-party / vendored directory"
        return None

    def _display_summary(self, summary: Dict) -> None:
        t = Table(title="File Inventory Summary", show_lines=True)
        t.add_column("Category",      style="cyan",  min_width=22)
        t.add_column("File Count",    style="green", justify="right")
        rows = [
            ("Source Code",      summary["source_files"]),
            ("Configuration",    summary["config_files"]),
            ("Tests",            summary["test_files"]),
            ("Documentation",    summary["doc_files"]),
            ("Build",            summary["build_files"]),
            ("Third-party",      summary["third_party_files"]),
            ("Other",            summary["other_files"]),
            ("Filtered Out",     summary["filtered_out"]),
        ]
        for label, count in rows:
            t.add_row(label, str(count))
        console.print(t)


# ============================================================================
# PHASE 1 ORCHESTRATOR
# ============================================================================

class Phase1:
    """
    Orchestrates the full Phase 1 pipeline:
        1. Validate & clone the GitHub URL
        2. Extract repository metadata  (Prompt 1.1)
        3. Categorize all files         (Prompt 1.2)
        4. Merge results and save to output/analysis_results.json
    """

    def __init__(self, github_url: str, github_token: Optional[str] = None):
        self.github_url   = github_url.strip()
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo_path: Optional[Path] = None

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        console.print(Panel.fit(
            "[bold green]Code-to-Doc  ·  Phase 1[/]\n"
            "Repository Setup & Code Scanning",
            border_style="green",
        ))

        # Step 1 – clone
        self.repo_path = self._clone()

        # Step 2 – metadata  (Prompt 1.1)
        metadata = RepoMetadataExtractor(self.repo_path).extract()

        # Step 3 – categorize  (Prompt 1.2)
        file_data = FileCategorizer(self.repo_path).categorize()

        # Step 4 – merge & save
        results = {
            "phase":          1,
            "github_url":     self.github_url,
            "repo_path":      str(self.repo_path),
            "metadata":       metadata,
            "file_inventory": file_data,
        }
        self._save(results)

        console.print(Panel.fit(
            f"[bold green]✓ Phase 1 complete[/]\n"
            f"Results saved → [cyan]{OUTPUT_FILE}[/]",
            border_style="green",
        ))
        return results

    # ------------------------------------------------------------------
    def _clone(self) -> Path:
        """Clone (or reuse) the repository; return the local path.

        Resiliency features:
            - retries transient network failures (e.g., "Empty reply from server")
            - uses HTTP/1.1 for better compatibility in unstable networks
            - falls back from token URL to plain HTTPS URL when token clone fails
        """
        if not self._valid_url(self.github_url):
            raise ValueError(f"Invalid GitHub URL: {self.github_url!r}")

        repo_name = self.github_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        dest = CLONE_DIR / repo_name

        if dest.exists():
            console.print(f"[yellow]⚠ Repo already exists at {dest} – reusing.[/]")
            console.print("[dim]  (Delete the folder to re-clone)[/]")
            return dest

        # Candidate URLs: token URL first, then plain URL fallback.
        clone_candidates: List[Tuple[str, str]] = []
        if (
            self.github_token
            and "github.com" in self.github_url
            and self.github_url.startswith("https://")
        ):
            token_url = self.github_url.replace(
                "https://", f"https://{self.github_token}@"
            )
            clone_candidates.append(("token", token_url))
        clone_candidates.append(("https", self.github_url))

        max_attempts = 3
        last_error: Optional[str] = None

        def _clear_partial_clone() -> None:
            if dest.exists():
                try:
                    shutil.rmtree(dest)
                except Exception:
                    pass

        console.print(f"\n[cyan]Cloning[/] {self.github_url} …")

        for mode, clone_url in clone_candidates:
            for attempt in range(1, max_attempts + 1):
                try:
                    _clear_partial_clone()
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,
                    ) as progress:
                        progress.add_task(
                            f"Cloning repository ({mode}, attempt {attempt}/{max_attempts})…",
                            total=None,
                        )
                        git.Repo.clone_from(
                            clone_url,
                            dest,
                            depth=50,
                            env={"GIT_HTTP_VERSION": "HTTP/1.1"},
                        )

                    console.print(f"[green]✓ Cloned to {dest}[/]")
                    return dest

                except git.exc.GitCommandError as e:
                    err_text = str(e)
                    last_error = err_text
                    lowered = err_text.lower()

                    auth_error = any(
                        token in lowered
                        for token in (
                            "authentication failed",
                            "access denied",
                            "repository not found",
                            "could not read username",
                            "invalid username or password",
                        )
                    )

                    network_error = any(
                        token in lowered
                        for token in (
                            "empty reply from server",
                            "connection reset",
                            "failed to connect",
                            "operation timed out",
                            "timeout",
                            "tls",
                            "http/2 stream",
                        )
                    )

                    if mode == "token" and auth_error:
                        console.print(
                            "[yellow]⚠ Token-based clone failed (auth). "
                            "Trying plain HTTPS URL…[/yellow]"
                        )
                        break

                    if attempt < max_attempts and network_error:
                        wait_s = min(2 ** attempt, 8)
                        console.print(
                            f"[yellow]⚠ Clone attempt {attempt}/{max_attempts} "
                            f"failed due to network issue. Retrying in {wait_s}s…[/yellow]"
                        )
                        time.sleep(wait_s)
                        continue

                    # Non-network error or retries exhausted for this mode.
                    if attempt >= max_attempts:
                        console.print(
                            f"[yellow]⚠ Clone failed after {max_attempts} attempts "
                            f"using {mode} URL.[/yellow]"
                        )
                        break

                except Exception as e:
                    last_error = str(e)
                    if attempt < max_attempts:
                        wait_s = min(2 ** attempt, 8)
                        console.print(
                            f"[yellow]⚠ Clone attempt {attempt}/{max_attempts} failed. "
                            f"Retrying in {wait_s}s…[/yellow]"
                        )
                        time.sleep(wait_s)
                        continue
                    console.print(
                        f"[yellow]⚠ Clone failed after {max_attempts} attempts "
                        f"using {mode} URL.[/yellow]"
                    )
                    break

        _clear_partial_clone()
        raise RuntimeError(
            "Repository clone failed after retries and fallback. "
            "Please check internet connectivity, GitHub availability, and repository access.\n"
            f"Last error: {last_error or 'unknown error'}"
        )

    @staticmethod
    def _valid_url(url: str) -> bool:
        return url.startswith("https://github.com/") or url.startswith("git@github.com:")

    @staticmethod
    def _save(data: Dict) -> None:
        """Persist results as pretty-printed JSON."""
        OUTPUT_FILE.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Saved → {OUTPUT_FILE}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    # Accept URL from CLI or prompt interactively
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = console.input(
            "\n[bold cyan]Enter GitHub repository URL[/] "
            "(e.g. https://github.com/owner/repo): "
        ).strip()

    phase1 = Phase1(github_url=url)
    phase1.run()
