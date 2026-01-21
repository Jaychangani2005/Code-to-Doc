"""
Code-to-Doc: Legacy Code Documentation Agent
Phase 1 & Phase 2 Implementation
"""

import os
import json
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging
from datetime import datetime
import chardet
import importlib
# from gitpython import git
from git import Repo
from git.exc import InvalidGitRepositoryError

from git import Repo
from git.exc import InvalidGitRepositoryError
import ast
import re
from collections import defaultdict
import inspect
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _rmtree_force(path: Path) -> None:
    """Remove a directory tree, handling Windows read-only files."""
    if not path.exists():
        return

    def _on_rm_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_on_rm_error)


def _load_magic_module():
    """Best-effort loader for python-magic / python-magic-bin.

    On some Windows installs, the distribution provides a namespace package
    ("magic") and the implementation lives in "magic.magic".
    """
    try:
        import magic as _magic  # type: ignore
        if hasattr(_magic, "Magic"):
            return _magic
        return importlib.import_module("magic.magic")
    except Exception:
        return None


_MAGIC = _load_magic_module()


# ============================================================================
# PHASE 1: FOUNDATION SETUP
# ============================================================================

class RepositoryManager:
    """Handles repository cloning and initial setup."""

    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.java': 'java',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c_header',
    }

    IGNORE_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__',
                   '.env', 'dist', 'build', '.pytest_cache', '.idea'}

    IGNORE_FILES = {'.gitignore', '.env', '.DS_Store', 'package-lock.json'}

    MAX_REPO_SIZE_GB = 2

    def __init__(self, max_size_gb: int = 2):
        """Initialize repository manager."""
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.max_size_gb = max_size_gb
        self.clone_dir = Path(__file__).resolve().parent / 'claude_cloned'

    def validate_github_url(self, url: str) -> bool:
        """Validate GitHub URL format."""
        github_pattern = r'^(https://|git@)github\.com[:/][\w\-\.]+/[\w\-\.]+\.git?$'
        return bool(re.match(github_pattern, url))

    def clone_repository(self, github_url: str, github_token: Optional[str] = None) -> Path:
        """
        Safely clone repository from GitHub URL.

        Args:
            github_url: HTTPS or SSH GitHub URL
            github_token: Optional GitHub token for private repos

        Returns:
            Path to cloned repository

        Raises:
            ValueError: If URL is invalid
            OSError: If clone fails
        """
        if not self.validate_github_url(github_url):
            raise ValueError(f"Invalid GitHub URL: {github_url}")

        logger.info(f"Starting repository clone from {github_url}")

        # Prepare clone URL with token if provided
        if github_token and 'https' in github_url:
            clone_url = github_url.replace(
                'https://', f'https://{github_token}@')
        else:
            clone_url = github_url

        try:
            # Clone repository into a stable local folder
            if self.clone_dir.exists():
                _rmtree_force(self.clone_dir)

            repo_path = self.clone_dir
            Repo.clone_from(clone_url, str(repo_path), depth=1)
            logger.info(f"Repository cloned successfully to {repo_path}")

            # Verify size
            repo_size = self._calculate_dir_size(repo_path)
            if repo_size > self.max_size_bytes:
                raise OSError(
                    f"Repository size ({repo_size / 1024 / 1024:.2f}MB) "
                    f"exceeds limit ({self.max_size_gb}GB)"
                )

            return repo_path

        except Exception as e:
            logger.error(f"Clone failed: {str(e)}")
            self._cleanup()
            raise

    def _calculate_dir_size(self, path: Path) -> int:
        """Calculate total directory size in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def _cleanup(self):
        """Clean up clone directory on failure."""
        if self.clone_dir.exists():
            _rmtree_force(self.clone_dir)
            logger.info("Clone directory cleaned up")

    def get_repo_metadata(self, repo_path: Path) -> Dict:
        """Extract repository metadata."""
        try:
            repo = Repo(repo_path)
            return {
                'name': repo_path.name,
                'url': repo.remotes.origin.url if repo.remotes else 'N/A',
                'branch': repo.active_branch.name,
                'commit': repo.head.commit.hexsha[:8],
                'last_commit': repo.head.commit.committed_datetime.isoformat(),
                'description': self._extract_description(repo_path),
            }
        except Exception as e:
            logger.warning(f"Could not extract full metadata: {e}")
            return {'name': repo_path.name, 'error': str(e)}

    def _extract_description(self, repo_path: Path) -> str:
        """Extract description from README if available."""
        readme_files = ['README.md', 'README.rst', 'README.txt']
        for readme in readme_files:
            readme_path = repo_path / readme
            if readme_path.exists():
                try:
                    with open(readme_path, 'r') as f:
                        return f.read()[:500]  # First 500 chars
                except:
                    pass
        return "No description available"


# ============================================================================
# STEP 2: CODE SCANNING & FILE ANALYSIS
# ============================================================================

class CodeScanner:
    """Scans and categorizes all code files in repository."""

    def __init__(self, repo_path: Path):
        """Initialize code scanner."""
        self.repo_path = repo_path
        self.mime = _MAGIC.Magic(mime=True) if _MAGIC and hasattr(
            _MAGIC, "Magic") else None

    def scan_repository(self) -> Dict[str, List[Dict]]:
        """
        Scan repository and categorize all code files.

        Returns:
            Dictionary with language as key and list of files as value
        """
        logger.info(f"Starting code scan on {self.repo_path}")

        code_files = defaultdict(list)
        file_stats = {
            'total_files': 0,
            'code_files': 0,
            'binary_files': 0,
            'ignored_files': 0,
        }

        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            # Filter out ignored directories
            dirnames[:] = [d for d in dirnames
                           if d not in RepositoryManager.IGNORE_DIRS]

            for filename in filenames:
                file_stats['total_files'] += 1
                filepath = Path(dirpath) / filename

                # Skip ignored files
                if filename in RepositoryManager.IGNORE_FILES:
                    file_stats['ignored_files'] += 1
                    continue

                file_info = self._analyze_file(filepath)

                if file_info:
                    language = file_info['language']
                    code_files[language].append(file_info)
                    file_stats['code_files'] += 1
                else:
                    file_stats['binary_files'] += 1

        logger.info(
            f"Scan complete: {file_stats['code_files']} code files found")
        logger.info(f"File statistics: {file_stats}")

        return {
            'files': dict(code_files),
            'statistics': file_stats,
        }

    def _analyze_file(self, filepath: Path) -> Optional[Dict]:
        """
        Analyze individual file and extract metadata.

        Returns:
            Dictionary with file info or None if not a code file
        """
        try:
            # Check file extension
            extension = filepath.suffix.lower()
            if extension not in RepositoryManager.SUPPORTED_EXTENSIONS:
                return None

            # Detect encoding
            encoding = self._detect_encoding(filepath)
            if not encoding:
                return None

            # Check if file is binary
            if self._is_binary(filepath):
                return None

            return {
                'path': str(filepath.relative_to(self.repo_path)),
                'absolute_path': str(filepath),
                'language': RepositoryManager.SUPPORTED_EXTENSIONS[extension],
                'extension': extension,
                'encoding': encoding,
                'size': filepath.stat().st_size,
                'lines': self._count_lines(filepath),
            }

        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
            return None

    def _detect_encoding(self, filepath: Path) -> Optional[str]:
        """Detect file encoding."""
        try:
            with open(filepath, 'rb') as f:
                raw = f.read(10000)
                result = chardet.detect(raw)
                return result.get('encoding', 'utf-8')
        except:
            return None

    def _is_binary(self, filepath: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(512)
                return b'\x00' in chunk
        except:
            return True

    def _count_lines(self, filepath: Path) -> int:
        """Count lines in file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except:
            return 0


# ============================================================================
# PHASE 2: CODE UNDERSTANDING
# ============================================================================

class CodeParser:
    """Parses code to extract functions, classes, methods, and docstrings."""

    def __init__(self, repo_path: Path, code_files: Dict[str, List[Dict]]):
        """Initialize code parser."""
        self.repo_path = repo_path
        self.code_files = code_files
        self.parsed_modules = {}

    def parse_all_files(self) -> Dict:
        """Parse all Python files and extract code elements."""
        logger.info("Parsing code files for detailed extraction")

        if 'python' in self.code_files:
            self._parse_python_files()

        if 'java' in self.code_files:
            self._parse_java_files()

        logger.info(f"Parsed {len(self.parsed_modules)} modules")
        return self.parsed_modules

    def _parse_python_files(self):
        """Parse Python files using AST."""
        for file_info in self.code_files['python']:
            filepath = Path(file_info['absolute_path'])
            relative_path = file_info['path']

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                tree = ast.parse(content)
                module_data = {
                    'file': relative_path,
                    'docstring': ast.get_docstring(tree),
                    'imports': [],
                    'functions': [],
                    'classes': [],
                    'constants': [],
                    'module_level_code': self._has_module_level_code(tree)
                }

                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.FunctionDef):
                        module_data['functions'].append(
                            self._extract_function(node))
                    elif isinstance(node, ast.ClassDef):
                        module_data['classes'].append(
                            self._extract_class(node))
                    elif isinstance(node, ast.Assign):
                        constants = self._extract_constants(node)
                        module_data['constants'].extend(constants)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        module_data['imports'].append(
                            self._format_import(node))

                self.parsed_modules[relative_path] = module_data

            except Exception as e:
                logger.warning(f"Error parsing {filepath}: {e}")

    def _extract_function(self, node: ast.FunctionDef) -> Dict:
        """Extract function details from AST node."""
        return {
            'name': node.name,
            'lineno': node.lineno,
            'end_lineno': node.end_lineno if hasattr(node, 'end_lineno') else None,
            'docstring': ast.get_docstring(node),
            'signature': self._get_function_signature(node),
            'parameters': self._extract_parameters(node),
            'return_type': self._get_return_annotation(node),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'is_private': node.name.startswith('_'),
            'complexity': self._calculate_function_complexity(node)
        }

    def _extract_class(self, node: ast.ClassDef) -> Dict:
        """Extract class details from AST node."""
        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'lineno': item.lineno
                        })

        return {
            'name': node.name,
            'lineno': node.lineno,
            'end_lineno': node.end_lineno if hasattr(node, 'end_lineno') else None,
            'docstring': ast.get_docstring(node),
            'bases': [self._get_base_name(base) for base in node.bases],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'methods': methods,
            'attributes': attributes,
            'is_private': node.name.startswith('_')
        }

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict]:
        """Extract function parameters with types and defaults."""
        params = []
        args = node.args

        # Regular arguments
        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param = {
                'name': arg.arg,
                'type': self._get_annotation(arg.annotation),
                'default': None,
                'kind': 'positional'
            }

            default_idx = i - defaults_offset
            if default_idx >= 0:
                param['default'] = self._get_default_value(
                    args.defaults[default_idx])

            params.append(param)

        # *args
        if args.vararg:
            params.append({
                'name': f"*{args.vararg.arg}",
                'type': self._get_annotation(args.vararg.annotation),
                'default': None,
                'kind': 'var_positional'
            })

        # **kwargs
        if args.kwarg:
            params.append({
                'name': f"**{args.kwarg.arg}",
                'type': self._get_annotation(args.kwarg.annotation),
                'default': None,
                'kind': 'var_keyword'
            })

        return params

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Generate function signature string."""
        params = []
        args = node.args

        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {self._get_annotation(arg.annotation)}"

            default_idx = i - defaults_offset
            if default_idx >= 0:
                default_val = self._get_default_value(
                    args.defaults[default_idx])
                param_str += f" = {default_val}"

            params.append(param_str)

        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")

        signature = f"{node.name}({', '.join(params)})"

        if node.returns:
            signature += f" -> {self._get_annotation(node.returns)}"

        return signature

    def _get_annotation(self, annotation) -> str:
        """Extract type annotation as string."""
        if annotation is None:
            return 'Any'

        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif isinstance(annotation, ast.Subscript):
                return ast.unparse(annotation)
            elif isinstance(annotation, ast.Attribute):
                return ast.unparse(annotation)
            else:
                return ast.unparse(annotation)
        except:
            return 'Any'

    def _get_return_annotation(self, node: ast.FunctionDef) -> str:
        """Extract return type annotation."""
        return self._get_annotation(node.returns)

    def _get_default_value(self, node) -> str:
        """Extract default parameter value."""
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
                return ast.unparse(node)
            else:
                return ast.unparse(node)
        except:
            return 'None'

    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name."""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    return decorator.func.id
                return ast.unparse(decorator.func)
            else:
                return ast.unparse(decorator)
        except:
            return 'unknown'

    def _get_base_name(self, base) -> str:
        """Extract base class name."""
        try:
            if isinstance(base, ast.Name):
                return base.id
            else:
                return ast.unparse(base)
        except:
            return 'unknown'

    def _extract_constants(self, node: ast.Assign) -> List[Dict]:
        """Extract module-level constants."""
        constants = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Consider uppercase names as constants
                if target.id.isupper():
                    constants.append({
                        'name': target.id,
                        'lineno': node.lineno,
                        'value': self._get_constant_value(node.value)
                    })
        return constants

    def _get_constant_value(self, node) -> str:
        """Extract constant value as string."""
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            else:
                return ast.unparse(node)[:100]  # Limit length
        except:
            return 'N/A'

    def _format_import(self, node) -> str:
        """Format import statement."""
        try:
            return ast.unparse(node)
        except:
            return 'import unknown'

    def _has_module_level_code(self, tree: ast.AST) -> bool:
        """Check if module has executable code at module level."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Expr, ast.If, ast.For, ast.While)):
                return True
        return False

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _parse_java_files(self):
        """Basic Java parsing using regex patterns."""
        for file_info in self.code_files['java']:
            filepath = Path(file_info['absolute_path'])
            relative_path = file_info['path']

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                module_data = {
                    'file': relative_path,
                    'package': self._extract_java_package(content),
                    'imports': self._extract_java_imports(content),
                    'classes': self._extract_java_classes(content),
                    'interfaces': self._extract_java_interfaces(content)
                }

                self.parsed_modules[relative_path] = module_data

            except Exception as e:
                logger.warning(f"Error parsing Java file {filepath}: {e}")

    def _extract_java_package(self, content: str) -> Optional[str]:
        """Extract Java package declaration."""
        match = re.search(r'package\s+([a-zA-Z0-9_.]+)\s*;', content)
        return match.group(1) if match else None

    def _extract_java_imports(self, content: str) -> List[str]:
        """Extract Java import statements."""
        pattern = r'import\s+(?:static\s+)?([a-zA-Z0-9_.]+)\s*;'
        return re.findall(pattern, content)

    def _extract_java_classes(self, content: str) -> List[Dict]:
        """Extract Java class declarations."""
        classes = []
        pattern = r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)'
        for match in re.finditer(pattern, content):
            classes.append({
                'name': match.group(1),
                'type': 'class'
            })
        return classes

    def _extract_java_interfaces(self, content: str) -> List[Dict]:
        """Extract Java interface declarations."""
        interfaces = []
        pattern = r'(?:public\s+)?interface\s+(\w+)'
        for match in re.finditer(pattern, content):
            interfaces.append({
                'name': match.group(1),
                'type': 'interface'
            })
        return interfaces


class DependencyGraphBuilder:
    """Builds dependency graph for code files."""

    def __init__(self, repo_path: Path, code_files: Dict[str, List[Dict]]):
        """Initialize dependency graph builder."""
        self.repo_path = repo_path
        self.code_files = code_files
        self.dependency_graph = defaultdict(lambda: {
            'imports': [],
            'imported_by': [],
            'external_deps': [],
        })

    def build_dependency_graph(self) -> Dict:
        """
        Build complete dependency graph for repository.

        Returns:
            Dictionary representing dependency relationships
        """
        logger.info("Building dependency graph")

        # Process Python files
        if 'python' in self.code_files:
            self._process_python_files()

        # Process Java files
        if 'java' in self.code_files:
            self._process_java_files()

        logger.info(f"Dependency graph built with "
                    f"{len(self.dependency_graph)} modules")

        return dict(self.dependency_graph)

    def _process_python_files(self):
        """Extract dependencies from Python files."""
        for file_info in self.code_files['python']:
            filepath = Path(file_info['absolute_path'])
            relative_path = file_info['path']

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = self._extract_python_imports(tree)
                external, internal = self._classify_imports(imports)

                self.dependency_graph[relative_path]['imports'] = internal
                self.dependency_graph[relative_path]['external_deps'] = external

                logger.debug(f"Processed {relative_path}: "
                             f"{len(internal)} internal, {len(external)} external")

            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")

    def _extract_python_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return list(set(imports))

    def _classify_imports(self, imports: List[str]) -> Tuple[List[str], List[str]]:
        """Classify imports as external or internal."""
        internal = []
        external = []

        # Get list of internal modules
        internal_modules = {
            str(f['path']).replace('.py', '').replace('/', '.')
            for f in self.code_files.get('python', [])
        }

        for imp in imports:
            base_module = imp.split('.')[0]
            if base_module in internal_modules or any(
                base_module in m for m in internal_modules
            ):
                internal.append(imp)
            else:
                external.append(imp)

        return external, internal

    def _process_java_files(self):
        """Extract dependencies from Java files."""
        for file_info in self.code_files['java']:
            filepath = Path(file_info['absolute_path'])
            relative_path = file_info['path']

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                imports = self._extract_java_imports(content)
                external, internal = self._classify_imports(imports)

                self.dependency_graph[relative_path]['imports'] = internal
                self.dependency_graph[relative_path]['external_deps'] = external

            except Exception as e:
                logger.warning(f"Error processing {filepath}: {e}")

    def _extract_java_imports(self, content: str) -> List[str]:
        """Extract import statements from Java code."""
        pattern = r'import\s+(?:static\s+)?([a-zA-Z0-9_.]+)\s*;'
        return re.findall(pattern, content)

    def calculate_complexity_metrics(self) -> Dict:
        """Calculate code complexity metrics."""
        metrics = {
            'total_files': sum(len(f) for f in self.code_files.values()),
            'files_by_language': {
                lang: len(files) for lang, files in self.code_files.items()
            },
            'total_lines': sum(
                f['lines'] for files in self.code_files.values()
                for f in files
            ),
            'average_file_size': 0,
            'circular_dependencies': self._detect_circular_deps(),
        }

        total_size = sum(
            f['size'] for files in self.code_files.values() for f in files
        )
        if metrics['total_files'] > 0:
            metrics['average_file_size'] = total_size / metrics['total_files']

        return metrics

    def _detect_circular_deps(self) -> List[List[str]]:
        """Detect circular dependencies."""
        # NOTE: self.dependency_graph is a defaultdict; referencing missing keys
        # during traversal would mutate it. Freeze a snapshot to keep DFS stable.
        graph = dict(self.dependency_graph)
        circular = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, {}).get('imports', []):
                # Only traverse known nodes to avoid pulling in external modules
                # and to prevent defaultdict mutation.
                if neighbor not in graph:
                    continue
                if neighbor not in visited:
                    dfs(neighbor, path[:])
                elif neighbor in rec_stack:
                    circular.append(path + [neighbor])

            rec_stack.remove(node)

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node, [])

        return circular


class ArchitectureAnalyzer:
    """Analyzes high-level system architecture."""

    def __init__(self, repo_path: Path, dependency_graph: Dict):
        """Initialize architecture analyzer."""
        self.repo_path = repo_path
        self.dependency_graph = dependency_graph

    def analyze_architecture(self) -> Dict:
        """
        Analyze architecture and identify core components.

        Returns:
            Dictionary with architecture analysis
        """
        logger.info("Analyzing system architecture")

        return {
            'core_modules': self._identify_core_modules(),
            'design_patterns': self._detect_design_patterns(),
            'entry_points': self._identify_entry_points(),
            'layers': self._identify_layers(),
        }

    def _identify_core_modules(self) -> List[str]:
        """Identify core/central modules."""
        # Modules with most dependencies
        importance = {}
        for module, deps in self.dependency_graph.items():
            importance[module] = len(deps['imported_by'])

        sorted_modules = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )
        return [m for m, _ in sorted_modules[:10]]

    def _identify_entry_points(self) -> List[str]:
        """Identify application entry points."""
        entry_points = []
        for module, deps in self.dependency_graph.items():
            # Modules not imported by others are potential entry points
            if not deps['imported_by']:
                entry_points.append(module)
        return entry_points

    def _detect_design_patterns(self) -> Dict:
        """Detect common design patterns."""
        patterns = {
            'factory': [],
            'singleton': [],
            'observer': [],
        }
        # Simple pattern detection based on naming
        for module in self.dependency_graph:
            if 'factory' in module.lower():
                patterns['factory'].append(module)
            if 'singleton' in module.lower():
                patterns['singleton'].append(module)

        return patterns

    def _identify_layers(self) -> Dict:
        """Identify architectural layers."""
        layers = {
            'presentation': [],
            'business': [],
            'data': [],
            'utility': [],
        }

        for module in self.dependency_graph:
            if any(x in module.lower() for x in ['view', 'ui', 'controller']):
                layers['presentation'].append(module)
            elif any(x in module.lower() for x in ['service', 'logic']):
                layers['business'].append(module)
            elif any(x in module.lower() for x in ['model', 'db', 'data']):
                layers['data'].append(module)
            else:
                layers['utility'].append(module)

        return layers


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

class CodeToDocOrchestrator:
    """Main orchestrator for Phases 1 and 2."""

    def __init__(self, github_url: str, github_token: Optional[str] = None):
        """Initialize orchestrator."""
        self.github_url = github_url
        self.github_token = github_token
        self.repo_manager = RepositoryManager()
        self.results = {}

    def execute_phase1(self) -> Dict:
        """Execute Phase 1: Foundation Setup."""
        logger.info("="*60)
        logger.info("PHASE 1: FOUNDATION SETUP")
        logger.info("="*60)

        # Step 1: Clone repository
        repo_path = self.repo_manager.clone_repository(
            self.github_url, self.github_token
        )

        # Extract metadata
        metadata = self.repo_manager.get_repo_metadata(repo_path)

        logger.info(f"Repository metadata: {json.dumps(metadata, indent=2)}")

        self.results['phase1'] = {
            'repo_path': str(repo_path),
            'metadata': metadata,
        }

        return repo_path

    def execute_phase2(self, repo_path: Path) -> Dict:
        """Execute Phase 2: Code Understanding."""
        logger.info("="*60)
        logger.info("PHASE 2: CODE UNDERSTANDING")
        logger.info("="*60)

        # Step 2: Scan repository
        scanner = CodeScanner(repo_path)
        scan_results = scanner.scan_repository()

        logger.info(
            f"Scan results: {json.dumps(scan_results['statistics'], indent=2)}")

        # Step 2.5: Parse code for detailed extraction
        code_parser = CodeParser(repo_path, scan_results['files'])
        parsed_code = code_parser.parse_all_files()

        logger.info(
            f"Parsed {len(parsed_code)} modules with detailed code information")

        # Step 3: Build dependency graph
        dep_builder = DependencyGraphBuilder(repo_path, scan_results['files'])
        dependency_graph = dep_builder.build_dependency_graph()
        complexity_metrics = dep_builder.calculate_complexity_metrics()

        logger.info(
            f"Complexity metrics: {json.dumps(complexity_metrics, indent=2)}")

        # Step 4: Analyze architecture
        arch_analyzer = ArchitectureAnalyzer(repo_path, dependency_graph)
        architecture = arch_analyzer.analyze_architecture()

        logger.info(
            f"Architecture analysis: {json.dumps(architecture, indent=2)}")

        self.results['phase2'] = {
            'code_files': scan_results['files'],
            'statistics': scan_results['statistics'],
            'parsed_modules': parsed_code,
            'dependency_graph': dependency_graph,
            'complexity_metrics': complexity_metrics,
            'architecture': architecture,
        }

        return self.results['phase2']

    def run_all(self) -> Dict:
        """Execute all phases."""
        try:
            repo_path = self.execute_phase1()
            self.execute_phase2(repo_path)

            logger.info("="*60)
            logger.info("ALL PHASES COMPLETED SUCCESSFULLY")
            logger.info("="*60)

            return self.results

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    orchestrator = CodeToDocOrchestrator(
        github_url="https://github.com/chavarera/s-tool.git",
        github_token=os.getenv("GITHUB_TOKEN")
    )

    results = orchestrator.run_all()

    # Save results
    output_path = Path("output/analysis_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
