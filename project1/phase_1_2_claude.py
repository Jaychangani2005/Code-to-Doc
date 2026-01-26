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
from dotenv import load_dotenv

# LangChain imports for HuggingFace integration
try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    # Create dummy classes for type hints
    StructuredTool = type('StructuredTool', (), {})
    HumanMessage = type('HumanMessage', (), {})
    SystemMessage = type('SystemMessage', (), {})
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain not fully available. Error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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


def _is_test_file(filepath: Path) -> bool:
    """Check if file is a test file."""
    test_keywords = {'test', 'spec', '_test.', 'tests/'}
    name_lower = filepath.name.lower()
    path_str = str(filepath).lower()
    return any(keyword in name_lower or keyword in path_str for keyword in test_keywords)


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
        self.backup_dir = Path(__file__).resolve().parent / 'backups'

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

        # Create backup of existing clone if it exists
        backup_path = self._create_backup()
        if backup_path:
            logger.info(f"Existing clone backed up to {backup_path}")

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

    def _create_backup(self) -> Optional[Path]:
        """Create timestamped backup of existing clone directory."""
        if not self.clone_dir.exists():
            return None
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"clone_backup_{timestamp}"
            shutil.copytree(self.clone_dir, backup_path)
            logger.info(f"Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return None

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

    def scan_repository(self, separate_tests: bool = True) -> Dict[str, List[Dict]]:
        """
        Scan repository and categorize all code files.

        Args:
            separate_tests: If True, separate test files into 'test_files' category

        Returns:
            Dictionary with language as key and list of files as value
        """
        logger.info(f"Starting code scan on {self.repo_path}")

        code_files = defaultdict(list)
        test_files = defaultdict(list)
        file_stats = {
            'total_files': 0,
            'code_files': 0,
            'test_files': 0,
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
                    # Separate test files if requested
                    if separate_tests and _is_test_file(filepath):
                        test_files[language].append(file_info)
                        file_stats['test_files'] += 1
                    else:
                        code_files[language].append(file_info)
                        file_stats['code_files'] += 1
                else:
                    file_stats['binary_files'] += 1

        logger.info(
            f"Scan complete: {file_stats['code_files']} code files, {file_stats['test_files']} test files found")
        logger.info(f"File statistics: {file_stats}")

        result = {
            'files': dict(code_files),
            'test_files': dict(test_files) if separate_tests else {},
            'statistics': file_stats,
        }
        
        return result

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
# ADJACENCY MATRIX BUILDER
# ============================================================================

class DependencyAdjacencyMatrixBuilder:
    """Builds adjacency matrix from dependency graph."""

    def __init__(self, dependency_graph: Dict):
        """Initialize matrix builder."""
        self.dependency_graph = dependency_graph

    def build_adjacency_matrix(self) -> Dict:
        """
        Convert dependency graph to adjacency matrix format.

        Returns:
            Dictionary with matrix representation
        """
        logger.info("Building dependency adjacency matrix")

        modules = sorted(list(self.dependency_graph.keys()))
        module_to_idx = {m: i for i, m in enumerate(modules)}
        n = len(modules)

        # Initialize matrix
        matrix = [[0 for _ in range(n)] for _ in range(n)]

        # Fill matrix: 1 if row module imports column module
        for module, deps in self.dependency_graph.items():
            row_idx = module_to_idx[module]
            for imported in deps.get('imports', []):
                if imported in module_to_idx:
                    col_idx = module_to_idx[imported]
                    matrix[row_idx][col_idx] = 1

        return {
            'modules': modules,
            'matrix': matrix,
            'total_dependencies': sum(sum(row) for row in matrix),
            'module_count': n
        }

    def generate_matrix_csv(self, matrix_data: Dict) -> str:
        """Generate CSV representation of adjacency matrix."""
        modules = matrix_data['modules']
        matrix = matrix_data['matrix']

        # Header
        csv_lines = [''] + modules
        csv_content = ','.join(csv_lines) + '\n'

        # Rows
        for i, module in enumerate(modules):
            row_values = [module] + [str(matrix[i][j]) for j in range(len(modules))]
            csv_content += ','.join(row_values) + '\n'

        return csv_content


# ============================================================================
# LLM-POWERED ARCHITECTURE ANALYZER
# ============================================================================

class LLMArchitectureAnalyzer:
    """Uses Hugging Face LLM to analyze code architecture in detail."""

    def __init__(self, hf_token: Optional[str] = None):
        """Initialize LLM analyzer."""
        self.hf_token = hf_token or os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
        self.repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.llm = None
        
        if LANGCHAIN_AVAILABLE and self.hf_token:
            try:
                endpoint = HuggingFaceEndpoint(
                    repo_id=self.repo_id,
                    huggingfacehub_api_token=self.hf_token,
                    max_new_tokens=512,
                    temperature=0.7,
                )
                self.llm = ChatHuggingFace(llm=endpoint)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")

    def analyze_architecture(self, parsed_data: Dict, dependency_graph: Dict) -> Dict:
        """
        Use LLM to analyze architecture patterns and provide insights.

        Args:
            parsed_data: Parsed code modules
            dependency_graph: Dependency relationships

        Returns:
            Dictionary with LLM analysis
        """
        logger.info("Running LLM-powered architecture analysis")

        if not self.llm:
            return {
                'error': 'LLM not available',
                'fallback': 'Using heuristic analysis only',
                'reason': 'LangChain or HuggingFace not properly configured'
            }

        try:
            # Prepare code summary for LLM
            code_summary = self._prepare_code_summary(parsed_data, dependency_graph)

            # Get LLM analysis
            analysis_prompt = self._create_analysis_prompt(code_summary)
            llm_response = self._query_llm(analysis_prompt)

            return {
                'llm_analysis': llm_response,
                'code_summary': code_summary,
                'analysis_timestamp': datetime.now().isoformat(),
                'model': self.repo_id
            }

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return {
                'error': str(e),
                'fallback': 'Using heuristic analysis only'
            }

    def _prepare_code_summary(self, parsed_data: Dict, dependency_graph: Dict) -> str:
        """Prepare code summary for LLM analysis."""
        summary = "CODE STRUCTURE SUMMARY:\n\n"

        # Module structure
        summary += "MODULES:\n"
        for module, data in list(parsed_data.items())[:10]:  # Limit to first 10
            summary += f"- {module}: {len(data.get('functions', []))} functions, {len(data.get('classes', []))} classes\n"

        # Dependencies
        summary += "\nKEY DEPENDENCIES:\n"
        top_deps = sorted(
            dependency_graph.items(),
            key=lambda x: len(x[1].get('imported_by', [])),
            reverse=True
        )[:5]
        for module, deps in top_deps:
            summary += f"- {module}: imported by {len(deps.get('imported_by', []))} modules\n"

        # External dependencies
        summary += "\nEXTERNAL DEPENDENCIES:\n"
        external_deps = set()
        for deps in dependency_graph.values():
            external_deps.update(deps.get('external_deps', []))
        for dep in list(external_deps)[:20]:
            summary += f"- {dep}\n"

        return summary

    def _create_analysis_prompt(self, code_summary: str) -> str:
        """Create LLM prompt for architecture analysis."""
        prompt = f"""Analyze the following code structure and provide architectural insights:

{code_summary}

Please identify and explain:
1. Core architectural components and their responsibilities
2. Main design patterns used
3. Data flow and information flow between modules
4. Potential architectural improvements
5. Risk areas and technical debt indicators

Keep response concise and actionable."""
        return prompt

    def _query_llm(self, prompt: str) -> str:
        """Query LLM using LangChain's ChatHuggingFace interface."""
        try:
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return f"Error querying LLM: {str(e)}"


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

        # Step 2: Scan repository (with test file separation)
        scanner = CodeScanner(repo_path)
        scan_results = scanner.scan_repository(separate_tests=True)

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

        # Step 3.5: Build adjacency matrix
        matrix_builder = DependencyAdjacencyMatrixBuilder(dependency_graph)
        adjacency_matrix = matrix_builder.build_adjacency_matrix()
        matrix_csv = matrix_builder.generate_matrix_csv(adjacency_matrix)

        logger.info(f"Built adjacency matrix: {adjacency_matrix['module_count']} modules")

        # Step 4: Analyze architecture
        arch_analyzer = ArchitectureAnalyzer(repo_path, dependency_graph)
        architecture = arch_analyzer.analyze_architecture()

        logger.info(
            f"Architecture analysis: {json.dumps(architecture, indent=2)}")

        # Step 4.5: LLM-powered architecture analysis
        llm_analyzer = LLMArchitectureAnalyzer()
        llm_analysis = llm_analyzer.analyze_architecture(parsed_code, dependency_graph)

        logger.info(f"LLM analysis completed")

        self.results['phase2'] = {
            'code_files': scan_results['files'],
            'test_files': scan_results.get('test_files', {}),
            'statistics': scan_results['statistics'],
            'parsed_modules': parsed_code,
            'dependency_graph': dependency_graph,
            'adjacency_matrix': adjacency_matrix,
            'matrix_csv': matrix_csv,
            'complexity_metrics': complexity_metrics,
            'architecture': architecture,
            'llm_analysis': llm_analysis,
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

            # Phase 3: Agent Setup & Documentation Generation
            logger.info("="*60)
            logger.info("PHASE 3: AGENT SETUP & DOCUMENTATION GENERATION")
            logger.info("="*60)
            
            agent = DocumentationAgent(
                analysis_results=self.results,
                hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            )
            
            documentation = agent.generate_documentation()
            self.results["phase3"] = documentation
            
            logger.info("="*60)
            logger.info("PHASE 3 COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            # Phase 4: Function-Level Docstring Generation
            logger.info("="*60)
            logger.info("PHASE 4: FUNCTION-LEVEL DOCSTRING GENERATION")
            logger.info("="*60)
            
            docstring_generator = FunctionDocstringGenerator(
                hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            )
            
            # Process all Python files
            phase2_files = self.results.get("phase2", {}).get("code_files", {}).get("python", [])
            phase4_results = {
                "docstrings_generated": [],
                "files_processed": 0,
                "total_functions_documented": 0,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            for file_info in phase2_files[:10]:  # Limit to first 10 files for speed
                file_path = file_info.get("absolute_path", "")
                if os.path.exists(file_path) and file_path.endswith(".py"):
                    logger.info(f"Generating docstrings for {file_path}")
                    result = docstring_generator.process_file(file_path)
                    
                    phase4_results["docstrings_generated"].append(result)
                    phase4_results["files_processed"] += 1
                    phase4_results["total_functions_documented"] += result.get("functions_documented", 0)
            
            self.results["phase4"] = phase4_results
            
            logger.info(f"Phase 4 completed: {phase4_results['total_functions_documented']} docstrings generated")
            
            # Phase 4B: Professional README Generation
            logger.info("="*60)
            logger.info("PHASE 4B: README GENERATION")
            logger.info("="*60)
            
            readme_generator = READMEGenerator(
                hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            )
            
            # Get clone path from Phase 1 results
            clone_path = self.results.get("phase1", {}).get("repo_path", self.github_url.split("/")[-1])
            readme_output_path = os.path.join(clone_path, "GENERATED_README.md")
            
            # Generate README using all phase results
            readme_content = readme_generator.generate_readme(
                phase1_results=self.results.get("phase1", {}),
                phase2_results=self.results.get("phase2", {}),
                phase3_results=self.results.get("phase3", {}),
                output_path=readme_output_path
            )
            
            self.results["phase4"]["readme_generated"] = {
                "path": readme_output_path,
                "status": "completed",
                "length": len(readme_content)
            }
            
            logger.info(f"README.md generated successfully at {readme_output_path}")
            logger.info("="*60)
            logger.info("ALL PHASES COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return self.results

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise


# ============================================================================
# PHASE 3: AGENT SETUP & CONFIGURATION
# ============================================================================

class DocumentationAgent:
    """
    LangChain agent configured to generate developer-friendly documentation
    by reasoning over the analyzed codebase structure.
    """

    def __init__(self, analysis_results: Dict[str, Any], hf_token: Optional[str] = None):
        """
        Initialize the documentation agent.
        
        Args:
            analysis_results: Complete Phase 1 & 2 analysis results
            hf_token: HuggingFace API token
        """
        self.analysis_results = analysis_results
        self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        
        # Extract key data structures
        self.dependency_graph = self._build_dependency_graph()
        self.modules_info = self._extract_modules_info()
        self.architecture = analysis_results.get("phase2", {}).get("architecture_analysis", {})
        
        # Initialize LLM if available
        self.llm = None
        if LANGCHAIN_AVAILABLE and self.hf_token:
            self._initialize_llm()
        
        logger.info("Documentation agent initialized")

    def _build_dependency_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a dependency graph representation from analysis results.
        
        Returns:
            Dependency graph in the format:
            {"module_a.py": {
                "imports": ["module_b", "module_c"],
                "imported_by": ["module_d"],
                "external_deps": ["requests", "numpy"]
            }}
        """
        graph = {}
        
        # Get parsed modules from Phase 2 (it's a dict, not a list)
        parsed_modules_dict = self.analysis_results.get("phase2", {}).get("parsed_modules", {})
        
        # Convert dict to list for easier processing
        for module_name, module_data in parsed_modules_dict.items():
            imports = module_data.get("imports", [])
            
            # Separate internal and external dependencies
            internal_imports = []
            external_deps = []
            
            for imp in imports:
                # Check if it's an internal import
                is_internal = False
                for other_module in parsed_modules_dict.keys():
                    module_base = other_module.replace("\\", ".").replace("/", ".").replace(".py", "")
                    if module_base in imp or imp.split()[0] == "from" and module_base in imp:
                        is_internal = True
                        break
                
                if is_internal:
                    internal_imports.append(imp)
                else:
                    external_deps.append(imp)
            
            graph[module_name] = {
                "imports": internal_imports,
                "imported_by": [],  # Will be populated in second pass
                "external_deps": external_deps,
                "functions": [f["name"] for f in module_data.get("functions", [])],
                "classes": [c["name"] for c in module_data.get("classes", [])]
            }
        
        # Second pass: populate imported_by relationships
        for module_name, data in graph.items():
            for imported in data["imports"]:
                for target_module in graph:
                    if imported in target_module or target_module.replace(".py", "") in imported:
                        if module_name not in graph[target_module]["imported_by"]:
                            graph[target_module]["imported_by"].append(module_name)
        
        return graph

    def _extract_modules_info(self) -> List[Dict[str, Any]]:
        """Extract detailed module information."""
        parsed_dict = self.analysis_results.get("phase2", {}).get("parsed_modules", {})
        # Convert dict to list for easier iteration
        return list(parsed_dict.values())

    def _initialize_llm(self):
        """Initialize the LangChain LLM for the agent."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=self.hf_token,
                temperature=0.3,  # Lower temperature for more focused documentation
            )
            self.llm = ChatHuggingFace(llm=endpoint)
            logger.info("LLM initialized for documentation agent")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    def _create_agent_tools(self) -> List[StructuredTool]:
        """
        Create tools that the agent can use to reason over the codebase.
        """
        tools = []

        # Tool 1: Get module information
        def get_module_info(module_name: str) -> str:
            """Get detailed information about a specific module."""
            for module in self.modules_info:
                if module_name in module.get("file", ""):
                    info = f"Module: {module.get('file')}\n"
                    info += f"Functions: {len(module.get('functions', []))}\n"
                    info += f"Classes: {len(module.get('classes', []))}\n"
                    info += f"Imports: {', '.join(module.get('imports', [])[:5])}\n"
                    return info
            return f"Module {module_name} not found"
        
        tools.append(StructuredTool.from_function(
            func=get_module_info,
            name="get_module_info",
            description="Get detailed information about a specific module including functions, classes, and imports"
        ))

        # Tool 2: Get dependency graph
        def get_dependencies(module_name: str) -> str:
            """Get dependencies for a module."""
            for mod, deps in self.dependency_graph.items():
                if module_name in mod:
                    return json.dumps(deps, indent=2)
            return f"No dependencies found for {module_name}"
        
        tools.append(StructuredTool.from_function(
            func=get_dependencies,
            name="get_dependencies",
            description="Get import and dependency information for a module"
        ))

        # Tool 3: Identify core modules
        def identify_core_modules() -> str:
            """Identify core modules and entry points."""
            core = self.architecture.get("core_modules", [])
            entry = self.architecture.get("entry_points", [])
            return f"Core Modules: {', '.join(core[:5])}\nEntry Points: {', '.join(entry[:5])}"
        
        tools.append(StructuredTool.from_function(
            func=identify_core_modules,
            name="identify_core_modules",
            description="Identify the core modules and entry points of the codebase"
        ))

        # Tool 4: Detect design patterns
        def detect_patterns() -> str:
            """Detect design patterns in the codebase."""
            patterns = self.architecture.get("design_patterns", {})
            return json.dumps(patterns, indent=2)
        
        tools.append(StructuredTool.from_function(
            func=detect_patterns,
            name="detect_patterns",
            description="Detect design patterns like Factory, Singleton, Observer in the code"
        ))

        # Tool 5: Analyze data flow
        def analyze_data_flow() -> str:
            """Analyze data flow between modules."""
            flow_info = ""
            for module, deps in list(self.dependency_graph.items())[:5]:
                flow_info += f"\n{module}:\n"
                flow_info += f"  Imports: {', '.join(deps['imports'][:3])}\n"
                flow_info += f"  Used by: {', '.join(deps['imported_by'][:3])}\n"
            return flow_info
        
        tools.append(StructuredTool.from_function(
            func=analyze_data_flow,
            name="analyze_data_flow",
            description="Analyze data flow and information exchange between modules"
        ))

        return tools

    def _get_agent_personality_prompt(self) -> str:
        """
        Return the agent's personality and instructions.
        
        Frames the agent as a Senior Software Engineer reviewing legacy code.
        """
        return """You are a Senior Software Engineer reviewing legacy code and creating documentation 
for junior developers. Your task is to generate clear, concise documentation that helps 
team members understand the codebase quickly.

For each module or function, provide:
1. **Purpose**: What does this module/function do? Why does it exist?
2. **Parameters**: List all parameters with their types and descriptions
3. **Return Values**: Describe what is returned, including type and meaning
4. **Examples**: Provide simple usage examples where helpful
5. **Dependencies**: Note important dependencies and their roles
6. **Design Patterns**: Identify any design patterns used

Keep explanations concise but comprehensive. Use the tools available to gather information 
about the codebase structure, dependencies, and architecture.

Focus on making the documentation accessible to developers who may be new to this codebase."""

    def generate_documentation(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate documentation using the agent.
        
        Args:
            module_name: Specific module to document (None for all modules)
            
        Returns:
            Generated documentation
        """
        logger.info("Starting Phase 3: Documentation generation")
        
        try:
            if not self.llm:
                logger.info("Using fallback documentation generation (no LLM)")
                return self._generate_documentation_fallback(module_name)
            
            # Generate documentation for each module
            all_docs = {
                "documentation_by_module": {},
                "timestamp": datetime.now().isoformat(),
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "total_modules_documented": 0,
                "overview": self._generate_codebase_overview()
            }
            
            # If specific module requested, document only that
            if module_name:
                modules_to_doc = [m for m in self.modules_info if module_name in m.get("file", "")]
            else:
                modules_to_doc = self.modules_info
            
            # Generate documentation for each module
            for module in modules_to_doc:
                module_file = module.get("file", "")
                logger.info(f"Generating documentation for {module_file}")
                
                doc = self._generate_module_documentation(module)
                all_docs["documentation_by_module"][module_file] = doc
                all_docs["total_modules_documented"] += 1
            
            logger.info("Documentation generation completed")
            return all_docs
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return self._generate_documentation_fallback(module_name)

    def _generate_codebase_overview(self) -> str:
        """
        Generate an overview of the entire codebase.
        """
        overview = f"""# {self.analysis_results.get('phase1', {}).get('metadata', {}).get('name', 'Codebase')} Documentation

## Repository Overview

**Name**: {self.analysis_results.get('phase1', {}).get('metadata', {}).get('name', 'Unknown')}
**URL**: {self.analysis_results.get('phase1', {}).get('metadata', {}).get('url', 'N/A')}
**Branch**: {self.analysis_results.get('phase1', {}).get('metadata', {}).get('branch', 'main')}

## Statistics
- **Total Modules**: {len(self.modules_info)}
- **Total Lines of Code**: {self.architecture.get('complexity_metrics', {}).get('total_lines', 0)}
- **Programming Languages**: Python

## Core Modules
"""
        
        core_modules = self.architecture.get("core_modules", [])
        for module in core_modules[:10]:
            overview += f"- {module}\n"
        
        return overview

    def _generate_module_documentation(self, module: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate documentation for a specific module.
        """
        module_file = module.get("file", "")
        module_prompt = self._build_module_prompt(module)
        
        try:
            # Query LLM for module documentation
            messages = [HumanMessage(content=module_prompt)]
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                doc_content = response.content
            else:
                doc_content = str(response)
            
            return {
                "file": module_file,
                "docstring": module.get("docstring", "N/A"),
                "functions": module.get("functions", []),
                "classes": module.get("classes", []),
                "imports": module.get("imports", []),
                "generated_documentation": doc_content,
                "dependencies": self.dependency_graph.get(module_file, {}),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"LLM generation failed for {module_file}, using fallback")
            return self._generate_module_documentation_fallback(module)

    def _build_module_prompt(self, module: Dict[str, Any]) -> str:
        """
        Build a detailed prompt for documenting a specific module.
        """
        module_file = module.get("file", "")
        functions = module.get("functions", [])
        classes = module.get("classes", [])
        imports = module.get("imports", [])
        docstring = module.get("docstring", "")
        
        prompt = f"""{self._get_agent_personality_prompt()}

## FILE TO DOCUMENT: {module_file}

### Original Docstring
{docstring if docstring else "(No docstring available)"}

### Module Imports
{json.dumps(imports[:10], indent=2)}

### Functions in Module
{json.dumps([f.get('name') for f in functions[:10]], indent=2) if functions else "No functions"}

### Classes in Module
"""
        
        for cls in classes[:10]:
            prompt += f"\n**{cls.get('name', 'Unknown')}**"
            methods = cls.get('methods', [])
            if methods:
                prompt += f"\n  Methods: {', '.join([m.get('name') for m in methods[:5]])}"
        
        prompt += f"""

Generate comprehensive documentation for the {module_file} file following this structure:

## File Purpose
Briefly explain what this module does and why it exists.

## Key Functions
For each important function, provide:
- **Function name**: description
  - Parameters: (types and descriptions)
  - Returns: (type and meaning)
  - Example usage

## Key Classes
For each important class, provide:
- **Class name**: description
- Important methods and their purposes

## Dependencies
List external and internal dependencies and their roles.

## Usage Examples
Provide 2-3 practical examples of how to use this module.

## Design Patterns
Identify any design patterns used in this module.

Format the documentation in clear, concise markdown suitable for junior developers.
"""
        
        return prompt

    def _generate_module_documentation_fallback(self, module: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate basic documentation for a module without LLM.
        """
        module_file = module.get("file", "")
        
        doc = f"""# {module_file}

## Module Purpose

This module contains {len(module.get('functions', []))} functions and {len(module.get('classes', []))} classes.

## Imports
"""
        
        for imp in module.get("imports", [])[:5]:
            doc += f"- {imp}\n"
        
        doc += "\n## Classes\n\n"
        
        for cls in module.get("classes", []):
            doc += f"### {cls.get('name', 'Unknown')}\n"
            doc += f"Location: Line {cls.get('lineno', 'N/A')}\n"
            methods = cls.get('methods', [])
            if methods:
                doc += f"Methods: {', '.join([m.get('name') for m in methods[:5]])}\n"
            doc += "\n"
        
        doc += "\n## Functions\n\n"
        
        for func in module.get("functions", []):
            doc += f"### {func.get('name', 'Unknown')}\n"
            doc += f"Location: Line {func.get('lineno', 'N/A')}\n\n"
        
        return {
            "file": module_file,
            "docstring": module.get("docstring", "N/A"),
            "functions": module.get("functions", []),
            "classes": module.get("classes", []),
            "imports": module.get("imports", []),
            "generated_documentation": doc,
            "dependencies": self.dependency_graph.get(module_file, {}),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_documentation_fallback(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate documentation without LLM.
        """
        logger.info("Using fallback documentation generation (no LLM)")
        
        all_docs = {
            "documentation_by_module": {},
            "timestamp": datetime.now().isoformat(),
            "overview": self._generate_codebase_overview(),
            "total_modules_documented": 0
        }
        
        # If specific module requested, document only that
        if module_name:
            modules_to_doc = [m for m in self.modules_info if module_name in m.get("file", "")]
        else:
            modules_to_doc = self.modules_info
        
        # Generate fallback documentation for each module
        for module in modules_to_doc:
            module_file = module.get("file", "")
            doc = self._generate_module_documentation_fallback(module)
            all_docs["documentation_by_module"][module_file] = doc
            all_docs["total_modules_documented"] += 1
        
        return all_docs


# ============================================================================
# PHASE 4: FUNCTION-LEVEL DOCSTRING GENERATION (imported from separate module)
# ============================================================================

# Import Phase 4 implementation
try:
    from phase4_docstring_generator import FunctionDocstringGenerator, READMEGenerator
except ImportError:
    logger.warning("Could not import from phase4_docstring_generator.py")
    # Provide stub classes for graceful fallback
    class FunctionDocstringGenerator:
        def __init__(self, hf_token=None):
            pass
        def process_file(self, file_path):
            return {"file": file_path, "status": "error", "functions_documented": 0}
    
    class READMEGenerator:
        def __init__(self, hf_token=None):
            pass
        def generate_readme(self, phase1_results, phase2_results, phase3_results=None, output_path=None):
            return "# README\n\nFallback README - Phase 4 module not available."


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    orchestrator = CodeToDocOrchestrator(
        # github_url="https://github.com/chavarera/s-tool.git",
        github_url="https://github.com/Jaychangani2005/Application.git",
        github_token=os.getenv("GITHUB_TOKEN")
    )

    results = orchestrator.run_all()

    # Save results
    output_path = Path("output/analysis_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
