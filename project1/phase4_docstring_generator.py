"""
Phase 4: Function-Level Docstring Generation
Generates Google-style docstrings for all functions in Python code using LLM.
"""

import os
import ast
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


class FunctionDocstringGenerator:
    """
    Generates Google-style docstrings for all functions in Python code.
    Uses LLM to analyze function logic and create comprehensive documentation.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the docstring generator with LLM support."""
        self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        self.llm = None
        self.max_context_size = 4000  # 4KB limit for function analysis

        if LANGCHAIN_AVAILABLE and self.hf_token:
            self._initialize_llm()

        logger.info("Function docstring generator initialized")

    def _initialize_llm(self):
        """Initialize LangChain LLM."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=self.hf_token,
                max_new_tokens=512,
                temperature=0.3,
            )
            self.llm = ChatHuggingFace(llm=endpoint)
            logger.info("LLM initialized for docstring generation")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a Python file and generate docstrings for all functions.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with results including updated code and statistics
        """
        logger.info(f"Processing file for docstrings: {file_path}")

        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Parse AST
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return {
                    "file": file_path,
                    "status": "error",
                    "error": str(e),
                    "functions_documented": 0
                }

            # Extract functions
            functions = self._extract_functions(tree, source_code)

            # Generate docstrings
            results = {
                "file": file_path,
                "status": "success",
                "total_functions": len(functions),
                "functions_documented": 0,
                "docstrings_generated": []
            }

            updated_lines = source_code.split('\n')

            for func_info in functions:
                docstring = self._generate_docstring(func_info)
                if docstring:
                    updated_lines = self._inject_docstring(
                        updated_lines, func_info, docstring
                    )
                    results["functions_documented"] += 1
                    results["docstrings_generated"].append({
                        "function": func_info["name"],
                        "line": func_info["lineno"]
                    })

            # Validate syntax
            updated_code = '\n'.join(updated_lines)
            if self._validate_syntax(updated_code):
                results["updated_code"] = updated_code
                results["syntax_valid"] = True
            else:
                logger.warning(f"Syntax validation failed for {file_path}")
                results["syntax_valid"] = False

            return results

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file": file_path,
                "status": "error",
                "error": str(e),
                "functions_documented": 0
            }

    def _extract_functions(self, tree: ast.AST, source_code: str) -> List[Dict[str, Any]]:
        """
        Extract all functions and methods from AST.
        """
        functions = []
        lines = source_code.split('\n')

        class FunctionExtractor(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.functions = []

            def visit_FunctionDef(self, node: ast.FunctionDef):
                # Get function source code
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(
                    node, 'end_lineno') else start_line + 20
                func_source = '\n'.join(
                    lines[start_line:min(end_line, len(lines))]
                )

                # Check if already has docstring
                has_docstring = (ast.get_docstring(node) is not None)

                self.functions.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": end_line,
                    "args": [arg.arg for arg in node.args.args],
                    "source": func_source[:self.parent.max_context_size],
                    "has_docstring": has_docstring,
                    "is_method": False
                })

                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                # Handle async functions similarly
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(
                    node, 'end_lineno') else start_line + 20
                func_source = '\n'.join(
                    lines[start_line:min(end_line, len(lines))]
                )
                has_docstring = (ast.get_docstring(node) is not None)

                self.functions.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": end_line,
                    "args": [arg.arg for arg in node.args.args],
                    "source": func_source[:self.parent.max_context_size],
                    "has_docstring": has_docstring,
                    "is_async": True,
                    "is_method": False
                })

                self.generic_visit(node)

        extractor = FunctionExtractor(self)
        extractor.visit(tree)

        return extractor.functions

    def _generate_docstring(self, func_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate a Google-style docstring for a function.
        """
        if func_info.get("has_docstring"):
            logger.debug(f"Skipping {func_info['name']} - already has docstring")
            return None

        if not self.llm:
            return self._generate_docstring_fallback(func_info)

        try:
            prompt = self._build_docstring_prompt(func_info)
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)

            if hasattr(response, 'content'):
                docstring = response.content
            else:
                docstring = str(response)

            return self._format_docstring(docstring, func_info["name"])

        except Exception as e:
            logger.warning(f"LLM generation failed for {func_info['name']}: {e}")
            return self._generate_docstring_fallback(func_info)

    def _build_docstring_prompt(self, func_info: Dict[str, Any]) -> str:
        """
        Build a prompt for LLM to generate docstring.
        """
        prompt = f"""Analyze the following Python function and generate a Google-style docstring.

Function signature:
def {func_info['name']}({', '.join(func_info['args'])}):

Function code:
{func_info['source']}

Generate a Google-style docstring with:
1. Brief one-line summary
2. Longer description if the logic is non-obvious (optional)
3. Args section with types and descriptions
4. Returns section with type and meaning
5. Raises section if the function raises exceptions (optional)
6. Example usage demonstrating how to call and interpret results

Output ONLY the properly formatted docstring content (starting with triple quotes):
"""
        return prompt

    def _format_docstring(self, content: str, func_name: str) -> str:
        """
        Format and clean the generated docstring.
        """
        # Remove markdown code blocks if present
        content = content.replace('```python', '').replace('```', '').strip()

        # Ensure proper indentation
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            # Add proper indentation (4 spaces for docstring content)
            if line.strip():
                if not line.startswith('    '):
                    line = '    ' + line
            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _generate_docstring_fallback(self, func_info: Dict[str, Any]) -> str:
        """
        Generate a basic docstring without LLM.
        """
        args_section = ""
        if func_info['args'] and func_info['args'][0] != 'self':
            args_section = "\n    Args:\n"
            for arg in func_info['args']:
                if arg != 'self':
                    args_section += f"        {arg}: Description of {arg}.\n"

        docstring = f'''    """Summary of {func_info['name']}.

    This function performs a specific task. More details about the implementation
    and algorithm used can be added here if non-obvious.{args_section}
    Returns:
        The return value and its meaning.

    Example:
        >>> result = {func_info['name']}()
        >>> print(result)
    """
        '''

        return docstring

    def _inject_docstring(self, lines: List[str], func_info: Dict[str, Any], docstring: str) -> List[str]:
        """
        Inject the generated docstring into the source code.
        """
        # Find the function definition line
        func_def_line = func_info["lineno"] - 1

        # Find the end of the function signature
        insert_line = func_def_line + 1
        while insert_line < len(lines) and not lines[insert_line].rstrip().endswith(':'):
            insert_line += 1

        insert_line += 1  # Insert after the colon

        # Insert docstring
        docstring_lines = docstring.split('\n')
        for i, line in enumerate(docstring_lines):
            lines.insert(insert_line + i, line)

        return lines

    def _validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class READMEGenerator:
    """
    Generates professional README.md for legacy codebases using LLM.
    Uses analysis results from previous phases to create comprehensive documentation.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the README generator with LLM support."""
        self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        self.llm = None

        if LANGCHAIN_AVAILABLE and self.hf_token:
            self._initialize_llm()

        logger.info("README generator initialized")

    def _initialize_llm(self):
        """Initialize LangChain LLM for README generation."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=self.hf_token,
                max_new_tokens=2048,  # Longer for comprehensive README
                temperature=0.5,
            )
            self.llm = ChatHuggingFace(llm=endpoint)
            logger.info("LLM initialized for README generation")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    def generate_readme(
        self, 
        phase1_results: Dict[str, Any],
        phase2_results: Dict[str, Any],
        phase3_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate professional README.md from analysis results.
        
        Args:
            phase1_results: Repository metadata from Phase 1
            phase2_results: Code analysis results from Phase 2
            phase3_results: Module documentation from Phase 3 (optional)
            output_path: Path where README.md should be saved (optional)
            
        Returns:
            Generated README content as string
        """
        logger.info("Generating README.md from analysis results")

        # Extract key information
        repo_info = self._extract_repo_info(phase1_results)
        code_stats = self._extract_code_stats(phase2_results)
        architecture = self._extract_architecture(phase2_results)
        dependencies = self._extract_dependencies(phase2_results)
        
        # Build comprehensive prompt for LLM
        prompt = self._build_readme_prompt(
            repo_info, code_stats, architecture, dependencies, phase3_results
        )

        # Generate README with LLM or fallback
        if self.llm:
            readme_content = self._generate_with_llm(prompt)
        else:
            readme_content = self._generate_fallback_readme(
                repo_info, code_stats, architecture, dependencies
            )

        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                logger.info(f"README.md saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save README.md: {e}")

        return readme_content

    def _extract_repo_info(self, phase1_results: Dict[str, Any]) -> Dict[str, str]:
        """Extract repository metadata (align with phase1 structure)."""
        meta = phase1_results.get("metadata", phase1_results.get("repository_metadata", {}))
        return {
            "name": meta.get("name") or meta.get("repository_name", "Unknown Project"),
            "description": meta.get("description", "Legacy codebase"),
            "url": meta.get("url") or meta.get("repository_url", ""),
            "language": meta.get("primary_language", "Python"),
            "clone_path": phase1_results.get("repo_path") or phase1_results.get("clone_path", ""),
        }

    def _extract_code_stats(self, phase2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract code statistics."""
        code_files = phase2_results.get("code_files", {})
        stats = phase2_results.get("statistics", {})
        return {
            "total_files": stats.get("total_files", phase2_results.get("total_files", 0)),
            "python_files": len(code_files.get("python", [])),
            "test_files": len(code_files.get("tests", [])),
            "total_functions": sum(
                len(m.get("functions", []))
                for m in phase2_results.get("parsed_modules", {}).values()
            ),
            "total_classes": sum(
                len(m.get("classes", []))
                for m in phase2_results.get("parsed_modules", {}).values()
            ),
            "language_breakdown": code_files,
        }

    def _extract_architecture(self, phase2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract architecture information (align with stored keys)."""
        arch = phase2_results.get("architecture", phase2_results.get("architecture_analysis", {}))
        llm = phase2_results.get("llm_analysis", phase2_results.get("llm_architecture_analysis", {}))

        return {
            "core_modules": arch.get("core_modules", []),
            "layer_structure": arch.get("layers") or arch.get("layer_structure", {}),
            "design_patterns": llm.get("design_patterns", []),
            "architecture_style": llm.get("architecture_style", "Unknown"),
            "key_components": llm.get("key_components", []),
            "dependency_graph": phase2_results.get("dependency_graph", {}),
            "adjacency_matrix": phase2_results.get("adjacency_matrix", {}),
        }

    def _extract_dependencies(self, phase2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dependency information."""
        dep_graph = phase2_results.get("dependency_graph", {})

        # Collect unique external dependencies
        external_deps = set()
        for deps in dep_graph.values():
            external_deps.update(deps.get("external", []))

        return {
            "external_libraries": sorted(list(external_deps)),
            "internal_modules": list(dep_graph.keys()),
            "dependency_graph": dep_graph,
        }

    def _build_readme_prompt(
        self,
        repo_info: Dict,
        code_stats: Dict,
        architecture: Dict,
        dependencies: Dict,
        phase3_results: Optional[Dict] = None
    ) -> str:
        """Build comprehensive prompt for README generation."""
        
        # Build ASCII dependency diagram
        ascii_diagram = self._build_ascii_diagram(architecture.get("dependency_graph", {}))
        
        prompt = f"""Generate a professional README.md for this legacy codebase that a junior developer can easily understand.

    INPUTS (real project data):

    Repository Metadata:
    - Name: {repo_info['name']}
    - Description: {repo_info['description']}
    - Primary Language: {repo_info['language']}
    - Repository URL: {repo_info.get('url', 'N/A')}

    Code Statistics:
    - Total Files: {code_stats['total_files']}
    - Python Files: {code_stats['python_files']}
    - Test Files: {code_stats['test_files']}
    - Total Functions: {code_stats['total_functions']}
    - Total Classes: {code_stats['total_classes']}

    Architecture Overview:
    - Architecture Style: {architecture.get('architecture_style', 'Modular')}
    - Core Modules: {', '.join(architecture.get('core_modules', [])[:5])}
    - Design Patterns: {', '.join(architecture.get('design_patterns', [])[:3])}
    - Layer Structure: {architecture.get('layer_structure', {})}

    Dependency Diagram (ASCII):
    {ascii_diagram}

    External Dependencies (top):
    {chr(10).join(f'- {dep}' for dep in dependencies.get('external_libraries', [])[:15])}

    Internal Modules (top):
    {chr(10).join(f'- {mod}' for mod in dependencies.get('internal_modules', [])[:10])}

    TASK:
    Create well-structured sections for:
    1. Overview - Project purpose and key features
    2. Architecture - High-level system design, core components and responsibilities, key design patterns, ASCII dependency diagram
    3. Dependencies - External libraries with brief descriptions, internal module relationships
    4. Getting Started - Prerequisites, installation steps, minimal usage example
    5. Module Guide - Summarize each core module, its purpose, key classes/functions
    6. API Reference (if applicable) - Key interfaces and usage
    7. Known Limitations - Technical debt, missing features
    8. Development Notes - How to contribute, testing approach

    Write in clear, concise, professional language suitable for onboarding a new developer.
    Output ONLY the contents of README.md, properly formatted in Markdown.
    Do not include any explanations or metadata outside the README content.
    """
        return prompt

    def _build_ascii_diagram(self, dependency_graph: Dict) -> str:
        """Build simple ASCII dependency diagram."""
        if not dependency_graph:
            return "No dependency information available"
        
        lines = []
        lines.append("```")
        
        # Show top 8 modules with their dependencies
        for i, (module, deps) in enumerate(list(dependency_graph.items())[:8]):
            module_name = module.split('/')[-1].replace('.py', '')
            internal = deps.get('internal', [])[:3]
            
            if internal:
                lines.append(f"{module_name}")
                for dep in internal:
                    dep_name = dep.split('/')[-1].replace('.py', '')
                    lines.append(f"  ├─> {dep_name}")
                lines.append("")
        
        lines.append("```")
        return "\n".join(lines)

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate README using LLM."""
        try:
            logger.info("Generating README with LLM...")
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            
            readme_content = response.content.strip()
            
            # Clean up any markdown code blocks if LLM wrapped the output
            if readme_content.startswith("```markdown"):
                readme_content = readme_content[11:]
            if readme_content.startswith("```"):
                readme_content = readme_content[3:]
            if readme_content.endswith("```"):
                readme_content = readme_content[:-3]
            
            logger.info("README generated successfully with LLM")
            return readme_content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_readme(repo_info, code_stats, architecture, dependencies)

    def _generate_fallback_readme(
        self,
        repo_info: Dict,
        code_stats: Dict,
        architecture: Dict,
        dependencies: Dict
    ) -> str:
        """Generate basic README without LLM."""
        logger.warning("Generating fallback README without LLM")
        
        # Prepare module guide entries
        module_entries = []
        for m in dependencies.get('internal_modules', [])[:5]:
            module_name = m.split('/')[-1]
            module_entries.append(f'### {m}\nCore module for {module_name}\n')
        module_guide = '\n'.join(module_entries)
        
        # Prepare external libraries list
        ext_libs = '\n'.join(f'- {lib}' for lib in dependencies.get('external_libraries', [])[:10])
        
        # Prepare core modules list
        core_mods = '\n'.join(f'- `{m}`' for m in architecture.get('core_modules', [])[:5])
        
        # Prepare design patterns list
        patterns = '\n'.join(f'- {p}' for p in architecture.get('design_patterns', []))
        
        readme = f"""# {repo_info.get('name', 'Project')}

{repo_info.get('description', 'Legacy codebase documentation')}

## Overview

This is a {repo_info.get('language', 'Python')} project with {code_stats.get('total_files', 0)} files containing {code_stats.get('total_functions', 0)} functions and {code_stats.get('total_classes', 0)} classes.

## Architecture

### Core Modules
{core_mods}

### Design Patterns
{patterns}

## Dependencies

### External Libraries
{ext_libs}

## Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
```bash
git clone {repo_info.get('url', '<repository-url>')}
cd {repo_info.get('name', 'project')}
pip install -r requirements.txt
```

### Usage
```python
# TODO: Add usage example
```

## Module Guide

{module_guide}

## Development Notes

See source code for detailed documentation.

## License

See LICENSE file for details.
"""
        return readme


if __name__ == "__main__":
    # Example usage
    generator = FunctionDocstringGenerator()

    # Process a sample file
    test_file = "example.py"
    if os.path.exists(test_file):
        result = generator.process_file(test_file)
        print(f"Processed {result['file']}")
        print(f"Functions documented: {result['functions_documented']}")
