import ast
import os
import networkx as nx

class ImportVisitor(ast.NodeVisitor):
    """
    Custom AST Visitor to extract imports from Python code.
    """
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        # Handles: import os, sys
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Handles: from .utils import helper
        module = node.module or ''
        # Logic to handle relative imports (dots)
        if node.level > 0:
            module = '.' * node.level + module
        self.imports.append(module)
        self.generic_visit(node)

def build_dependency_graph(file_paths, root_dir):
    """
    Builds a directed graph of internal file dependencies.
    
    Args:
        file_paths (list): List of full file paths (from crawler).
        root_dir (str): Root directory of the repo (to normalize paths).
        
    Returns:
        nx.DiGraph: A NetworkX Directed Graph.
    """
    G = nx.DiGraph()
    
    # Create a lookup map: "utils" -> "src/utils.py"
    # This helps us map "import utils" back to the actual file path.
    module_map = {}
    for f in file_paths:
        # Create a Python-style module name from the file path
        # e.g., "repo/src/utils.py" -> "src.utils"
        rel_path = os.path.relpath(f, root_dir)
        module_name = rel_path.replace(os.sep, '.').replace('.py', '')
        
        # specific fix for __init__.py which is often imported by its parent folder name
        if module_name.endswith('.__init__'):
            module_name = module_name[:-9]
            
        module_map[module_name] = rel_path
        
        # Add the file as a node in the graph
        G.add_node(rel_path)

    print("🕸️  Building Dependency Graph...")

    for file_path in file_paths:
        current_node = os.path.relpath(file_path, root_dir)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            # Extract imports
            visitor = ImportVisitor()
            visitor.visit(tree)
            
            for imported_module in visitor.imports:
                # We only care if the import is INSIDE our project (not 'os', 'json', etc.)
                # We check if the imported name matches one of our files.
                
                # Simple matching logic (can be expanded)
                matched_file = None
                
                # Exact match?
                if imported_module in module_map:
                    matched_file = module_map[imported_module]
                
                # If found, add an edge!
                if matched_file:
                    G.add_edge(current_node, matched_file)
                    # print(f"  Link found: {current_node} -> {matched_file}")
                    
        except SyntaxError:
            print(f"⚠️  Skipping {file_path} (Syntax Error)")
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")

    print(f"✅ Graph built with {G.number_of_nodes()} files and {G.number_of_edges()} dependencies.")
    return G

# --- Integration Test ---
if __name__ == "__main__":
    # Test with the data we (hypothetically) have from previous steps
    # You would import get_code_files here normally
    import sys
    
    # Mocking the previous step for demonstration
    root = "./repo_data"
    if os.path.exists(root):
        # 1. Get Files (Reusing crawler logic essentially)
        all_files = []
        for r, d, f in os.walk(root):
            for file in f:
                if file.endswith(".py"):
                    all_files.append(os.path.join(r, file))
        
        # 2. Build Graph
        graph = build_dependency_graph(all_files, root)
        
        # 3. Print "Hub" files (Files that are imported by many others)
        print("\n--- Top 3 Most Critical Modules (Most In-Degree) ---")
        # In-degree = how many other files depend on this one
        top_dependencies = sorted(graph.in_degree, key=lambda x: x[1], reverse=True)[:3]
        for node, degree in top_dependencies:
            print(f"{node} is used by {degree} other files.")