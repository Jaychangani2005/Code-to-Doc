import os
import sys

# Import functions
from ingest import clone_repository, get_code_files
from graph_builder import build_dependency_graph


def _load_env_file(env_path: str = ".env") -> None:
    """Minimal .env loader (no external deps).

    Loads KEY=VALUE lines into os.environ if KEY is not already set.
    Supports quoted values and ignores blank lines and comments.
    """
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key or key in os.environ:
                    continue

                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                os.environ[key] = value
    except OSError:
        # If .env can't be read, we just proceed using the process env.
        return

def main():
    # 0. Load environment variables from .env (if present)
    _load_env_file(".env")

    # Import after environment is loaded (avoids import-order surprises)
    from doc_generator import generate_file_docs

    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
        print(
            "❌ Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env "
            "or in your environment before running."
        )
        return

    print("🚀 Starting Code-to-Doc Agent...")

    # 1. Get Repo URL
    # url = input("🔗 Enter GitHub Repo URL: ").strip()
    # url ="https://github.com/chavarera/s-tool.git".strip()
    url ="https://github.com/pallets/flask.git".strip()

    if not url: return

    # 2. Clone
    repo_path = clone_repository(url)
    if not repo_path: return

    # 3. Scan Files
    file_paths = get_code_files(repo_path)
    if not file_paths: return

    # 4. Build Graph
    print(f"\n🧠 Analyzing dependencies...")
    graph = build_dependency_graph(file_paths, repo_path)
    
    # 5. GENERATE DOCUMENTATION
    print("\n📝 Starting Documentation Phase...")
    
    output_dir = "./generated_docs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 3 files for testing so we don't wait forever
    for file_path in file_paths[:3]: 
        rel_path = os.path.relpath(file_path, repo_path)
        
        # Get neighbors from the graph (Context!)
        # We look for the node in the graph using the relative path
        neighbors = []
        if graph.has_node(rel_path):
            neighbors = list(graph.successors(rel_path)) # Files this file IMPORTS
        
        # Generate Docs
        doc_content = generate_file_docs(file_path, neighbors)
        
        # Save to Markdown file
        safe_name = rel_path.replace("/", "_").replace("\\", "_") + ".md"
        with open(os.path.join(output_dir, safe_name), "w", encoding="utf-8") as f:
            f.write(doc_content)
            
    print(f"\n✅ Documentation generated in {output_dir}")

if __name__ == "__main__":
    main()