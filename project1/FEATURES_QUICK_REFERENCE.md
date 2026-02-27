# Quick Reference - New Features

## 1️⃣ Backup Creation

**What it does:** Saves existing clone before creating a new one

**Where:** `RepositoryManager._create_backup()`

**Backup location:** `./backups/clone_backup_YYYYMMDD_HHMMSS/`

**Automatic trigger:** Called in `clone_repository()` if clone already exists

```python
# Manual usage
backup_path = repo_manager._create_backup()
if backup_path:
    print(f"Backup created at {backup_path}")
```

---

## 2️⃣ Test File Separation

**What it does:** Separates test files from production code

**Test patterns detected:**
- Files with `test` in name/path
- Files with `spec` in name
- Files with `_test.` pattern
- Files in `tests/` directory

**Enable/Disable:**
```python
# Enable (default)
scanner.scan_repository(separate_tests=True)

# Disable
scanner.scan_repository(separate_tests=False)
```

**Output structure:**
```json
{
  "files": {"python": [...], "java": [...]},
  "test_files": {"python": [...], "java": [...]},
  "statistics": {
    "code_files": 35,
    "test_files": 10,
    ...
  }
}
```

---

## 3️⃣ Dependency Adjacency Matrix

**What it does:** Creates mathematical representation of module dependencies

**Matrix format:**
- Rows: importing modules
- Columns: imported modules
- Cell value: 1 if row imports column, 0 otherwise

**Access in results:**
```python
results = orchestrator.run_all()
matrix_data = results['phase2']['adjacency_matrix']

# Use matrix
modules = matrix_data['modules']
matrix = matrix_data['matrix']
total_deps = matrix_data['total_dependencies']

# Use CSV export
csv_string = results['phase2']['matrix_csv']
with open('dependencies.csv', 'w') as f:
    f.write(csv_string)
```

**Example matrix:**
```
     core.py  driver.py  utils.py
core.py    0        0          1     (core imports utils)
driver.py  1        0          1     (driver imports core and utils)
utils.py   0        0          0     (utils imports nothing)
```

---

## 4️⃣ LLM Architecture Analysis

**What it does:** Uses AI to analyze architecture patterns and provide insights

**Provider:** Hugging Face (Mistral-7B model, free API)

**Setup required:**
```
# In .env file:
HUGGINGFACEHUB_ACCESS_TOKEN=your_token_here
```

Get free token at: https://huggingface.co/settings/tokens

**Features:**
- ✓ Identifies architectural components
- ✓ Detects design patterns
- ✓ Maps data flow
- ✓ Suggests improvements
- ✓ Identifies risk areas

**Access in results:**
```python
results = orchestrator.run_all()
llm_analysis = results['phase2']['llm_analysis']

if 'error' not in llm_analysis:
    print(llm_analysis['llm_analysis'])  # AI insights
    print(llm_analysis['code_summary'])   # Structured summary
else:
    print("LLM analysis skipped, using heuristics")
```

**Auto-retry behavior:**
- Retries up to 3 times on failures
- Handles model loading delays (503 errors)
- Timeout handling (30 seconds per request)
- Graceful fallback to heuristic analysis

**API Usage (Free Tier):**
- Model: mistralai/Mistral-7B-Instruct-v0.1
- Max length: 512 tokens
- Temperature: 0.7 (creative but focused)
- Endpoint: https://api-inference.huggingface.co/models/

---

## 📊 Complete Phase 2 Output Structure

```json
{
  "phase2": {
    "code_files": {
      "python": [{...}, {...}],
      "java": [{...}]
    },
    "test_files": {
      "python": [{...}],
      "java": [{...}]
    },
    "statistics": {
      "total_files": 250,
      "code_files": 35,
      "test_files": 10,
      "binary_files": 15,
      "ignored_files": 190
    },
    "parsed_modules": {
      "core.py": {
        "file": "core.py",
        "functions": [...],
        "classes": [...],
        ...
      }
    },
    "dependency_graph": {
      "core.py": {
        "imports": ["utils.py"],
        "imported_by": ["driver.py"],
        "external_deps": ["requests", "numpy"]
      }
    },
    "adjacency_matrix": {
      "modules": ["core.py", "driver.py", "utils.py"],
      "matrix": [[0,0,1], [1,0,1], [0,0,0]],
      "total_dependencies": 3,
      "module_count": 3
    },
    "matrix_csv": "module,core.py,driver.py,utils.py\n...",
    "complexity_metrics": {
      "total_files": 45,
      "total_lines": 15230,
      "circular_dependencies": []
    },
    "architecture": {
      "core_modules": ["core.py", "utils.py"],
      "entry_points": ["main.py"],
      "design_patterns": {...},
      "layers": {...}
    },
    "llm_analysis": {
      "llm_analysis": "Core architectural components...",
      "code_summary": "MODULES:\n- core.py: 5 functions...",
      "analysis_timestamp": "2026-01-26T10:30:45.123456"
    }
  }
}
```

---

## 🔍 Troubleshooting

**Issue:** LLM analysis returns error
- **Solution:** Check `HUGGINGFACEHUB_ACCESS_TOKEN` in `.env`
- **Fallback:** Heuristic analysis still runs

**Issue:** Backup fails
- **Solution:** Check disk space and folder permissions
- **Impact:** Clone proceeds without backup (non-blocking)

**Issue:** Matrix CSV is too large
- **Solution:** Filter modules before creating matrix
- **Tip:** Use in tools like Excel, Gephi for visualization

**Issue:** Test files not separated
- **Solution:** Verify file names match patterns
- **Patterns:** `test`, `spec`, `_test.`, `tests/` directory

---

## 💡 Tips & Tricks

1. **Export matrix for visualization:**
   ```python
   csv = results['phase2']['matrix_csv']
   # Import into Gephi or Excel for dependency graphs
   ```

2. **Filter LLM analysis:**
   ```python
   llm = results['phase2']['llm_analysis']
   if 'error' not in llm:
       insights = llm['llm_analysis']
   ```

3. **Backup recovery:**
   ```python
   # Find backup folder
   from pathlib import Path
   backups = list(Path('backups').iterdir())
   latest = max(backups, key=lambda p: p.stat().st_mtime)
   ```

4. **Skip test files in analysis:**
   ```python
   main_files = results['phase2']['code_files']  # Excludes tests
   test_files = results['phase2']['test_files']  # Tests only
   ```

5. **Check for circular dependencies:**
   ```python
   cycles = results['phase2']['complexity_metrics']['circular_dependencies']
   if cycles:
       print(f"⚠️ Found {len(cycles)} circular dependencies")
   ```
