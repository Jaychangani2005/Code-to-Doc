# Configuration & API Reference

## Environment Variables

### Hugging Face LLM Integration

**Get token at:** https://huggingface.co/settings/tokens

**Token type:** `read` access (for inference)

---

## LLM Model Details

| Property | Value |
|----------|-------|
| **Provider** | Hugging Face |
| **Model** | mistralai/Mistral-7B-Instruct-v0.1 |
| **Type** | Open-source LLM (7B parameters) |
| **Endpoint** | https://api-inference.huggingface.co/models |
| **Cost** | Free (rate-limited) |
| **Max Input** | Unlimited (uses model default) |
| **Max Output** | 512 tokens |
| **Temperature** | 0.7 (balanced creativity) |
| **Timeout** | 30 seconds |
| **Retries** | 3 attempts |

---

## API Request Format

### Hugging Face Inference Request

```python
import requests

url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {hf_token}"
}

payload = {
    "inputs": "Your analysis prompt here...",
    "parameters": {
        "max_length": 512,
        "temperature": 0.7,
    }
}

response = requests.post(url, json=payload, headers=headers, timeout=30)
result = response.json()
```

### Response Format (Success)

```json
[
  {
    "generated_text": "Analysis text here..."
  }
]
```

### Response Format (Model Loading)

```json
{
  "error": "Model is currently loading"
}
```

**Status Code:** 503 (Service Unavailable - will retry)

---

## Class Configuration

### LLMArchitectureAnalyzer

```python
from phase_1_2_claude import LLMArchitectureAnalyzer

# Initialize with explicit token
analyzer = LLMArchitectureAnalyzer(hf_token="hf_xxx...")

# Or use .env (automatic)
analyzer = LLMArchitectureAnalyzer()  # Reads HUGGINGFACEHUB_ACCESS_TOKEN

# Call analysis
result = analyzer.analyze_architecture(parsed_data, dependency_graph)
```

**Parameters:**
- `hf_token` (str, optional): Hugging Face API token. If None, reads from env.

**Methods:**
- `analyze_architecture(parsed_data, dependency_graph)` → Dict
- `_prepare_code_summary(parsed_data, dependency_graph)` → str
- `_create_analysis_prompt(code_summary)` → str
- `_query_huggingface(prompt, max_retries=3)` → str

---

### DependencyAdjacencyMatrixBuilder

```python
from phase_1_2_claude import DependencyAdjacencyMatrixBuilder

# Initialize with dependency graph
builder = DependencyAdjacencyMatrixBuilder(dependency_graph)

# Build matrix
matrix_data = builder.build_adjacency_matrix()

# Get CSV export
csv_content = builder.generate_matrix_csv(matrix_data)
```

**Methods:**
- `build_adjacency_matrix()` → Dict with keys:
  - `modules` (List[str]): Sorted module names
  - `matrix` (List[List[int]]): 2D adjacency matrix
  - `total_dependencies` (int): Count of 1s in matrix
  - `module_count` (int): Number of modules

- `generate_matrix_csv(matrix_data)` → str: CSV format

---

## Orchestrator Configuration

### CodeToDocOrchestrator

```python
from phase_1_2_claude import CodeToDocOrchestrator

# Initialize
orchestrator = CodeToDocOrchestrator(
    github_url="https://github.com/user/repo.git",
    github_token=None  # Optional, for private repos
)

# Run all phases
results = orchestrator.run_all()

# Or run individually
repo_path = orchestrator.execute_phase1()
phase2_results = orchestrator.execute_phase2(repo_path)
```

**Parameters:**
- `github_url` (str): GitHub repository URL
- `github_token` (str, optional): GitHub token for private repos

---

## Scanner Configuration

### CodeScanner Options

```python
from phase_1_2_claude import CodeScanner

scanner = CodeScanner(repo_path)

# With test file separation (NEW)
results = scanner.scan_repository(separate_tests=True)

# Without test file separation
results = scanner.scan_repository(separate_tests=False)
```

**Parameters:**
- `separate_tests` (bool): Default True. If True, test files go to separate category.

**Return keys:**
- `files` (Dict): Production code files by language
- `test_files` (Dict): Test files by language
- `statistics` (Dict): File counts and stats

---

## Repository Manager Configuration

### RepositoryManager

```python
from phase_1_2_claude import RepositoryManager

# Default 2GB limit
manager = RepositoryManager()

# Custom size limit (in GB)
manager = RepositoryManager(max_size_gb=5)
```

**Directories:**
- Clone location: `./claude_cloned/`
- Backup location: `./backups/clone_backup_YYYYMMDD_HHMMSS/`

**Size limits:**
- Default: 2 GB
- Can be customized per instance

---

## Logging Configuration

### Current Setup

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### To Change Log Level

```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # More verbose
```

### Log Output Examples

```
2026-01-26 10:30:45,123 - INFO - Starting repository clone from https://github.com/...
2026-01-26 10:30:46,456 - INFO - Created backup at ./backups/clone_backup_20260126_103045
2026-01-26 10:30:52,789 - INFO - Repository cloned successfully to ./claude_cloned
2026-01-26 10:31:00,123 - INFO - Starting code scan on ./claude_cloned
2026-01-26 10:31:05,456 - INFO - Scan complete: 35 code files, 10 test files found
2026-01-26 10:31:10,789 - INFO - Running LLM-powered architecture analysis
2026-01-26 10:31:20,123 - INFO - LLM analysis completed
```

---

## File Structure & Locations

```
project1/
├── .env                              # Configuration
├── phase_1_2_claude.py               # Main script
├── output/
│   └── analysis_results.json         # Final output
├── claude_cloned/                    # Current repository clone
└── backups/                          # Backup directory (NEW)
    ├── clone_backup_20260126_103045/
    ├── clone_backup_20260126_113212/
    └── clone_backup_20260126_120000/
```

---

## Performance Metrics

### Typical Execution Times

| Phase | Operation | Time (small repo*) | Time (large repo**) |
|-------|-----------|-------------------|-------------------|
| 1 | Clone | 5-10s | 30-60s |
| 1 | Metadata | 1s | 1s |
| 2 | Scan | 2-3s | 10-20s |
| 2 | Parse | 3-5s | 30-60s |
| 2 | Dependencies | 2-3s | 10-20s |
| 2 | Adjacency Matrix | 1s | 5-10s |
| 2 | LLM Analysis | 15-30s | 20-40s |
| **Total** | **All phases** | **30-55s** | **110-220s** |

*Small repo: ~50 files, <5000 lines
**Large repo: ~500+ files, >100,000 lines

---

## Error Handling

### LLM Analysis Failures

```json
{
  "llm_analysis": {
    "error": "API Error: 503",
    "fallback": "Using heuristic analysis only"
  }
}
```

**Retry logic:**
- 503 errors: Wait 5s, retry up to 3 times
- Timeouts: Retry up to 3 times
- Other errors: Fail gracefully, use heuristics

### Backup Failures

```
WARNING - Failed to create backup: Permission denied
```

**Impact:** Clone proceeds without backup (non-blocking failure)

### Clone Failures

- URL validation error → Raises ValueError
- Git error → Raises OSError
- Size limit exceeded → Raises OSError

---

## Output JSON Schema

### Full Phase 2 Structure

```json
{
  "type": "object",
  "properties": {
    "code_files": {
      "type": "object",
      "description": "Production code files by language"
    },
    "test_files": {
      "type": "object",
      "description": "Test files by language (NEW)"
    },
    "statistics": {
      "type": "object",
      "properties": {
        "total_files": {"type": "integer"},
        "code_files": {"type": "integer"},
        "test_files": {"type": "integer"},
        "binary_files": {"type": "integer"},
        "ignored_files": {"type": "integer"}
      }
    },
    "parsed_modules": {"type": "object"},
    "dependency_graph": {"type": "object"},
    "adjacency_matrix": {
      "type": "object",
      "properties": {
        "modules": {"type": "array"},
        "matrix": {"type": "array"},
        "total_dependencies": {"type": "integer"},
        "module_count": {"type": "integer"}
      }
    },
    "matrix_csv": {"type": "string"},
    "complexity_metrics": {"type": "object"},
    "architecture": {"type": "object"},
    "llm_analysis": {"type": "object"}
  }
}
```
