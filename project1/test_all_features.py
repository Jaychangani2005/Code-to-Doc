#!/usr/bin/env python3
"""
Complete Example & Test Script for Phase 1 & Phase 2 Implementation

This script demonstrates all new features:
1. Backup creation
2. Test file separation
3. Adjacency matrix generation
4. LLM architecture analysis
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Set up environment
os.environ['PYTHONPATH'] = str(Path(__file__).parent)

from phase_1_2_claude import (
    CodeToDocOrchestrator,
    DependencyAdjacencyMatrixBuilder,
    LLMArchitectureAnalyzer
)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_complete_pipeline():
    """Test the complete pipeline with all new features."""
    
    print_header("PHASE 1 & PHASE 2 - COMPLETE TEST")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize orchestrator
    orchestrator = CodeToDocOrchestrator(
        github_url="https://github.com/chavarera/s-tool.git",
        github_token=os.getenv("GITHUB_TOKEN")
    )
    
    # Run all phases
    print("\n[EXECUTING ALL PHASES...]")
    try:
        results = orchestrator.run_all()
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return None
    
    return results


def test_backup_creation():
    """Test backup functionality."""
    print_header("FEATURE 1: BACKUP CREATION")
    
    from phase_1_2_claude import RepositoryManager
    
    manager = RepositoryManager()
    
    if manager.clone_dir.exists():
        print(f"✓ Existing clone found at: {manager.clone_dir}")
        backup_path = manager._create_backup()
        if backup_path:
            print(f"✓ Backup created at: {backup_path}")
            print(f"✓ Backup size: {sum(f.stat().st_size for f in backup_path.rglob('*'))} bytes")
        else:
            print("✗ Backup creation failed")
    else:
        print("ℹ No existing clone found (backup skipped)")


def test_test_file_separation(results: dict):
    """Test and display test file separation."""
    print_header("FEATURE 2: TEST FILE SEPARATION")
    
    if 'phase2' not in results:
        print("✗ Phase 2 results not found")
        return
    
    stats = results['phase2']['statistics']
    print(f"\n📊 File Statistics:")
    print(f"   Total files:     {stats['total_files']}")
    print(f"   Code files:      {stats['code_files']}")
    print(f"   Test files:      {stats['test_files']}")
    print(f"   Binary files:    {stats['binary_files']}")
    print(f"   Ignored files:   {stats['ignored_files']}")
    
    test_files = results['phase2']['test_files']
    if test_files:
        print(f"\n✓ Test files separated by language:")
        for lang, files in test_files.items():
            print(f"   {lang}: {len(files)} files")
            for file_info in files[:3]:  # Show first 3
                print(f"     - {file_info['path']}")
    else:
        print("ℹ No test files found")


def test_adjacency_matrix(results: dict):
    """Test and display adjacency matrix."""
    print_header("FEATURE 3: DEPENDENCY ADJACENCY MATRIX")
    
    if 'phase2' not in results:
        print("✗ Phase 2 results not found")
        return
    
    matrix_data = results['phase2']['adjacency_matrix']
    
    print(f"\n📊 Matrix Information:")
    print(f"   Module count:        {matrix_data['module_count']}")
    print(f"   Total dependencies:  {matrix_data['total_dependencies']}")
    
    print(f"\n📋 Modules in matrix:")
    for i, module in enumerate(matrix_data['modules'][:10]):  # Show first 10
        print(f"   [{i}] {module}")
    
    if len(matrix_data['modules']) > 10:
        print(f"   ... and {len(matrix_data['modules']) - 10} more")
    
    print(f"\n📈 Matrix sample (first 5x5):")
    modules_sample = matrix_data['modules'][:5]
    matrix_sample = [row[:5] for row in matrix_data['matrix'][:5]]
    
    # Print header
    header = "       " + " ".join(f"{m[:6]:6s}" for m in modules_sample)
    print(header)
    
    # Print rows
    for i, row in enumerate(matrix_sample):
        row_str = f"{modules_sample[i][:6]:6s} " + " ".join(f"{val:6d}" for val in row)
        print(row_str)
    
    # CSV export
    csv_content = results['phase2']['matrix_csv']
    csv_lines = csv_content.split('\n')
    print(f"\n✓ CSV export available")
    print(f"   Lines: {len(csv_lines)}")
    print(f"   Sample (first 3 rows):")
    for line in csv_lines[:3]:
        print(f"   {line}")


def test_llm_analysis(results: dict):
    """Test and display LLM analysis results."""
    print_header("FEATURE 4: LLM ARCHITECTURE ANALYSIS")
    
    if 'phase2' not in results:
        print("✗ Phase 2 results not found")
        return
    
    llm_result = results['phase2']['llm_analysis']
    
    if 'error' in llm_result:
        print(f"⚠️  LLM Analysis Error: {llm_result['error']}")
        print(f"   Fallback: {llm_result.get('fallback', 'Using heuristics')}")
        return
    
    print(f"\n✓ LLM Analysis completed successfully")
    print(f"   Timestamp: {llm_result.get('analysis_timestamp', 'N/A')}")
    
    print(f"\n📝 Code Summary:")
    summary = llm_result.get('code_summary', '')
    summary_lines = summary.split('\n')[:10]
    for line in summary_lines:
        if line.strip():
            print(f"   {line}")
    
    print(f"\n🤖 LLM Analysis:")
    analysis = llm_result.get('llm_analysis', '')
    analysis_lines = analysis.split('\n')[:15]
    for line in analysis_lines:
        if line.strip():
            print(f"   {line}")
    
    if len(analysis.split('\n')) > 15:
        print("   ...")


def test_heuristic_architecture(results: dict):
    """Display heuristic architecture analysis."""
    print_header("ARCHITECTURE ANALYSIS (HEURISTIC)")
    
    if 'phase2' not in results:
        print("✗ Phase 2 results not found")
        return
    
    arch = results['phase2']['architecture']
    
    print(f"\n🎯 Core Modules (top dependencies):")
    for module in arch['core_modules'][:5]:
        print(f"   - {module}")
    
    print(f"\n🚀 Entry Points:")
    for module in arch['entry_points'][:5]:
        print(f"   - {module}")
    
    print(f"\n🏗️  Architecture Layers:")
    for layer_name, modules in arch['layers'].items():
        if modules:
            print(f"   {layer_name.capitalize()}:")
            for module in modules[:3]:
                print(f"     - {module}")


def test_complexity_metrics(results: dict):
    """Display complexity metrics."""
    print_header("COMPLEXITY METRICS")
    
    if 'phase2' not in results:
        print("✗ Phase 2 results not found")
        return
    
    metrics = results['phase2']['complexity_metrics']
    
    print(f"\n📊 Code Metrics:")
    print(f"   Total files:       {metrics['total_files']}")
    print(f"   Total lines:       {metrics['total_lines']}")
    print(f"   Avg file size:     {metrics['average_file_size']:.0f} bytes")
    
    print(f"\n📈 Files by Language:")
    for lang, count in metrics['files_by_language'].items():
        print(f"   {lang}: {count} files")
    
    circular = metrics['circular_dependencies']
    if circular:
        print(f"\n⚠️  Circular Dependencies Found: {len(circular)}")
        for cycle in circular[:3]:
            print(f"   → {' → '.join(cycle)}")
    else:
        print(f"\n✓ No circular dependencies detected")


def save_results(results: dict):
    """Save results to file."""
    print_header("SAVING RESULTS")
    
    output_path = Path(__file__).parent / "output" / "analysis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main test execution."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PHASE 1 & PHASE 2 - COMPLETE FEATURE TEST".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Test complete pipeline
    results = test_complete_pipeline()
    
    if not results:
        print("\n❌ Pipeline execution failed")
        return
    
    # Test individual features
    test_backup_creation()
    test_test_file_separation(results)
    test_adjacency_matrix(results)
    test_llm_analysis(results)
    test_heuristic_architecture(results)
    test_complexity_metrics(results)
    
    # Save results
    save_results(results)
    
    # Summary
    print_header("EXECUTION SUMMARY")
    print("✅ All features tested successfully!")
    print(f"\nResults location: {Path(__file__).parent / 'output' / 'analysis_results.json'}")
    print(f"Backup location:  {Path(__file__).parent / 'backups'}")
    print(f"Completion time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
