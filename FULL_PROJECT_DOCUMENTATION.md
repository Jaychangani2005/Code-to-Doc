# Software Project Report

Project Title: Code-to-Doc Automated Documentation System

---

## 1.0 Introduction

### 1.1 Project Summary
Code-to-Doc is an automated software documentation pipeline that clones a source repository, performs static code analysis, generates module and function level documentation using LLMs, validates documentation quality, and optionally injects generated docstrings/comments back into source files.

It is implemented as a multi-phase Python workflow and currently demonstrates analysis over a Django application (Application) and a FastAPI plus Streamlit AI platform (EduMentor) available in the workspace.

### 1.2 Purpose
The purpose of this project is to reduce manual effort in code documentation by automatically producing structured technical artifacts such as architecture summaries, module documentation, function docstrings, QA reports, and repository-level reports.

### 1.3 Scope
In scope:

1. Repository cloning and inventory generation.
2. Python AST-based structural analysis.
3. LLM-assisted module documentation generation.
4. LLM-assisted function docstring generation.
5. Documentation QA and consistency validation.
6. Optional in-place comment/docstring injection with backups.

Out of scope:

1. Full semantic understanding equivalent to a human domain expert.
2. Guaranteed correctness of LLM outputs in all cases.
3. Complete runtime test automation for every analyzed external project.
4. Production deployment orchestration.

### 1.4 Objective (What It Can and Cannot Do)
Can do:

1. Analyze source structure from code and imports.
2. Generate markdown documentation for modules and functions.
3. Produce consolidated reports per phase.
4. Detect selected architectural metrics such as coupling and complexity.
5. Preserve backups before source modifications.

Cannot do (current limitations):

1. Reliably infer undocumented business logic intent.
2. Guarantee zero hallucination in LLM-generated text.
3. Fully cover non-Python language AST semantics in this current implementation.
4. Replace secure code review and test engineering practices.

### 1.5 Technology and Literature Review
Primary technologies:

1. Python for orchestration and analysis.
2. AST module for static parsing.
3. NetworkX for dependency graph processing.
4. LangChain for standardized prompt orchestration and model invocation.
5. LangGraph for state-based phase orchestration.
6. Groq with Llama models for generation and review tasks (via LangChain ChatGroq).
7. Rich for CLI UX and report rendering.

Supporting technologies in analyzed projects:

1. Django web stack (Application project).
2. FastAPI and Streamlit (EduMentor project).
3. FAISS and sentence-transformers for retrieval-augmented workflows.

Relevant technical references used by implementation:

1. Python AST documentation for syntax-tree traversal patterns.
2. Static analysis concepts: cyclomatic complexity, fan-in/fan-out, dependency graphs.
3. Prompt engineering principles for controlled technical documentation output.
4. Documentation quality metrics: completeness, consistency, clarity.

---

## 2.0 Project Management

### 2.1 Project Planning

#### 2.1.1 Project Development Approach and Justification
Approach selected: Iterative and incremental pipeline development.

Justification:

1. Each phase is independently testable and produces verifiable artifacts.
2. Failures can be isolated by phase and debugged faster.
3. LLM-dependent components can be improved without rewriting analysis stages.
4. Incremental delivery allows early value from Phase 1 and Phase 2 outputs.

#### 2.1.2 Project Effort and Time, Cost Estimation

Effort estimate by phase (indicative):

| Work Package | Estimated Person-Days |
|---|---:|
| Requirements and architecture design | 3 |
| Phase 1 and Phase 2 development | 5 |
| Phase 3 and Phase 4 generation workflow | 6 |
| Phase 5 and Phase 6 validation workflow | 3 |
| Injection module and reporting | 2 |
| Integration, testing, and documentation | 4 |
| Total | 23 |

Cost estimate model (sample for academic planning):

1. Team size: 3 members.
2. Average effort: 23 person-days.
3. Effective rate assumption: 3000 INR per person-day.
4. Estimated development cost: 69000 INR.
5. API and compute contingency: 10000 INR.
6. Total estimated project cost: 79000 INR.

#### 2.1.3 Roles and Responsibilities

| Role | Responsibilities |
|---|---|
| Project Lead | Scope control, milestones, integration decisions, risk tracking |
| Backend Engineer | Pipeline coding, AST analysis, report generation modules |
| AI/LLM Engineer | Prompt design, quality tuning, LLM integration and evaluation |
| QA Engineer | Test planning, report validation, defect tracking |
| Documentation Owner | Final report curation, formatting, evidence collection |

#### 2.1.4 Group Dependencies

1. Phase 2 depends on successful Phase 1 inventory output.
2. Phases 3 to 6 depend on availability of analysis_results.json.
3. LLM phases depend on valid API keys and network access.
4. Injection stage depends on Phase 4 report format consistency.

### 2.2 Project Scheduling (Gantt Chart/PERT/Network Chart)

Indicative Gantt-style schedule (weeks):

| Task | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---|---|---|---|---|---|
| Requirement study | X |  |  |  |  |  |
| Architecture and planning | X | X |  |  |  |  |
| Phase 1 and Phase 2 implementation |  | X | X |  |  |  |
| Phase 3 and Phase 4 implementation |  |  | X | X |  |  |
| Phase 5 and Phase 6 implementation |  |  |  | X | X |  |
| Integration and stabilization |  |  |  |  | X |  |
| Testing and final report |  |  |  |  | X | X |

PERT-style dependency chain:

1. A: Requirements.
2. B: Architecture design (after A).
3. C: Core analysis phases (after B).
4. D: LLM generation phases (after C).
5. E: QA and validation (after D).
6. F: Final documentation and closure (after E).

---

## 3.0 System Requirements Study

### 3.1 User Characteristics

Primary user groups:

1. Student developers who need project documentation quickly.
2. Team leads who require architecture summaries and risk hotspots.
3. Technical writers who convert generated outputs into formal reports.
4. Code reviewers and maintainers who need function-level understanding.

User skill assumptions:

1. Basic command-line usage.
2. Basic understanding of Python projects and repository structure.
3. Ability to configure environment variables for API keys.

### 3.2 Hardware and Software Requirements

Minimum hardware:

1. CPU: Dual-core 64-bit processor.
2. RAM: 8 GB minimum, 16 GB recommended.
3. Disk: 5 GB free storage for clones, outputs, and caches.

Software requirements:

1. OS: Windows/Linux/macOS.
2. Python: 3.10+ recommended.
3. Git client for repository cloning.
4. Internet for LLM API calls.
5. Required Python packages from requirements.txt.

### 3.3 Assumptions and Dependencies

Assumptions:

1. Target repository is accessible and cloneable.
2. Source files are mostly parseable text files.
3. API credentials are valid when LLM phases are executed.

Dependencies:

1. Third-party APIs for LLM generation.
2. Availability of chardet/networkx/groq ecosystem packages.
3. Stable report file format between phases.

---

## 4.0 System Analysis

### 4.1 Study of Current System
In typical academic or small-team projects, documentation is either missing, outdated, or manually written late in the cycle. Existing process usually relies on ad hoc README updates and individual comments, which are inconsistent across modules.

### 4.2 Problems and Weaknesses of Current System

1. High manual effort and low consistency.
2. Lack of traceable documentation quality checks.
3. Poor visibility into dependency and complexity hotspots.
4. Documentation generation is not integrated into development workflow.

### 4.3 Requirements of New System

#### 4.3.1 Functional Requirements

1. System shall accept a GitHub repository URL.
2. System shall clone and inventory repository content.
3. System shall parse Python modules and extract structural metadata.
4. System shall generate module-level markdown documentation.
5. System shall generate function-level docstrings and explanations.
6. System shall generate consolidated phase reports.
7. System shall evaluate docstring quality and consistency.
8. System shall support optional source comment/docstring injection with backup.

#### 4.3.2 Non Functional Requirements

1. Maintainability: phase-wise modular architecture.
2. Reliability: graceful failure handling with phase-level status.
3. Usability: clear terminal output and generated artifacts.
4. Performance: complete medium-size repository analysis in acceptable time.
5. Security: avoid exposing secrets in generated output.

### 4.4 Feasibility Study

#### 4.4.1 Contribution to Organizational Objectives
Yes. The system improves engineering productivity, accelerates onboarding, and increases documentation coverage quality for maintainability goals.

#### 4.4.2 Technology, Cost, and Schedule Feasibility
Yes. The solution uses mature libraries and a phased architecture. Estimated effort and cost remain practical for an academic project timeline.

#### 4.4.3 Integration Feasibility
Yes. Outputs are markdown/json and can integrate with existing repository workflows, CI reporting, and documentation portals.

### 4.5 Activity/Process in New System (Event Table)

| Event | Trigger | Input | Process | Output |
|---|---|---|---|---|
| E1: Start pipeline | User runs run_pipeline.py | GitHub URL | Validate inputs and workspace state | Pipeline initialized |
| E2: Clone and inventory | Phase 1 execution | URL, token optional | Clone repo, scan files, classify entries | analysis_results.json (base) |
| E3: Static analysis | Phase 2 execution | file inventory | AST parse, graph build, complexity compute | enriched analysis_results.json |
| E4: Module docs generation | Phase 3 execution | enriched analysis data | LLM prompt and markdown generation | generated_docs/*.md |
| E5: Function docs generation | Phase 4 execution | parsed modules | per-function docstring and complexity explanations | phase4_reports |
| E6: QA review | Phase 6 execution | phase4 reports | syntax and consistency checks plus LLM review | QA_REPORT.md |
| E7: Comment injection | inject stage | phase4 module docs | inject docstrings/comments and create backups | updated source and injection report |

### 4.6 Features of New System

1. End-to-end automated documentation pipeline.
2. Hybrid static analysis plus LLM generation approach.
3. Multi-level outputs: inventory, architecture, module docs, function docs, QA reports.
4. Recoverability via backup files before source updates.
5. Extensible phase-wise architecture.

### 4.7 Class Diagram (OO Approach)

Major classes and relationships:

1. Phase1, Phase2, Phase3, Phase4, Phase5, Phase6, InjectComments.
2. RepoMetadataExtractor, CodeParser, DependencyGraphBuilder.
3. GroqLLMClient wrappers use LangChain ChatGroq in generation and QA stages.
4. PromptBuilder, PythonDocstringAgent, ComplexLogicAgent.
5. DocstringExtractor, QAReviewer, CodeInjector.

Textual UML summary:

1. run_pipeline orchestrates phase objects.
2. run_pipeline (graph mode) orchestrates phase nodes using LangGraph state transitions.
3. Phase2 composes parser and graph builder.
4. Phase3 and Phase4 aggregate LLM client plus prompt/agent helpers.
5. inject_comments composes extractor plus injector.

### 4.8 System Activity (Use Case/Scenario)

Primary use cases:

1. UC1: Generate full documentation for a repository.
2. UC2: Generate function-level docstrings only.
3. UC3: Validate generated documentation quality.
4. UC4: Inject generated docstrings into source files.

Scenario for UC1:

1. User provides repository URL.
2. System executes phases 1 to 6.
3. System stores reports in output directory.
4. User reviews generated markdown and JSON artifacts.

### 4.9 Sequence Diagram (Textual)

Actor -> run_pipeline: provide repository URL

run_pipeline -> Phase1: clone and inventory

Phase1 -> output/analysis_results.json: write base metadata

run_pipeline -> Phase2: analyze structure and dependencies

Phase2 -> output/analysis_results.json: append architecture and complexity

run_pipeline -> Phase3/4/5/6: generate docs and QA

Phase modules -> output: write markdown reports

run_pipeline -> inject_comments: optional source update

inject_comments -> cloned_repo: write .bak and modified source files

### 4.10 Functions of System (Procedural View)

1. Repository preparation and cleanup.
2. Source scanning and file categorization.
3. AST extraction and metric computation.
4. LLM prompt generation and response persistence.
5. Quality validation and exception-safe reporting.

### 4.11 Context Diagram
External entities and system boundaries:

1. User/Developer submits repository URL and starts pipeline.
2. GitHub repository provides source content.
3. LLM service provides generated documentation text.
4. File system stores outputs and updated source files.

### 4.12 Data Flow Diagram (Level 0 and Level 1)

Level 0:

1. Input: Repository URL.
2. Process: Documentation pipeline.
3. Output: Reports, markdown docs, optional source updates.

Level 1 process breakdown:

1. P1: Clone and inventory.
2. P2: Analyze code structure.
3. P3: Generate module docs.
4. P4: Generate function docs.
5. P5: Validate and QA.
6. P6: Inject comments.

Data stores:

1. D1: cloned_repo.
2. D2: output/analysis_results.json.
3. D3: generated markdown reports.

### 4.13 Data Modeling

#### 4.13.1 Data Dictionary

| Data Item | Type | Description |
|---|---|---|
| github_url | string | User-provided repository URL |
| metadata.project_name | string | Name of analyzed project |
| metadata.primary_language | string | Dominant source language |
| file_inventory.inventory | object/list | Categorized file listing |
| phase2.parsed_modules | object | AST-extracted module structures |
| phase2.graph_analysis | object | dependency and coupling analysis |
| phase4_reports | markdown files | per-module function documentation |
| qa_report | markdown file | quality and consistency results |

#### 4.13.2 ER Diagram (Conceptual)
Conceptual entities in analysis domain:

1. Repository has many Files.
2. File has many Functions and Classes.
3. File imports many Dependencies.
4. Function has one or many GeneratedDoc entries.
5. Repository has many GeneratedReports.

### 4.14 Main Modules of New System

1. Pipeline Orchestrator: run_pipeline.py
2. Inventory Engine: phase1.py
3. Static Analysis Engine: phase2.py
4. Module Documentation Engine: phase3.py
5. Function Documentation Engine: phase4.py
6. README Generator: phase5.py
7. QA Engine: phase6.py
8. Source Injection Engine: inject_comments.py

### 4.15 Selection of Hardware and Software and Justification

Hardware selection justification:

1. 8 GB RAM baseline is adequate for medium project parsing and report generation.
2. SSD storage improves clone and I/O-intensive report generation speed.

Software selection justification:

1. Python ecosystem enables rapid, readable tooling development.
2. AST and NetworkX are mature and suitable for static analysis tasks.
3. LLM API allows high documentation throughput with low local compute burden.

---

## 5.0 System Design

Design style selected: Object-oriented modular pipeline design with phase encapsulation.

### 5.1 System Application Design

1. Controller layer in run_pipeline coordinates execution order.
2. Each phase encapsulates its own validation, processing, and output writing.
3. Shared data contract centered on output/analysis_results.json.

#### 5.1.1 Method Pseudo Code

```text
START
  read user input URL
  if --graph flag is set then route execution to run_pipeline_langgraph
  prepare workspace based on previous state
  run phase1(URL)
  if phase1 fail then STOP
  run phase2()
  if phase2 fail then STOP
  run phase3()   # non-critical
  run phase4()
  if phase4 fail then STOP
  run phase5()   # non-critical
  run phase6()   # non-critical
  run inject_comments()  # optional/non-critical
  save final state
END
```

### 5.2 Database Design/Data Structure Design

#### 5.2.1 Table and Relationship
Core pipeline uses file-based data storage, primarily JSON and markdown artifacts.

Primary internal structures:

1. metadata map.
2. file inventory arrays.
3. parsed_modules keyed by relative file paths.
4. dependency graph structures.

#### 5.2.2 Logical Description of Data

1. analysis_results.json is the canonical intermediate data store.
2. Markdown reports are derived views for human consumption.
3. Injection report tracks write operations and backups.

### 5.3 Input/Output and Interface Design

Input interface:

1. Command line execution with URL input.
2. Environment variables for API credentials.

Output interface:

1. Rich terminal progress and status tables.
2. Markdown artifacts under output directory.
3. JSON analysis artifact for machine-readable processing.

#### 5.3.1 State Transition/UML Diagram (Textual)

States:

1. Idle.
2. Initializing.
3. Analyzing.
4. Generating.
5. Validating.
6. Injecting.
7. Completed/Failed.

Transitions occur on per-phase success or failure.

#### 5.3.2 Samples of Forms, Reports and Interface

Representative artifacts:

1. output/analysis_results.json.
2. output/generated_docs/INDEX.md.
3. output/phase4_reports/MASTER_REPORT.md.
4. output/phase6_qa/QA_REPORT.md.
5. output/documented_code/INJECTION_REPORT.md.

### 5.4 System Structural Design
Structural decomposition:

1. Main controller module.
2. Parsing and analysis subsystem.
3. LLM documentation subsystem.
4. QA subsystem.
5. Injection subsystem.

---

## 6.0 Implementation Planning

### 6.1 Implementation Environment

1. Mode: Single-user local execution by default.
2. Interface type: Non-GUI command line execution with markdown outputs.
3. Workspace model: local folders cloned_repo and output.

### 6.2 Program/Modules Specification

| Module | Responsibility | Input | Output |
|---|---|---|---|
| run_pipeline.py | phase orchestration and cleanup | URL, env vars | phase statuses, reports |
| phase1.py | repository clone and inventory | URL | analysis_results.json (base) |
| phase2.py | AST and architecture analysis | base analysis | enriched analysis |
| phase3.py | module docs generation | enriched analysis | generated_docs |
| phase4.py | function docs generation | parsed modules | phase4_reports |
| phase5.py | README generation | analysis summary | output README |
| phase6.py | QA validation | phase4 reports | QA_REPORT |
| inject_comments.py | source injection | phase4 reports | source updates + report |

### 6.3 Security Features

1. Backup creation before in-place source modifications.
2. Environment-variable based API key usage.
3. Controlled file filtering to avoid unwanted binary processing.

### 6.4 Coding Standards

1. PEP 8 style conventions for Python code.
2. Modular class-based design for maintainability.
3. Structured logging and explicit exception handling.
4. Clear docstrings for major classes and methods.

---

## 7.0 Testing

### 7.1 Testing Plan

1. Unit-level tests for helper functions and parsers.
2. Integration tests for cross-phase contracts.
3. End-to-end test using a sample repository URL.
4. Regression checks on report structure and expected keys.

### 7.2 Testing Strategy

1. Bottom-up validation of each phase.
2. Contract testing on analysis_results.json schema.
3. Fault injection for missing API keys and bad repository URLs.

### 7.3 Testing Methods

1. Functional black-box testing for CLI behavior.
2. White-box testing for parsing and complexity computation.
3. Static verification of generated report existence and content patterns.

### 7.4 Test Suites Design

Suite A: Input and setup validation.

Suite B: Static analysis extraction correctness.

Suite C: LLM generation pipeline behavior.

Suite D: QA and injection behavior.

### 7.5 Test Cases

#### 7.5.1 Purpose
To verify that each phase produces valid outputs and that failure behavior is controlled.

#### 7.5.2 Required Input

1. Public GitHub repository URL.
2. Optional API keys for LLM phases.

#### 7.5.3 Expected Result

| Test Case | Required Input | Expected Result |
|---|---|---|
| TC01 Valid URL, no LLM key | URL only | Phase 1 and 2 succeed, LLM phases report key-related failure |
| TC02 Valid URL with key | URL + GROQ_API_KEY | Phases 1 to 6 complete and output docs generated |
| TC03 Invalid URL | bad URL | Clone fails with controlled error and stop |
| TC04 Re-run same URL | same URL twice | cloned_repo reused, output refreshed |
| TC05 Injection with reports | phase4 reports available | backup files created and injection report generated |
| TC06 Graph mode execution | URL + --graph | pipeline runs using LangGraph with equivalent phase outcomes |

---

## 8.0 Conclusion and Discussion

### 8.1 Self Analysis of Project Viabilities
The project is technically viable and academically strong because it combines software engineering practices (modular architecture, static analysis, QA reporting) with practical AI integration for high-value automation.

### 8.2 Problem Encountered and Possible Solutions

Problems encountered:

1. LLM dependency and network/API constraints.
2. Inconsistency risk in generated text format.
3. Source injection mismatch if report patterns differ.

Possible solutions:

1. Add retries and fallback model strategy.
2. Use stricter output schemas and parser guards.
3. Add dry-run injection mode with diff preview.

### 8.3 Summary of Project Work
The project successfully establishes a complete repository-to-documentation automation pipeline with phase-wise outputs, architecture insights, function-level documentation generation, and post-generation quality checks. It demonstrates practical applicability for student projects and small engineering teams.

---

## 9.0 Limitation and Future Enhancement

Current limitations:

1. Heavy reliance on external LLM service availability.
2. Primary deep analysis focus is Python code.
3. Generated documentation may still require human editorial review.
4. Current injection stage may not cover all edge cases in dynamic code patterns.

Future enhancements:

1. Add support for JavaScript/TypeScript and Java AST pipelines.
2. Add CI integration for automatic documentation refresh on commits.
3. Add quality score dashboards and trend tracking.
4. Add policy checks for security smells and maintainability gates.
5. Provide HTML/PDF export formats for direct academic submission.

