# Task Orchestration System Design

> **Status note (2026):** the framework-comparison sections of this
> document are now historical. We rolled our own task runner instead of
> using Snakemake/Dagster/etc. See
> [`pipeline-runner.md`](pipeline-runner.md) for the current
> implementation. The architectural ideas around observability, build
> tracking, and chunk composition below are still relevant as a roadmap.

## Goals

### Primary Goals

1. **Replace Perl-based orchestration** with modern Python-based workflow management
2. **Enable parallelism** at multiple levels:
   - Per-file: Process multiple audio files concurrently
   - Per-iteration: Run multiple training iterations in parallel
   - Per-step: Execute independent pipeline stages concurrently
3. **Maintain compatibility** with existing C CLI programs while adding Python orchestration
4. **Support both sequential and independent dependencies** in training pipelines
5. **Scale from local to distributed** execution (local multi-core → GCE/cluster)

### Secondary Goals

1. **Library-first design**: All business logic in `pstrain.lib`, orchestration is thin wrapper
2. **JSON-serializable outputs**: All pipeline results return JSON for web/API integration
3. **Robust incremental execution**: Only rebuild what changed, skip completed work reliably
4. **Resume capability**: Restart from any point, never redo completed computation
5. **Stop and inspect**: Gracefully stop workflows, inspect current state, robustly restart
6. **Easy debugging**: Clear error messages, dry-run mode, logging
7. **Extensibility**: Easy to add new pipeline steps and workflows

## Requirements

### Functional Requirements

1. **Execute C programs** (sphinx_fe, bw, norm, etc.) as pipeline steps
2. **Call Python functions** from `pstrain.lib` as pipeline steps
3. **Handle file dependencies**: Output of step N is input to step N+1
4. **Support batch processing**: Process multiple files with wildcards
5. **Configuration management**: YAML/JSON config files for pipeline parameters
6. **Error handling**: Retry failed steps, clear error messages
7. **Progress tracking**: Show which steps are running/completed
8. **Build tracking**: Track all builds, their status (success/failure), timestamps, and metadata
9. **Logging**: Comprehensive logging at multiple levels (debug, info, warning, error)
10. **Failure analysis**: Track what failed, why it failed, and how to fix it
11. **Incremental execution**: Robustly skip completed steps, never redo work unnecessarily
12. **Resume/restart**: Resume from any point after interruption, handle partial builds
13. **Stop workflows**: Gracefully stop running workflows (Ctrl+C, SIGTERM, CLI command)
14. **Inspect state**: Query current workflow state (what's running, what's done, what's pending)
15. **Robust restart**: Restart interrupted workflows from last checkpoint, never redo completed work
16. **Product tracking**: Track all files/products created by each step/phase
17. **Phase cleanup**: Clean out products from specific phases before restarting
18. **Output validation**: Verify outputs are valid before skipping computation
14. **Change detection**: Detect when inputs/config changed vs when they're identical
15. **Profiling and observability**: Comprehensive performance profiling, metrics, and tracing
16. **Resource monitoring**: Track CPU, memory, disk, network usage per step
17. **Performance analysis**: Identify bottlenecks, slow steps, resource constraints
18. **Ad-hoc step execution**: Run individual steps from CLI, track them separately
19. **Unified tracking**: Manual steps and pipeline steps tracked in same system
20. **Step reuse**: Pipeline can detect and reuse manually executed steps
21. **Trace inspection**: Honeycomb-style trace inspection with rich attributes and queryable spans
22. **Call chain visualization**: See complete execution flow with parent-child relationships
23. **High-cardinality data**: Rich span attributes for filtering, grouping, and analysis
24. **Reusable workflow chunks**: Modular, composable workflow components
25. **Simple and composed chunks**: Both atomic steps and multi-step workflows as reusable units
26. **Easy configuration**: Simple, declarative configuration system for defining and composing chunks

### Non-Functional Requirements

1. **Performance**: Efficient parallel execution, minimal overhead
2. **Reliability**: Handle failures gracefully, resume from checkpoints
3. **Maintainability**: Clear code structure, good documentation
4. **Portability**: Works on macOS, Linux (Windows optional)
5. **Observability**: Comprehensive logging, metrics, tracing, and profiling
6. **Auditability**: Complete history of builds, changes, and outcomes
7. **Robustness**: Never redo completed work, handle interruptions gracefully
8. **Correctness**: Verify outputs are valid, detect corruption or incomplete files
9. **Performance visibility**: Detailed profiling of all operations, identify bottlenecks
10. **Resource awareness**: Track and optimize resource usage across all steps

## Design Decisions

### Framework Evaluation

**IMPORTANT:** Given our extensive requirements (concurrency, incremental execution, observability, composition, profiling, logging, build tracking), we should evaluate using an existing framework rather than building everything ourselves.

**See:** `docs/design/framework-evaluation.md` for detailed comparison.

**Top Candidates:**
1. **Dagster** - Strong asset tracking (incremental execution), excellent observability, composition
2. **Prefect 3.0** - Modern workflow orchestration, excellent observability, composition
3. **Snakemake** ⭐ **SELECTED** - Excellent for file-based workflows, but limited observability (would need to build)

**Decision:** **Snakemake** - Chosen for simplicity, quick start, and full control. Will build observability, chunks, and profiling as needed. Can migrate to Dagster later if service mode becomes critical.

**See:** `docs/design/snakemake-vs-dagster.md` for detailed decision rationale.

### Build System Choice: Snakemake (Current)

**Rationale:**
- Excellent parallelism support (`-j N` for local, cluster support for distributed)
- Automatic dependency resolution based on file inputs/outputs
- Handles both sequential and independent dependencies naturally
- Wildcards enable batch processing (process all `{sample}.wav` files)
- Scales from local to distributed (GCE via Kubernetes)
- Python-based DSL (familiar, can call Python functions)
- Widely used in scientific computing (proven for similar use cases)

**Limitations:**
- Limited observability (would need to build OpenTelemetry integration)
- No built-in chunk/composition system (would need to build)
- No built-in profiling (would need to build)
- Standalone execution would need to be built

**Alternatives Considered:**
- **Dagster**: Has asset tracking, observability, composition built-in - **EVALUATED, DEFERRED** (can migrate later if service mode needed)
- **Prefect**: Has observability, composition built-in - **EVALUATED, DEFERRED**
- **doit**: Good but less automatic dependency resolution
- **invoke**: Too manual, no built-in parallelism
- **luigi/temporal**: More complex setup, overkill for initial needs

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│  (CLI: pstrain train, pstrain extract-features, etc.)          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Snakemake Workflows                         │
│  (Snakefile: defines pipeline steps and dependencies)    │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼────┐ ┌────▼────┐ ┌────▼──────────┐
│  pstrain.lib   │ │ C CLI   │ │  Config       │
│  (Python   │ │ Programs │ │  Files       │
│   API)     │ │          │ │  (YAML)       │
└────────────┘ └──────────┘ └───────────────┘
```

### Key Components

1. **Snakefile**: Defines pipeline rules and dependencies
2. **pstrain.lib.tasks**: Python functions that wrap C programs and provide high-level API
3. **Config files**: YAML configuration for pipeline parameters
4. **CLI integration**: `pstrain.cli` commands that invoke Snakemake workflows
5. **Step registry**: Registry of available steps (from workflows and manual registration)
6. **Step executor**: Unified execution engine for pipeline and standalone steps
7. **Chunk system**: Reusable workflow components (simple and composed)
8. **Chunk registry**: Registry of available chunks with metadata
9. **Chunk composer**: System for composing chunks into larger workflows
10. **Build tracker**: Database/storage for tracking builds, status, and metadata
11. **Logging system**: Structured logging with multiple levels and output formats
12. **Failure analysis**: Tools for analyzing and reporting failures
13. **Observability system**: OpenTelemetry-based tracing, metrics, and profiling
14. **Resource monitor**: Track CPU, memory, disk, network usage per step

## Implementation Plan

### Phase 1: Foundation (MVP)

#### TODO 1.1: Set up Snakemake infrastructure
- [ ] Add `snakemake` to `pyproject.toml` optional dependencies
- [ ] Create `workflows/` directory structure
- [ ] Create basic `Snakefile` template
- [ ] Add `make workflows` target to build system
- [ ] Document Snakemake setup in `docs/`

#### TODO 1.2: Create Python task wrappers
- [ ] Create `pstrain/lib/tasks.py` module
- [ ] Implement wrapper functions for key C programs:
  - [ ] `extract_features()` - wraps sphinx_fe
  - [ ] `normalize()` - wraps norm
  - [ ] `train_bw()` - wraps bw (Baum-Welch)
  - [ ] `init_gaussians()` - wraps init_gau
- [ ] Each function should:
  - Accept Python parameters
  - Call C program via subprocess or direct C bindings
  - Return JSON-serializable results
  - Handle errors gracefully

#### TODO 1.3: Chunk system and configuration
- [ ] **Design chunk configuration schema**:
  - [ ] Simple chunks (single step)
  - [ ] Composed chunks (multiple steps)
  - [ ] Chunk parameters and inputs/outputs
  - [ ] Chunk metadata (name, description, version)
- [ ] **Implement chunk registry**:
  - [ ] `pstrain.lib.chunks` module
  - [ ] Chunk discovery (from config files)
  - [ ] Chunk validation
  - [ ] Chunk dependency resolution
- [ ] **Create example chunks**:
  - [ ] Simple: `extract_features`, `normalize`, `init_gaussians`
  - [ ] Composed: `feature_pipeline` (extract + normalize)
  - [ ] Composed: `training_pipeline` (init + train)
- [ ] **Configuration system**:
  - [ ] YAML-based chunk definitions
  - [ ] Chunk composition syntax
  - [ ] Parameter passing between chunks
  - [ ] Default values and validation

#### TODO 1.4: Basic workflow example
- [ ] Create `workflows/basic_training/Snakefile`
- [ ] Use chunk system to compose workflow:
  1. Extract features from audio (chunk)
  2. Normalize features (chunk)
  3. Initialize Gaussians (chunk)
  4. Train model (chunk)
- [ ] Create example config file using chunks
- [ ] Test with sample data

#### TODO 1.5: CLI integration
- [ ] Add `pstrain train` command to `pstrain/cli.py`
- [ ] Add `pstrain workflow` command for running workflows
- [ ] Add `pstrain step` command for running individual steps:
  - [ ] `pstrain step extract-features --input audio.wav --output features.mfc`
  - [ ] `pstrain step normalize --input accum.acc --output norm.norm`
  - [ ] `pstrain step train --config train.yaml`
- [ ] Integrate with Snakemake programmatically
- [ ] Add `--dry-run` flag
- [ ] Add `--parallel N` flag
- [ ] Track ad-hoc step executions in build tracker
- [ ] Allow steps to be run standalone or as part of pipeline

### Phase 2: Parallelism and Batch Processing

#### TODO 2.1: Per-file parallelism
- [ ] Implement wildcard-based rules for processing multiple files
- [ ] Example: Process all `data/{sample}.wav` → `features/{sample}.mfc`
- [ ] Test with multiple audio files
- [ ] Verify parallel execution (`snakemake -j 4`)

#### TODO 2.2: Per-iteration parallelism
- [ ] Support multiple training iterations
- [ ] Example: `models/{iteration}/model.mdef` for iteration 1..N
- [ ] Aggregate results from multiple iterations
- [ ] Test parallel iteration execution

#### TODO 2.3: Per-step parallelism
- [ ] Identify independent pipeline steps
- [ ] Configure rules to run in parallel when dependencies allow
- [ ] Test complex dependency graphs

### Phase 3: Advanced Features

#### TODO 3.1: Chunk configuration and composition system
- [ ] **Chunk definition format**:
  - [ ] YAML schema for chunk definitions
  - [ ] Support for simple chunks (single step)
  - [ ] Support for composed chunks (multiple steps/chunks)
  - [ ] Parameter definitions (inputs, outputs, config)
  - [ ] Dependency specification
- [ ] **Chunk registry implementation**:
  - [ ] Load chunks from config files
  - [ ] Validate chunk definitions
  - [ ] Resolve chunk dependencies
  - [ ] Cache chunk metadata
- [ ] **Chunk composition**:
  - [ ] Compose chunks into workflows
  - [ ] Parameter passing between chunks
  - [ ] Input/output mapping
  - [ ] Dependency resolution
- [ ] **Configuration management**:
  - [ ] Design YAML config schema for workflows
  - [ ] Implement config loading in `pstrain.lib.tasks`
  - [ ] Support config inheritance/merging
  - [ ] Validate config files
  - [ ] Support chunk references in config

#### TODO 3.2: Robust incremental execution, stop, inspect, and restart
- [ ] **File-based dependency tracking**:
  - [ ] Use file timestamps and checksums for change detection
  - [ ] Store file metadata (size, mtime, checksum) in build tracker
  - [ ] Compare current file state vs last successful build
  - [ ] Only mark step as "needs rerun" if inputs changed
- [ ] **Output validation**:
  - [ ] Verify output files exist and are non-empty
  - [ ] Check file integrity (checksums, magic numbers)
  - [ ] Validate file format (e.g., MFC files are valid)
  - [ ] Store output checksums in build tracker
- [ ] **Stop workflows**:
  - [ ] Handle Ctrl+C gracefully (SIGINT)
  - [ ] Handle SIGTERM for graceful shutdown
  - [ ] `pstrain stop <build-id>` CLI command
  - [ ] Stop scheduling new jobs, let current jobs finish (graceful)
  - [ ] Force stop option (kill all jobs immediately)
  - [ ] Update build status to "cancelled" or "interrupted"
  - [ ] Save checkpoint before stopping
- [ ] **Inspect state**:
  - [ ] `pstrain status <build-id>` - Show current workflow state
  - [ ] `pstrain list-builds` - List all builds with status
  - [ ] Query what's running, what's done, what's pending
  - [ ] Show step-by-step progress
  - [ ] Show resource usage (CPU, memory, time)
  - [ ] Show logs for running/completed steps
  - [ ] Integration with Snakemake's dry-run mode
- [ ] **Product tracking**:
  - [ ] Track all output files created by each step
  - [ ] Store product list in build tracker (step outputs table)
  - [ ] Track intermediate files (not just final outputs)
  - [ ] Track files by phase/workflow stage
  - [ ] Support querying products by step, phase, or build
  - [ ] Track file metadata (path, size, checksum, creation time)
- [ ] **Phase cleanup**:
  - [ ] `pstrain clean <build-id> --phase <phase-name>` - Clean products from specific phase
  - [ ] `pstrain clean <build-id> --step <step-name>` - Clean products from specific step
  - [ ] `pstrain clean <build-id> --all` - Clean all products from build
  - [ ] List products before cleaning (dry-run mode)
  - [ ] Verify products are tracked before deletion
  - [ ] Handle partial cleanup (some files missing is OK)
  - [ ] Update build tracker after cleanup (mark products as deleted)
  - [ ] Support cleaning before restart (clean phase N, restart from phase N)
- [ ] **Robust restart**:
  - [ ] Detect interrupted builds on startup
  - [ ] Query build tracker for last successful step
  - [ ] Resume from last checkpoint, not from beginning
  - [ ] Handle partial outputs (clean up or reuse)
  - [ ] `pstrain restart <build-id>` CLI command
  - [ ] `pstrain restart <build-id> --from-phase <phase>` - Restart from specific phase (after cleanup)
  - [ ] Validate outputs before resuming (don't trust incomplete files)
  - [ ] Use Snakemake's `--rerun-incomplete` flag
  - [ ] Mark incomplete files for rerun
  - [ ] Integration with phase cleanup (clean phase, then restart)
- [ ] **Change detection**:
  - [ ] Track config file changes (hash config files)
  - [ ] Track input file changes (checksums)
  - [ ] Track code changes (if using Python functions)
  - [ ] Only rebuild when dependencies actually changed
- [ ] **State persistence**:
  - [ ] Save build state after each step completion
  - [ ] Atomic state updates (prevent corruption)
  - [ ] State recovery on restart
  - [ ] Checkpoint system (save state periodically)
- [ ] **Error handling and recovery**:
  - [ ] Implement retry logic for failed steps
  - [ ] Clear error messages with context
  - [ ] Mark failed steps for retry, don't redo successful ones

#### TODO 3.3: Build tracking system
- [ ] Design build tracking schema (SQLite/JSON database)
- [ ] Implement `pstrain.lib.tracker` module:
  - [ ] `start_build()` - Record build start
  - [ ] `record_step()` - Track individual step execution
  - [ ] `finish_build()` - Record build completion/failure
  - [ ] `query_builds()` - Query build history
- [ ] Store build metadata:
  - [ ] Build ID (UUID)
  - [ ] Timestamp (start, end, each step)
  - [ ] Status (running, success, failed, cancelled)
  - [ ] Config used
  - [ ] Steps executed with status
  - [ ] Error messages and stack traces
  - [ ] Resource usage (CPU, memory, time)
- [ ] Integrate with Snakemake (hooks/callbacks)

#### TODO 3.4: Logging system
- [ ] Design logging architecture:
  - [ ] Structured logging (JSON format option)
  - [ ] Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - [ ] Per-step log files
  - [ ] Build-level log aggregation
  - [ ] Log rotation and retention
- [ ] Implement `pstrain.lib.logging` module:
  - [ ] `setup_logging()` - Configure logging
  - [ ] `get_logger()` - Get logger for module/step
  - [ ] Context managers for step logging
- [ ] Log formats:
  - [ ] Human-readable (console)
  - [ ] Structured JSON (for parsing/analysis)
  - [ ] Detailed (file logs with full context)
- [ ] Integration points:
  - [ ] C program stdout/stderr capture
  - [ ] Python exception logging
  - [ ] Snakemake execution logs
  - [ ] Build tracker events

#### TODO 3.5: Failure analysis and reporting
- [ ] Implement failure analysis:
  - [ ] Categorize failures (timeout, error code, exception, etc.)
  - [ ] Extract error patterns
  - [ ] Suggest fixes based on error type
- [ ] Build reports:
  - [ ] Summary report (JSON/HTML)
  - [ ] Failed steps with error details
  - [ ] Success rate statistics
  - [ ] Performance metrics
- [ ] Query and filter:
  - [ ] Find all failed builds
  - [ ] Find builds with specific error
  - [ ] Compare builds
  - [ ] Trend analysis

#### TODO 3.6: Progress tracking and reporting
- [ ] Integrate with Snakemake's progress reporting
- [ ] Add custom progress callbacks
- [ ] Generate pipeline reports (JSON/HTML)
- [ ] Show estimated time remaining
- [ ] Real-time status updates
- [ ] Integration with build tracker

#### TODO 3.7: OpenTelemetry integration and profiling
- [ ] Set up OpenTelemetry SDK:
  - [ ] Add `opentelemetry-api`, `opentelemetry-sdk` to dependencies
  - [ ] Add `opentelemetry-exporter-otlp` for Honeycomb/OTLP export
  - [ ] Configure exporters (console, OTLP, file, Honeycomb)
  - [ ] Set up trace context propagation
- [ ] **Tracing with rich attributes** (Honeycomb-style):
  - [ ] Create spans for each build
  - [ ] Create spans for each step
  - [ ] Create spans for C program execution
  - [ ] Create spans for Python function calls
  - [ ] **High-cardinality span attributes**:
    - [ ] Build metadata (build_id, workflow_name, config_hash)
    - [ ] Step metadata (step_name, rule_name, execution_type)
    - [ ] File information (input_files, output_files, file_sizes, checksums)
    - [ ] Resource usage (cpu_time, memory_peak, disk_io)
    - [ ] Performance metrics (duration, throughput)
    - [ ] Error information (error_type, error_message, retry_count)
    - [ ] Configuration (config values, parameters)
    - [ ] Environment (hostname, python_version, snakemake_version)
  - [ ] Track parent-child relationships (build → step → sub-operation)
  - [ ] Add span events for important milestones
  - [ ] Add span links for related operations
  - [ ] Export traces to file/console/OTLP endpoint/Honeycomb
- [ ] **Metrics**:
  - [ ] Step duration (histogram)
  - [ ] Build duration (histogram)
  - [ ] Success/failure rates (counter)
  - [ ] Resource usage (CPU, memory, disk I/O)
  - [ ] File sizes processed
  - [ ] Throughput (files/second, steps/second)
- [ ] **Profiling**:
  - [ ] CPU profiling (cProfile integration)
  - [ ] Memory profiling (memory_profiler or tracemalloc)
  - [ ] I/O profiling (track file read/write operations)
  - [ ] Per-step resource usage
  - [ ] Identify hot spots and bottlenecks
- [ ] **Integration points**:
  - [ ] Wrap C program calls with tracing
  - [ ] Wrap Python functions with tracing
  - [ ] Integrate with Snakemake execution
  - [ ] Store profiling data in build tracker
- [ ] **Trace inspection and analysis** (Honeycomb-style):
  - [ ] **Query interface**:
    - [ ] Query traces by build_id, step_name, status, etc.
    - [ ] Filter by span attributes (high-cardinality filtering)
    - [ ] Group by any attribute (workflow, step, config, etc.)
    - [ ] Aggregate metrics (avg duration, p95, p99, etc.)
    - [ ] Compare traces (side-by-side comparison)
  - [ ] **Visualization**:
    - [ ] Call chain visualization (parent-child span relationships)
    - [ ] Timeline view (Gantt chart of spans)
    - [ ] Waterfall view (see parallel execution)
    - [ ] Flame graph (CPU time breakdown)
    - [ ] Service map (dependency graph)
  - [ ] **Analysis tools**:
    - [ ] Generate performance reports
    - [ ] Visualize traces (Honeycomb, Jaeger, Zipkin, or local viewer)
    - [ ] Identify slow steps (query by duration)
    - [ ] Resource usage trends (query by resource metrics)
    - [ ] Compare builds (performance regression detection)
    - [ ] Find patterns (e.g., "all failed steps have memory > X")
    - [ ] Debug issues (trace through call chain to find root cause)
  - [ ] **Export and integration**:
    - [ ] Export traces to Honeycomb (OTLP)
    - [ ] Export to local file (JSON, OTLP format)
    - [ ] Export to Jaeger/Zipkin (if needed)
    - [ ] API for programmatic trace access

#### TODO 3.8: Resource monitoring
- [ ] **Per-step resource tracking**:
  - [ ] CPU time (user, system, total)
  - [ ] Memory usage (peak, average, final)
  - [ ] Disk I/O (bytes read/written)
  - [ ] Network I/O (if applicable)
  - [ ] GPU usage (if applicable, future)
- [ ] **Real-time monitoring**:
  - [ ] Monitor resource usage during step execution
  - [ ] Alert on resource limits (memory, disk)
  - [ ] Track resource usage over time
- [ ] **Storage**:
  - [ ] Store resource metrics in build tracker
  - [ ] Export to time-series database (optional)
  - [ ] Generate resource usage reports

### Phase 4: Integration and Polish

#### TODO 4.1: Chunk system implementation
- [ ] **Chunk loader**:
  - [ ] Load chunk definitions from YAML files
  - [ ] Validate chunk schemas
  - [ ] Resolve chunk dependencies
  - [ ] Cache loaded chunks
- [ ] **Chunk executor**:
  - [ ] Execute simple chunks (single step)
  - [ ] Execute composed chunks (recursive)
  - [ ] Handle parameter substitution
  - [ ] Map inputs/outputs between chunks
- [ ] **Chunk registry**:
  - [ ] `pstrain.lib.chunks` module
  - [ ] `register_chunk()` - Register chunk definition
  - [ ] `get_chunk()` - Get chunk by name
  - [ ] `list_chunks()` - List all available chunks
  - [ ] `validate_chunk()` - Validate chunk definition
- [ ] **Chunk composition**:
  - [ ] Compose chunks into workflows
  - [ ] Parameter passing and substitution
  - [ ] Input/output mapping
  - [ ] Dependency resolution
- [ ] **CLI commands**:
  - [ ] `pstrain chunk list` - List available chunks
  - [ ] `pstrain chunk show <name>` - Show chunk definition
  - [ ] `pstrain chunk validate <file>` - Validate chunk file
  - [ ] `pstrain chunk run <name>` - Run chunk standalone

#### TODO 4.2: Standalone step execution
- [ ] **CLI commands for individual steps**:
  - [ ] `pstrain step extract-features` - Run feature extraction
  - [ ] `pstrain step normalize` - Run normalization
  - [ ] `pstrain step train` - Run training step
  - [ ] `pstrain step list` - List available steps
  - [ ] `pstrain step status <step-id>` - Check step status
- [ ] **Step execution logic**:
  - [ ] Create standalone build record for tracking
  - [ ] Execute step with same logic as pipeline
  - [ ] Track execution in build tracker
  - [ ] Support same flags as pipeline (--dry-run, --config, etc.)
- [ ] **Integration with pipelines**:
  - [ ] Pipeline checks for existing standalone steps
  - [ ] Reuse standalone step outputs if valid
  - [ ] Link standalone steps to pipeline builds when used
- [ ] **Step discovery**:
  - [ ] Auto-discover available steps from workflows
  - [ ] Register steps in step registry
  - [ ] Generate CLI commands dynamically

#### TODO 4.3: Integration with pstrain.lib
- [ ] Ensure all `pstrain.lib` functions are workflow-compatible
- [ ] Ensure all `pstrain.lib` functions are CLI-callable
- [ ] Add workflow-specific utilities to `pstrain.lib`
- [ ] Document workflow API
- [ ] Document CLI step API

#### TODO 4.4: Documentation
- [ ] Write workflow user guide
- [ ] Document all available workflows
- [ ] Document CLI step commands
- [ ] Add examples for common use cases
- [ ] Add examples for standalone step execution
- [ ] Create tutorial for writing custom workflows
- [ ] Document how standalone steps integrate with pipelines

#### TODO 4.4: Trace inspection and query interface
- [ ] **Trace query API**:
  - [ ] `pstrain.lib.traces.query()` - Query traces by attributes
  - [ ] `pstrain.lib.traces.get_trace()` - Get full trace by trace_id
  - [ ] `pstrain.lib.traces.get_call_chain()` - Get call chain for build
  - [ ] `pstrain.lib.traces.compare()` - Compare multiple traces
- [ ] **CLI commands**:
  - [ ] `pstrain traces query` - Query traces from CLI
  - [ ] `pstrain traces show <trace-id>` - Show trace details
  - [ ] `pstrain traces compare <trace-id1> <trace-id2>` - Compare traces
  - [ ] `pstrain traces export` - Export traces to Honeycomb/file
- [ ] **Visualization**:
  - [ ] Generate trace visualization (HTML/JSON)
  - [ ] Call chain diagram
  - [ ] Timeline view
  - [ ] Waterfall view
- [ ] **Honeycomb integration**:
  - [ ] Configure Honeycomb exporter
  - [ ] Export traces to Honeycomb dataset
  - [ ] Link traces in build tracker to Honeycomb
  - [ ] Query Honeycomb from CLI (optional)

#### TODO 4.5: Testing
- [ ] Unit tests for `pstrain.lib.tasks` functions
- [ ] Integration tests for workflows
- [ ] Test parallel execution correctness
- [ ] Test error handling and recovery
- [ ] Test trace generation and export
- [ ] Test trace query interface

### Phase 5: Distributed Execution (Future)

#### TODO 5.1: GCE/Kubernetes support
- [ ] Research Snakemake Kubernetes executor
- [ ] Design GCE deployment strategy
- [ ] Implement cloud config
- [ ] Test distributed execution

#### TODO 5.2: Resource management
- [ ] Define resource requirements for each step
- [ ] Implement resource allocation
- [ ] Monitor resource usage

## File Structure

```
pstrain/
├── lib/
│   ├── tasks.py          # Python task wrappers
│   ├── chunks.py         # Chunk registry and composition
│   ├── tracker.py        # Build tracking
│   ├── logging.py        # Logging utilities
│   ├── observability.py  # OpenTelemetry setup and utilities
│   ├── profiling.py      # Profiling helpers
│   ├── traces.py         # Trace query and inspection utilities
│   └── ...
├── chunks/
│   ├── __init__.py
│   ├── simple/           # Simple (atomic) chunks
│   │   ├── extract_features.yaml
│   │   ├── normalize.yaml
│   │   ├── init_gaussians.yaml
│   │   └── train_bw.yaml
│   └── composed/         # Composed (multi-step) chunks
│       ├── feature_pipeline.yaml
│       ├── training_pipeline.yaml
│       └── full_training.yaml
├── workflows/
│   ├── __init__.py
│   ├── basic_training/
│   │   ├── Snakefile
│   │   └── config.yaml
│   ├── advanced_training/
│   │   ├── Snakefile
│   │   └── config.yaml
│   └── common/
│       └── utils.py      # Shared workflow utilities
├── builds/               # Build tracking database
│   ├── builds.db         # SQLite database (or JSON files)
│   ├── logs/             # Build logs
│   │   └── {build_id}/
│   │       ├── build.log
│   │       └── steps/
│   │           └── {step_name}.log
│   └── traces/           # OpenTelemetry traces
│       └── {build_id}/
│           └── trace.jsonl  # OTLP JSON format
└── cli.py                # CLI commands for workflows

docs/
├── workflows/
│   ├── getting-started.md
│   ├── writing-workflows.md
│   └── examples.md
└── ...
```

## Chunk System Design

### Chunk Types

**1. Simple Chunks** (atomic operations):
- Single step execution
- Direct mapping to a C program or Python function
- Example: `extract_features`, `normalize`, `init_gaussians`

**2. Composed Chunks** (workflow components):
- Multiple steps or chunks combined
- Can include other composed chunks
- Example: `feature_pipeline`, `training_pipeline`, `full_training`

### Chunk Definition Format

```yaml
# chunks/extract_features.yaml
chunk:
  name: "extract_features"
  version: "1.0"
  type: "simple"  # or "composed"
  description: "Extract acoustic features from audio file"

  # For simple chunks
  step:
    program: "sphinx_fe"  # C program name
    # or
    function: "pstrain.lib.tasks.extract_features"  # Python function

  inputs:
    - name: "audio_file"
      type: "file"
      required: true
      description: "Input audio file (WAV format)"

  outputs:
    - name: "features_file"
      type: "file"
      description: "Output features file (MFC format)"

  parameters:
    - name: "sample_rate"
      type: "int"
      default: 16000
      description: "Audio sample rate"
    - name: "argfile"
      type: "file"
      default: "config/feat.params"
      description: "Feature extraction parameter file"

  config:
    sphinx_fe:
      argfile: "config/feat.params"
      output_format: "mfc"
```

```yaml
# chunks/feature_pipeline.yaml
chunk:
  name: "feature_pipeline"
  version: "1.0"
  type: "composed"
  description: "Complete feature extraction and normalization pipeline"

  # Composed from other chunks
  steps:
    - chunk: "extract_features"
      inputs:
        audio_file: "{audio_file}"
      outputs:
        features_file: "{workdir}/features.mfc"
      parameters:
        sample_rate: "{sample_rate}"

    - chunk: "normalize"
      inputs:
        input_file: "{workdir}/features.mfc"
      outputs:
        output_file: "{workdir}/features.norm"
      parameters:
        method: "{normalization_method}"

  inputs:
    - name: "audio_file"
      type: "file"
      required: true

  outputs:
    - name: "normalized_features"
      type: "file"
      description: "Normalized feature file"

  parameters:
    - name: "sample_rate"
      type: "int"
      default: 16000
    - name: "normalization_method"
      type: "string"
      default: "cmn"
    - name: "workdir"
      type: "path"
      default: "work"
```

### Workflow Configuration Using Chunks

```yaml
# workflows/basic_training/config.yaml
workflow:
  name: "basic_training"
  version: "1.0"

  # Use chunks to compose workflow
  chunks:
    - name: "feature_pipeline"
      inputs:
        audio_file: "data/{sample}.wav"
      outputs:
        normalized_features: "work/{sample}.norm"
      parameters:
        sample_rate: 16000
        normalization_method: "cmn"

    - name: "init_gaussians"
      inputs:
        features: "work/{sample}.norm"
      outputs:
        gaussians: "models/{sample}.gau"
      parameters:
        n_gaussians: 256

    - name: "train_bw"
      inputs:
        features: "work/{sample}.norm"
        gaussians: "models/{sample}.gau"
      outputs:
        model: "models/{sample}.mdef"
      parameters:
        iterations: 5
        argfile: "config/bw.params"

data:
  samples: ["sample1", "sample2"]  # or use wildcards
  workdir: "work"
  output_dir: "models"

parallelism:
  max_workers: 4
  per_file: true
```

### Chunk Registry Structure

```
pstrain/
├── chunks/
│   ├── __init__.py
│   ├── simple/
│   │   ├── extract_features.yaml
│   │   ├── normalize.yaml
│   │   ├── init_gaussians.yaml
│   │   └── train_bw.yaml
│   └── composed/
│       ├── feature_pipeline.yaml
│       ├── training_pipeline.yaml
│       └── full_training.yaml
└── lib/
    └── chunks.py  # Chunk registry and composition logic
```

## Configuration Schema (Draft)

```yaml
# config.yaml
workflow:
  name: "basic_training"
  version: "1.0"

data:
  audio_dir: "data/audio"
  output_dir: "output"
  samples: ["sample1", "sample2"]  # or use wildcards

features:
  sphinx_fe:
    argfile: "config/feat.params"
    output_format: "mfc"

training:
  iterations: 5
  bw:
    argfile: "config/bw.params"
  init_gau:
    argfile: "config/init_gau.params"

parallelism:
  max_workers: 4
  per_file: true
  per_iteration: false  # Set to true for parallel iterations
```

## Success Criteria

1. ✅ Can run basic training pipeline end-to-end
2. ✅ Processes multiple files in parallel
3. ✅ Handles dependencies correctly (only rebuilds what changed)
4. ✅ Clear error messages when steps fail
5. ✅ Can resume from checkpoints
6. ✅ JSON outputs for all pipeline results
7. ✅ Well-documented and easy to extend
8. ✅ Complete build history tracked and queryable
9. ✅ Comprehensive logging at all levels
10. ✅ Failure analysis helps identify and fix issues
11. ✅ Can query "what builds failed?" and "why did this fail?"
12. ✅ Can run individual steps from CLI (`pstrain step <name>`)
13. ✅ Standalone steps tracked same as pipeline steps
14. ✅ Pipeline can detect and reuse standalone-executed steps
15. ✅ Unified tracking system handles both execution types
16. ✅ Rich trace inspection with Honeycomb-style attributes
17. ✅ Queryable call chains and spans
18. ✅ High-cardinality data for filtering and analysis
19. ✅ Trace visualization and comparison tools
20. ✅ Reusable chunk system (simple and composed)
21. ✅ Easy configuration for defining and composing chunks
22. ✅ Chunk registry and discovery
23. ✅ Parameter passing and composition between chunks

## Build Tracking Design

### Database Schema (SQLite)

```sql
-- Builds table
CREATE TABLE builds (
    build_id TEXT PRIMARY KEY,
    workflow_name TEXT,  -- NULL for ad-hoc step executions
    config_hash TEXT NOT NULL,  -- Hash of config file for change detection
    status TEXT NOT NULL,  -- 'running', 'success', 'failed', 'cancelled', 'interrupted'
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    duration_seconds REAL,
    error_message TEXT,
    error_type TEXT,
    metadata JSON,
    last_checkpoint TEXT,  -- Last successfully completed step
    execution_type TEXT NOT NULL DEFAULT 'pipeline'  -- 'pipeline' or 'step'
);

-- Steps table
CREATE TABLE steps (
    step_id TEXT PRIMARY KEY,
    build_id TEXT,  -- NULL for standalone steps (tracked separately)
    step_name TEXT NOT NULL,
    rule_name TEXT,
    phase_name TEXT,  -- Optional: group steps by phase (e.g., 'feature_extraction', 'training')
    status TEXT NOT NULL,  -- 'pending', 'running', 'success', 'failed', 'skipped'
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    duration_seconds REAL,
    input_files TEXT,  -- JSON array with checksums
    output_files TEXT,  -- JSON array with checksums
    input_checksums TEXT,  -- JSON object: {file: checksum}
    output_checksums TEXT,  -- JSON object: {file: checksum}
    error_message TEXT,
    error_type TEXT,
    retry_count INTEGER DEFAULT 0,
    resource_usage JSON,  -- CPU, memory, disk I/O, etc.
    validated BOOLEAN DEFAULT 0,  -- Whether outputs were validated
    trace_id TEXT,  -- OpenTelemetry trace ID
    span_id TEXT,  -- OpenTelemetry span ID
    execution_type TEXT NOT NULL DEFAULT 'pipeline',  -- 'pipeline' or 'standalone'
    standalone_build_id TEXT,  -- For standalone steps, link to their build record
    FOREIGN KEY (build_id) REFERENCES builds(build_id),
    FOREIGN KEY (standalone_build_id) REFERENCES builds(build_id)
);

-- Products table (tracks all files created by steps)
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL,
    build_id TEXT NOT NULL,
    phase_name TEXT,  -- Optional: group products by phase
    file_path TEXT NOT NULL,
    file_type TEXT,  -- 'output', 'intermediate', 'log', 'checkpoint'
    size_bytes INTEGER,
    checksum TEXT,
    created_at TIMESTAMP NOT NULL,
    deleted_at TIMESTAMP,  -- NULL if not deleted, timestamp if cleaned
    metadata JSON,  -- Additional file metadata
    FOREIGN KEY (step_id) REFERENCES steps(step_id),
    FOREIGN KEY (build_id) REFERENCES builds(build_id)
);
CREATE INDEX idx_products_build_id ON products(build_id);
CREATE INDEX idx_products_step_id ON products(step_id);
CREATE INDEX idx_products_phase_name ON products(phase_name);
CREATE INDEX idx_products_file_path ON products(file_path);

-- File state tracking (for incremental execution)
CREATE TABLE file_states (
    file_path TEXT PRIMARY KEY,
    last_build_id TEXT,
    last_step_id TEXT,
    checksum TEXT NOT NULL,
    size INTEGER,
    mtime REAL,  -- Modification time
    validated BOOLEAN DEFAULT 0,
    FOREIGN KEY (last_build_id) REFERENCES builds(build_id),
    FOREIGN KEY (last_step_id) REFERENCES steps(step_id)
);

-- Config tracking (detect config changes)
CREATE TABLE config_states (
    config_path TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    last_build_id TEXT,
    last_used_at TIMESTAMP,
    FOREIGN KEY (last_build_id) REFERENCES builds(build_id)
);

-- Performance metrics (for analysis and comparison)
CREATE TABLE performance_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,  -- 'duration', 'cpu_time', 'memory_peak', etc.
    metric_value REAL NOT NULL,
    unit TEXT,  -- 'seconds', 'bytes', 'percent', etc.
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (step_id) REFERENCES steps(step_id)
);

-- Traces (link to OpenTelemetry traces)
CREATE TABLE traces (
    trace_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    duration_seconds REAL,
    span_count INTEGER,
    exported_to_honeycomb BOOLEAN DEFAULT 0,
    honeycomb_dataset TEXT,
    FOREIGN KEY (build_id) REFERENCES builds(build_id)
);

-- Span attributes (for querying without accessing full trace)
CREATE TABLE span_attributes (
    span_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    attribute_key TEXT NOT NULL,
    attribute_value TEXT,  -- JSON-encoded for complex values
    attribute_type TEXT,  -- 'string', 'int', 'float', 'bool', 'json'
    PRIMARY KEY (span_id, attribute_key),
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);

-- Create indexes for common queries
CREATE INDEX idx_span_attributes_key_value ON span_attributes(attribute_key, attribute_value);
CREATE INDEX idx_span_attributes_trace ON span_attributes(trace_id);

-- Create indexes
CREATE INDEX idx_builds_status ON builds(status);
CREATE INDEX idx_builds_started ON builds(started_at);
CREATE INDEX idx_builds_config ON builds(config_hash);
CREATE INDEX idx_builds_type ON builds(execution_type);
CREATE INDEX idx_steps_build ON steps(build_id);
CREATE INDEX idx_steps_status ON steps(status);
CREATE INDEX idx_steps_validated ON steps(validated);
CREATE INDEX idx_steps_name ON steps(step_name);
CREATE INDEX idx_steps_type ON steps(execution_type);
CREATE INDEX idx_steps_standalone ON steps(standalone_build_id);
CREATE INDEX idx_file_states_build ON file_states(last_build_id);
CREATE INDEX idx_config_states_hash ON config_states(config_hash);
CREATE INDEX idx_performance_metrics_step ON performance_metrics(step_id);
CREATE INDEX idx_performance_metrics_name ON performance_metrics(metric_name);
CREATE INDEX idx_traces_build ON traces(build_id);
CREATE INDEX idx_span_attributes_key_value ON span_attributes(attribute_key, attribute_value);
CREATE INDEX idx_span_attributes_trace ON span_attributes(trace_id);
```

### Logging Architecture

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (non-fatal issues)
- `ERROR`: Error messages (step failures)
- `CRITICAL`: Critical errors (build failures)

**Log Outputs:**
1. **Console**: Human-readable, colored output
2. **Build log**: Single file per build with all steps
3. **Step logs**: Individual log files per step
4. **Structured logs**: JSON format for parsing/analysis

**Log Format:**
```
[2025-01-12 15:30:45.123] [INFO] [build:abc123] [step:extract_features] Starting feature extraction
[2025-01-12 15:30:45.456] [DEBUG] [build:abc123] [step:extract_features] Command: sphinx_fe -i input.wav -o output.mfc
[2025-01-12 15:30:47.789] [INFO] [build:abc123] [step:extract_features] Completed successfully (2.3s)
```

**Structured JSON Format:**
```json
{
  "timestamp": "2025-01-12T15:30:45.123Z",
  "level": "INFO",
  "build_id": "abc123",
  "step_name": "extract_features",
  "message": "Starting feature extraction",
  "metadata": {
    "input_file": "input.wav",
    "output_file": "output.mfc"
  }
}
```

### Failure Analysis

**Failure Categories:**
1. **Timeout**: Step exceeded time limit
2. **Exit code**: C program returned non-zero
3. **Exception**: Python exception raised
4. **Missing dependency**: Required input file missing
5. **Resource error**: Out of memory, disk full, etc.
6. **Configuration error**: Invalid config parameters

**Failure Reporting:**
- Error message with context
- Stack trace (for Python errors)
- Command that failed (for C programs)
- Input/output files involved
- Suggested fixes based on error type

## Incremental Execution Strategy

### Principles

1. **Never redo completed work**: If a step completed successfully and inputs haven't changed, skip it
2. **Validate before skipping**: Always verify outputs are valid before assuming work is done
3. **Resume from last checkpoint**: On restart, continue from last successful step, not from beginning
4. **Detect changes robustly**: Use checksums, not just timestamps, to detect real changes
5. **Handle interruptions gracefully**: Partial builds should be resumable, not restarted

### Implementation Approach

**1. File State Tracking**
- Store checksums (SHA256) of all input and output files
- Compare current file state vs last successful build
- Only rebuild if:
  - Input file checksum changed
  - Output file missing or invalid
  - Config file changed
  - Step code changed (for Python functions)

**2. Output Validation**
- Check file exists and is non-empty
- Verify file format (magic numbers, headers)
- Store checksum of output for future comparison
- Mark step as "validated" only after all checks pass

**3. Snakemake Integration**
- Use Snakemake's built-in file-based dependency tracking
- Enhance with checksum-based change detection
- Use Snakemake's `checkpoint` feature for resume points
- Integrate with build tracker for state persistence

**4. Resume Logic**
```python
def should_skip_step(step_name, inputs, outputs, config_hash):
    """Determine if step can be skipped.

    Checks both pipeline-executed steps and standalone CLI-executed steps.
    """
    # Check if step completed successfully before (pipeline or standalone)
    last_step = get_last_step(step_name, config_hash, execution_type=None)
    if not last_step:
        return False  # Never run before

    # Check if inputs changed
    for input_file in inputs:
        current_checksum = compute_checksum(input_file)
        if current_checksum != last_step.input_checksums.get(input_file):
            return False  # Input changed

    # Check if outputs are valid
    for output_file in outputs:
        if not file_exists(output_file):
            return False  # Output missing
        if not validate_output(output_file):
            return False  # Output invalid
        current_checksum = compute_checksum(output_file)
        if current_checksum != last_step.output_checksums.get(output_file):
            return False  # Output changed (corrupted?)

    return True  # All checks passed, can skip
```

**5. Unified Step Execution**
- Steps can be executed:
  - As part of a pipeline (via Snakemake)
  - Standalone from CLI (`pstrain step <step-name>`)
- Both execution types tracked in same database
- Pipeline can detect and reuse standalone-executed steps
- Standalone steps create their own "build" record for tracking

**6. Checkpoint System**
- After each step completes, save checkpoint
- On restart, load last checkpoint
- Resume from checkpoint, not from beginning
- Handle partial outputs (clean up incomplete files)
- Checkpoints work for both pipeline and standalone steps

## Implementation Decisions

### Resolved Decisions

1. **C program integration**: Start with subprocess, optimize to C bindings if needed
   - **Decision:** Use `subprocess.run()` initially, can add CFFI later
   - **Status:** ✅ Decided

2. **Config format**: YAML for config files, JSON for API responses
   - **Decision:** Use `pyyaml` for parsing
   - **Status:** ✅ Decided

3. **Workflow versioning**: Semantic versioning in config file, tracked in build tracker
   - **Decision:** `version: "1.0.0"` in workflow config
   - **Status:** ✅ Decided

4. **State management**: Snakemake handles file dependencies, we add checksum validation
   - **Decision:** Use Snakemake's file tracking + our checksum validation
   - **Status:** ✅ Decided

5. **Build storage**: SQLite for now, can migrate to PostgreSQL later
   - **Decision:** SQLite database in `pstrain/builds/builds.db`
   - **Status:** ✅ Decided

6. **Log retention**: Configurable retention policy
   - **Decision:** Keep last N builds, archive older ones
   - **Status:** ✅ Decided

7. **Checksum algorithm**: SHA256 for correctness, cache checksums
   - **Decision:** Default SHA256, allow MD5 for large files
   - **Status:** ✅ Decided

8. **Validation strategy**: Quick checks by default, full validation optional
   - **Decision:** `validate_outputs: true` config option
   - **Status:** ✅ Decided

9. **OpenTelemetry setup**: Local file export initially, OTLP for production
   - **Decision:** File-based traces for local analysis, optional OTLP for distributed
   - **Status:** ✅ Decided

10. **Profiling overhead**: Full profiling by default (can disable for production)
    - **Decision:** Config option `profiling_enabled: true`
    - **Status:** ✅ Decided

11. **Metrics storage**: Store in SQLite, export capability for external systems
    - **Decision:** SQLite for now, export functions for Prometheus/InfluxDB
    - **Status:** ✅ Decided

12. **Step execution model**: Unified tracking with execution_type field
    - **Decision:** Standalone steps create their own build record, pipeline can reference them
    - **Status:** ✅ Decided

13. **Trace inspection tool**: Support both - export to Honeycomb, local query interface
    - **Decision:** OTLP export to Honeycomb, local SQLite for querying
    - **Status:** ✅ Decided

14. **Chunk system design**: YAML for configuration, Python for complex logic
    - **Decision:** YAML definitions with Python validation and execution
    - **Status:** ✅ Decided

15. **Chunk composition**: Template-based with variable substitution
    - **Decision:** Jinja2-style templates in YAML configs
    - **Status:** ✅ Decided

## Next Steps

1. ✅ **Framework Decision: Snakemake**
   - Decision made: Use Snakemake for orchestration
   - Will build observability, chunks, and profiling as needed
   - Can migrate to Dagster later if service mode becomes critical
   - See `docs/design/snakemake-vs-dagster.md` for decision rationale

2. **Start with Phase 1 (Foundation)**
   - Implement TODO 1.1 (Snakemake infrastructure)
   - Set up basic workflow structure
   - Create initial task wrappers

3. **Iterate based on feedback**

## Observability Architecture

### OpenTelemetry Integration (Honeycomb-style)

**Components:**
1. **Tracing**: Track execution flow with rich attributes for inspection
2. **Metrics**: Collect performance and resource metrics
3. **Logging**: Structured logs with trace correlation

**Implementation:**

```python
# pstrain/lib/observability.py
from opentelemetry import trace
from opentelemetry import metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# Setup with rich resource attributes
resource = Resource.create({
    "service.name": "pstrain",
    "service.version": __version__,
    "deployment.environment": os.getenv("ENV", "development"),
})

trace.set_tracer_provider(TracerProvider(resource=resource))
metrics.set_meter_provider(MeterProvider(resource=resource))

# Exporters (Honeycomb via OTLP, or local file)
honeycomb_exporter = OTLPSpanExporter(
    endpoint="https://api.honeycomb.io",
    headers={"x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY")}
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(honeycomb_exporter)
)

# Usage in tasks with rich attributes
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

def extract_features(input_file, output_file, config):
    with tracer.start_as_current_span(
        "extract_features",
        attributes={
            # High-cardinality attributes for filtering
            "step.name": "extract_features",
            "step.type": "feature_extraction",
            "input.file": str(input_file),
            "output.file": str(output_file),
            "input.size_bytes": input_file.stat().st_size,
            "config.sample_rate": config.get("sample_rate"),
            "config.window_size": config.get("window_size"),
        }
    ) as span:
        # Track sub-operations
        with tracer.start_as_current_span(
            "sphinx_fe_execution",
            attributes={
                "program": "sphinx_fe",
                "command": f"sphinx_fe -i {input_file} -o {output_file}",
            }
        ):
            result = run_sphinx_fe(input_file, output_file)

        # Add result attributes
        span.set_attribute("output.size_bytes", output_file.stat().st_size)
        span.set_attribute("duration_seconds", result.duration)
        span.set_attribute("success", result.success)

        # Add event for milestone
        span.add_event("feature_extraction_complete", {
            "features_extracted": result.feature_count,
        })

        return result
```

**Trace Structure with Rich Attributes:**
```
build:abc123 (span)
  attributes: {
    build_id: "abc123",
    workflow_name: "basic_training",
    config_hash: "xyz789",
    execution_type: "pipeline",
    total_steps: 4,
    parallelism: 4
  }
├── step:extract_features (span)
│   attributes: {
│     step.name: "extract_features",
│     input.file: "audio.wav",
│     input.size_bytes: 656672,
│     output.file: "features.mfc",
│     output.size_bytes: 123456,
│     duration_seconds: 2.3,
│     cpu_time: 1.8,
│     memory_peak_mb: 128
│   }
│   ├── sphinx_fe_execution (span)
│   │   attributes: {
│   │     program: "sphinx_fe",
│   │     exit_code: 0,
│   │     stdout_lines: 42
│   │   }
│   └── file_validation (span)
│       attributes: {
│         validation.passed: true,
│         checksum: "sha256:abc..."
│       }
├── step:normalize (span)
│   attributes: { ... }
└── step:train (span)
    attributes: { ... }
```

**Query Examples (Honeycomb-style):**
```python
# Query all failed steps
WHERE step.status = "failed"

# Find slow steps
WHERE step.duration_seconds > 10
GROUP BY step.name
ORDER BY avg(duration_seconds) DESC

# Compare builds
WHERE build_id IN ("abc123", "def456")
GROUP BY build_id, step.name
SELECT avg(duration_seconds), p95(duration_seconds)

# Find memory issues
WHERE step.memory_peak_mb > 1000
AND step.status = "failed"

# Trace call chain for a specific build
WHERE build_id = "abc123"
ORDER BY span.start_time
```

**Metrics Collected:**
- `build.duration` - Total build time
- `step.duration` - Per-step duration
- `step.cpu_time` - CPU time per step
- `step.memory_peak` - Peak memory usage
- `step.disk_io` - Disk I/O bytes
- `build.success_rate` - Success/failure rate
- `step.retry_count` - Number of retries
- All metrics include high-cardinality attributes for filtering

**Profiling Integration:**
- cProfile for CPU profiling
- tracemalloc for memory profiling
- Custom I/O profiler for file operations
- Per-step profiling data stored in build tracker

## References

- [Snakemake Documentation](https://snakemake.readthedocs.io/)
- [Snakemake Best Practices](https://snakemake.readthedocs.io/en/stable/snakefiles/best_practices.html)
- [Scientific Workflow Management](https://snakemake.readthedocs.io/en/stable/tutorial/basics.html)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenTelemetry Best Practices](https://opentelemetry.io/docs/specs/otel/)
