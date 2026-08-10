# Getting Started

## Installation

Install from source (recommended):

```bash
git clone https://github.com/lenzo-ka/pstrain
cd pstrain
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Set up a new project
pstrain setup my_project \
    --transcription transcripts.txt \
    --dictionary dictionary.dict \
    --audio audio/

# Validate the project
pstrain validate-project my_project

# Split data into train/test sets
pstrain split --project-dir my_project

# Extract features
pstrain features --project-dir my_project

# Initialize flat model
pstrain flat --project-dir my_project
```

### Python API

```python
from pstrain.api import setup_project, validate_project, create_model
from pathlib import Path

# Set up a new project
result = setup_project(
    project_dir=Path("my_project"),
    transcription_path=Path("transcripts.txt"),
    dictionary_path=Path("dictionary.dict"),
)

# Validate the project
errors = validate_project(Path("my_project"))
if errors:
    print(f"Validation errors: {errors}")

# Create a model
model = create_model("ci", config="baseline")
print(f"Model: {model.display_name}")
print(f"Default topn: {model.default_topn}")
```

## Building from Source

The C library must be built before using CFFI bindings:

```bash
# Build C library
cmake -S . -B build
cmake --build build

# Install Python package in development mode
pip install -e .
```
