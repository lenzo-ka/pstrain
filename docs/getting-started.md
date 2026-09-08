# Getting Started

## Installation

Install from source (recommended):

```bash
git clone https://github.com/lenzo-ka/pstrain
cd pstrain
pip install -e .
```

## Quick Start

For the turnkey path, provide the corpus inputs in one command. It builds the
default `cd-8g` context-dependent model:

```bash
pstrain train my_project \
    --audio audio/ \
    --prompts prompts.txt \
    --dictionary dictionary.dict
```

Pass `--target ci-1g` to stop at the context-independent bootstrap stage.

See [Input formats](input-formats.md) for the accepted dictionary, transcript,
phoneset, and audio formats.

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

# Package a trained model for distribution
pstrain package cd-8g --project-dir my_project
```

Packages created before the `pstrain-package.json` marker was introduced are
recognizable as legacy packages, but replacement now requires explicit consent.
The training pipeline deliberately does not set that consent automatically:
doing so would make a legacy pipeline directory indistinguishable from a
hand-assembled decoder package with the same structure. Before rerunning a
package target against pre-marker output, move
`dist/models/<model>-<profile>` aside so it remains recoverable. Alternatively,
after inspecting the directory, replace it explicitly with the equivalent
package command:

```bash
pstrain package cd-8g --project-dir my_project \
    --out my_project/dist/models \
    --name cd-8g-default \
    --overwrite
```

Subsequent pipeline runs recognize the marker written by that command and do
not require another opt-in.

### Supplying an existing train/test split

To preserve a corpus's canonical partition or your own held-out set, create all
four Sphinx-format files before running `pstrain split` or `pstrain build`:

```text
my_project/experiments/default/etc/train.fileids
my_project/experiments/default/etc/test.fileids
my_project/experiments/default/etc/train.transcription
my_project/experiments/default/etc/test.transcription
```

Each `.fileids` file contains one utterance ID per line. Its matching
`.transcription` uses one of the {ref}`accepted transcript forms <transcripts>`
and contains the same IDs in exactly the same order, with the transcript text
from `etc/all.transcription`. Together, train and test must
partition `all.transcription` exactly, may not overlap, and every ID must have a
matching `audio/<fileid>.wav` (nested file IDs are supported).

When all four files are supplied, they are authoritative: pstrain validates but
does not rewrite or reorder them. Any mismatch, omission, overlap, transcript
change, or missing audio is an error. If the files are absent, the existing
automatic 95/5 seeded split remains the default.

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
report = validate_project(Path("my_project"))
if not report.is_valid:
    print(report.summary())

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
