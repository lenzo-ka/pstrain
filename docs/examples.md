# Examples

Basic usage examples.

> **Note**: `pstrain.api` is the recommended public API. `pstrain.lib` provides the same
> functions and can be used interchangeably.

## Project Setup

```python
from pstrain.api import setup_project
from pathlib import Path

# Set up a new project
result = setup_project(
    project_dir=Path("my_project"),
    transcription_path=Path("transcripts.txt"),
    audio_path=Path("audio/"),
    dictionary_path=Path("dictionary.dict"),
    link_audio=True,  # Symlink instead of copy
)

print(f"Project created at: {result['project_dir']}")
```

## Project Validation

```python
from pstrain.api import validate_project
from pathlib import Path

# Validate a project
report = validate_project(Path("my_project"))

if not report.is_valid:
    print(report.summary())
else:
    print("Project is valid!")
```

## Model Creation

```python
from pstrain.api import create_model, CIModel

# Create a CI model instance
model = create_model("ci", config="baseline")

# Access model properties
print(f"Model type: {model.display_name}")
print(f"Default topn: {model.default_topn}")
print(f"Dependencies: {model.get_training_dependencies()}")

# Get directory paths
hmm_dir = model.get_hmm_dir("experiments/baseline")
print(f"HMM directory: {hmm_dir}")

# Get default training parameters
params = model.get_default_training_params()
print(f"Save alignments: {params['save_alignments']}")
```

## Data Structures

```python
from pstrain.api import Dictionary, Phoneset, get_fileids, parse_transcription_file
from pathlib import Path

# Load dictionary
dictionary = Dictionary.from_file(Path("shared/dictionary.dict"))
print(f"Dictionary has {len(dictionary)} words")

# Extract phoneset from dictionary
phoneset = Phoneset.from_dictionary(dictionary)
print(f"Phoneset has {len(phoneset)} phones")

# Parse transcription file
transcripts = parse_transcription_file(Path("etc/all.transcription"))
fileids = get_fileids(Path("etc/all.transcription"))
print(f"Found {len(fileids)} fileids")
```

## Log-domain Math

Use the guarded wrapper for native log-domain arithmetic. Raw library handles
and symbols in the private `_pstrainc` module are implementation details.

```python
from pstrain.lib._pstrainc import LogMath


def main():
    logmath = LogMath(base=1.0001)
    log_probability = logmath.log(0.5)
    print(logmath.exp(log_probability))
    print(logmath.add(log_probability, log_probability))
    print(logmath.get_base())


if __name__ == "__main__":
    main()
```
