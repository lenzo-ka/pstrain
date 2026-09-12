# Getting started

This walkthrough trains a small acoustic model from the audio included in the
repository. You do not need to write Python. Use Terminal on **macOS or Linux**;
native Windows cannot run the training pipeline. Windows users can install a
Linux environment using [Microsoft's WSL instructions](https://learn.microsoft.com/en-us/windows/wsl/install)
and work inside its Linux terminal. WSL is not a separately tested pstrain platform.
See [platform support](support.md) for the precise limits.

## Install the current checkout

You need Git, **Python 3.11–3.13**, and a C/C++ compiler for this walkthrough.
Python 3.12 is a suitable choice; these Python versions are covered by CI. If
Python is missing, install one of these versions from the
[official Python downloads](https://www.python.org/downloads/). If
`python3 --version` reports an older version, use the installed version
explicitly—for example, replace `python3` with `python3.13` in the version check and the
`-m venv` command below. After activation, continue using `python` as shown.

On macOS, install Apple's Command Line Tools with
`xcode-select --install` if needed; see [Apple's tools page](https://developer.apple.com/xcode/resources/).
On Debian/Ubuntu, the relevant system packages are `git`, `build-essential`,
`python3-dev`, and `python3-venv`; check that your distribution's `python3` is
new enough. Install missing system tools before continuing.

Copy these commands into your terminal, one line at a time:

```bash
python3 --version
git clone https://github.com/lenzo-ka/pstrain.git
cd pstrain
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[test]"
pstrain info
```

The `.venv` directory keeps pstrain's Python dependencies separate from your
other software. Activation makes `python` and `pstrain` refer to that environment;
see [Python's virtual-environment guide](https://docs.python.org/3/library/venv.html).
The install command builds the bundled native code and installs the evaluation
metrics extra (`test`). It obtains CMake 3.25 or newer and other Python build
requirements automatically; you do not need a separate `cmake --build` step.
The first installation needs internet access for dependencies and the pinned
PocketSphinx source and can take several minutes.

In `pstrain info`, check that **C library: available** appears. If compilation
fails, check the compiler prerequisite and the first error in pip's output.
If `pstrain` is not found, activate `.venv` again. In a new terminal, return to
this checkout and run `source .venv/bin/activate`. Run `deactivate` when finished.

This installs the checked-out source, not the published PyPI release. After
updating the checkout, repeat `python -m pip install ".[test]"` to rebuild it.
For development and editable installs, see [the development guide](development.md).

## Train a small model

Stay in the checkout directory, with the environment active. The following
uses the bundled mini Arctic corpus, so it needs no corpus download. Choose a
new output directory; `demo` below must not already contain another project.

```bash
pstrain train demo \
  --audio tests/fixtures/mini_arctic/wav \
  --prompts tests/fixtures/mini_arctic/transcription.txt \
  --dictionary tests/fixtures/mini_arctic/dictionary.dict \
  --phoneset tests/fixtures/mini_arctic/phoneset.txt \
  --filler-dict tests/fixtures/mini_arctic/filler.dict \
  --target ci-1g -j 1
```

The backslash continues a command onto the next line; keep it as the last
character on each continued line. `ci-1g` requests a small context-independent
model with one Gaussian per state, and `-j 1` uses one training worker.
Success ends with `Status: trained` and prints the model directory:
`demo/shared/models/ci-1g/default`. A pass summary may say `converged=False`:
that means the configured iteration limit was reached, not that convergence
was demonstrated. This tiny run checks the workflow, not recognition quality.

Decode its held-out utterance and package the model:

```bash
pstrain test ci-1g --project-dir demo
pstrain package ci-1g --project-dir demo
```

The test command should report `1/1 decoded` and prints WER (word error rate);
a high value is expected for this tiny corpus and model, and insertions can
make WER exceed 100%. It builds
a language model from the training transcript. These results are not the
[Arctic benchmark](benchmarks/arctic-pin.md). The package command prints its
output files under `demo/packages/ci-1g/`, including the acoustic model,
dictionaries, and `pstrain-package.json` manifest.

To continue through the larger default model, reuse the prepared project:

```bash
pstrain build cd-8g --project-dir demo -j 1
```

Then use `cd-8g` in the test and package commands above. For your own corpus,
replace the input paths in `pstrain train` and use a new project directory.
Read [input formats](input-formats.md) first: utterance IDs must match the audio
paths, and prompt words must match the pronunciation dictionary. Run
`pstrain train --help` for options, including explicit resume behavior.

## Install a published release instead

If you already have your own corpus and want a release rather than this
checkout, create a separate environment in a new directory:

```bash
mkdir pstrain-work
cd pstrain-work
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "pstrain[test]"
pstrain info
pstrain train --help
```

A compatible wheel includes the native library; when no wheel matches your
Python/platform, pip builds from source and needs the compiler prerequisites
above. The repository's mini corpus is not part of the installed package.
Release commands and fixes can lag this checkout: consult the installed
`--help` and the release's documentation rather than assuming equal behavior
from the version number alone.

For a longer guided notebook, `pstrain tutorial` copies the bundled tutorial
into the current directory. Opening it requires a Jupyter installation; its
setup cells describe additional dependencies and corpus downloads. It is
optional and separate from the terminal walkthrough above.

## Further workflows

The lower-level `setup`, `validate-project`, `split`, `features`, and `flat` commands
are available for decomposed workflows. Each accepts `--help`.

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
