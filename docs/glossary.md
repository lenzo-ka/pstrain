# Glossary

Technical terms used in pstrain and acoustic model training.

## Model Types

### CI (Context-Independent)
Models where each phone is modeled independently of its neighboring phones. Also called **monophone** models. Simpler and faster to train, but less accurate than CD models. CI models are typically trained first and used to bootstrap CD training.

### CD (Context-Dependent)
Models where each phone is modeled based on its surrounding context (typically the preceding and following phones). Also called **triphone** models. More accurate but require more training data. Example: the "AE" in "cat" is modeled differently than the "AE" in "bat".

### Tied vs Untied
- **Untied**: Each triphone state has its own parameters. Leads to data sparsity problems since many triphones are rare.
- **Tied**: Similar triphone states share parameters via decision tree clustering. Reduces the number of parameters while maintaining context sensitivity.

## Core Concepts

### HMM (Hidden Markov Model)
A statistical model where the system being modeled is assumed to be a Markov process with hidden states. In speech recognition, HMMs model the temporal evolution of speech sounds. Each phone is typically modeled as a 3-5 state HMM.

### GMM (Gaussian Mixture Model)
A probability distribution modeled as a weighted sum of Gaussian distributions. Used to model the emission probabilities of HMM states. Each state can have multiple Gaussian components (mixture components).

### State
A single position in an HMM. Phones are typically modeled with 3 emitting states (beginning, middle, end) plus entry and exit states.

### Senone (Tied State)
A unique HMM state after state tying. Multiple triphone states that behave similarly are clustered together and share the same senone. The number of senones determines model size.

## Features

### MFCC (Mel-Frequency Cepstral Coefficients)
The standard acoustic features for speech recognition. Derived from the spectrum of audio, compressed using the mel scale (which approximates human hearing) and decorrelated using the discrete cosine transform.

### Cepstrum
The inverse Fourier transform of the log spectrum. Separates the vocal tract response from the excitation signal.

### Delta / Delta-Delta
First and second derivatives of features over time. Capture dynamic information about how features change. A 13-dimensional MFCC with deltas becomes 39-dimensional (13 + 13 + 13).

### CMN (Cepstral Mean Normalization)
Subtracting the mean of cepstral features to reduce channel effects (microphone, room acoustics). Can be done per-utterance (batch) or with a running average (live).

### AGC (Automatic Gain Control)
Normalizing audio amplitude to reduce volume variation effects.

## Training Algorithms

### Baum-Welch
An Expectation-Maximization (EM) algorithm for training HMM parameters. Iteratively re-estimates model parameters to maximize the likelihood of the training data. Also called forward-backward algorithm.

### Viterbi
Algorithm for finding the most likely state sequence through an HMM given observations. Used during alignment and decoding.

### Forced Alignment
Using a known transcription to determine the exact timing of phones in an audio file. The Viterbi algorithm finds the best alignment of the transcription to the audio.

## Model Parameters

### Means / Variances
The parameters of Gaussian distributions. Each Gaussian component has a mean vector and variance (or covariance) describing its center and spread in feature space.

### Mixture Weights
The relative weights of Gaussian components in a GMM. Must sum to 1.0 for each state.

### Transition Matrices (tmat)
Probabilities of transitioning between HMM states. Typically include self-loops (staying in the same state) and forward transitions.

### Density
A single Gaussian component. "4 densities per state" means each state's GMM has 4 Gaussian components.

## Files and Formats

### mdef (Model Definition)
Defines the structure of the acoustic model: which phones exist, how many states each has, and the mapping from triphones to tied states.

### ctl (Control File)
A list of utterance IDs, one per line. Used to specify which files to process.

### fileids
Same as ctl file - a list of utterance identifiers.

### transcription
Text file mapping utterance IDs to their word transcripts. Format: `<s> word1 word2 ... </s> (utterance_id)`

### feat.params
The complete training-time front-end record carried by a trained model. Pstrain's
`Decoder`, `Aligner`, and `package_model` entry points require it, but init and
training do not while they are still constructing a model.

At decode time, the trained model's `feat.params` is authoritative over the
active schema profile because it records how that model's features were actually
made. PocketSphinx reads it after pstrain's pre-initialization assignments, so a
value in the file wins. A complete file defines all 19 schema fields plus the
`unit_area` and `round_filters` training-engine invariants; none is optional.
Pstrain validates that inventory before its supported decode, align, and package
entry points continue. A source gate rejects direct use of the pip PocketSphinx
`Decoder` in the `pstrain` package, but code outside the package can bypass this
contract and is not covered by the guarantee.

### sendump
Precomputed senone dump file for faster model loading during decoding.

## Training Stages

### Flat Initialization
Creating initial model parameters before training. "Flat" means all parameters start with the same values (uniform mixture weights, global mean/variance).

### Convergence
When training iterations stop improving significantly. Measured by change in log-likelihood between iterations.

### Iteration
One complete pass through the training data with the Baum-Welch algorithm. Training typically runs 8-20 iterations until convergence.

### Gaussian Splitting
Increasing model capacity by splitting each Gaussian into two. Start with 1 density, train to convergence, split to 2, train again, split to 4, etc.

## Decision Trees

### Question
A binary test about phonetic context. Examples: "Is the left phone a vowel?", "Is the right phone a nasal?"

### Quest File
File containing all the phonetic questions used for decision tree building.

### Pruning
Removing branches from a decision tree to prevent overfitting. Controlled by a threshold on the minimum improvement required to keep a split.

## Abbreviations Reference

| Abbrev | Meaning |
|--------|---------|
| CI | Context-Independent |
| CD | Context-Dependent |
| HMM | Hidden Markov Model |
| GMM | Gaussian Mixture Model |
| MFCC | Mel-Frequency Cepstral Coefficients |
| CMN | Cepstral Mean Normalization |
| AGC | Automatic Gain Control |
| EM | Expectation-Maximization |
| LDA | Linear Discriminant Analysis |
| MLLT | Maximum Likelihood Linear Transform |
| MLLR | Maximum Likelihood Linear Regression |
| MAP | Maximum A Posteriori (adaptation) |
| BW | Baum-Welch (algorithm) |
| tmat | Transition Matrix |
| mdef | Model Definition |
| ctl | Control (file) |
