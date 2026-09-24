# Alignment mass and coverage report

A corpus alignment accepts or rejects what the wider-beam retry recovers, and
the overall success rate says how many utterances survived. It does not say
whether what survived still covers the corpus: which speakers kept their
audio, which phones and triphones rely on retries, and which ones the output
lost. The mass-and-coverage report answers those questions.

**The report never changes acceptance.** It is built after the corpus pass,
from outcomes the pass already decided. It does not change which retry-recovered
alignments are accepted or rejected, the acceptance threshold, its calibration,
or any default.

## Where it appears

- `pstrain align` prints it after the per-rung retry line, under
  `Mass and coverage by outcome (reporting only; acceptance is unchanged)`.
- `align_corpus` attaches it to the returned job as `AlignmentJob.coverage`, an
  `AlignmentCoverage`. `coverage.format()` returns the text report and
  `coverage.to_dict()` a JSON-serializable form. Pass `coverage_report=False`
  to skip it. If the report cannot be built, for example because the
  dictionary cannot be read, the run logs a warning and `coverage` is `None`.
  The alignment itself is unaffected.
- `alignment_coverage(job, transcripts, dict_path, ...)` builds the report for
  a finished job. Use it to pass your own speaker mapping or flag parameters.

Training does not produce this report. Baum-Welch training retries failed
utterances but has no acceptance check, and its failed-alignment policy is
described in [Failed-alignment policy](design/failed-alignment-policy.md).

## Outcomes

Every utterance ends in exactly one outcome:

| Outcome | Meaning |
| --- | --- |
| first pass | Aligned at the nominal beam. |
| retry accepted | The retry recovered it and the acceptance check accepted it. With the check off, every recovery counts here. |
| retry rejected | The retry recovered it and the acceptance check rejected it. |
| not recovered | It did not align for any other reason: the retry failed or was not run, the audio was too short for the transcript, the audio was missing, a word had no pronunciation, or the aligner did not start. |

## What is counted

**Mass.** Utterances and audio duration per outcome, overall and per speaker.
Duration comes from the alignment's frames when there is an alignment and from
the WAV header otherwise. An utterance whose audio is missing or unreadable is
counted with no duration, and the report says how many there were.

**Speakers.** pstrain has no speaker field on the alignment path. The report
takes the speaker to be the text before the first `/` in the utterance ID
(`bdl/arctic_a0001` is speaker `bdl`), the same convention the Arctic
benchmark uses. IDs without a `/` share one speaker, `(no speaker prefix)`,
and the text report then says no speakers are named. Pass
`speaker_of=` to `alignment_coverage` to use another mapping. The text report
lists each speaker when there are 20 or fewer, and otherwise gives counts.

**Phones and triphones.** Token counts per outcome.

- For first-pass and retry-accepted utterances, the phones are the alignment's
  own phone segments. The base phone of a context-dependent segment is used.
  When phones were not captured (`--no-phones`), the aligned words are expanded
  through the dictionary, using the pronunciation variant the aligner chose.
- For retry-rejected and not-recovered utterances there is no alignment to
  read. Their transcripts are expanded through the dictionary instead, using
  each word's first pronunciation. Words with no pronunciation are listed.
- A triphone is written `L-P+R`: the phone with its neighbors in that sequence,
  across word boundaries. The utterance edges count as `SIL`, and so does any
  filler neighbor. Filler phones (`SIL` and the filler dictionary's phones)
  are never counted as units. Aligned sequences include the pauses the aligner
  inserted between words. Transcript expansions have none, so a cross-word
  triphone from a failed utterance may be one that aligned utterances realized
  with a pause.

## Flags

| Flag | Rule |
| --- | --- |
| Speakers with no aligned audio | Every utterance of the speaker was rejected or not recovered. |
| Retry-only phones and triphones | The unit has aligned tokens only from accepted retries, and none from the first pass. |
| Phones and triphones seen only in failed utterances | The unit has no aligned tokens and appears only in rejected or unrecovered utterances. |
| Thin phones | A possible lexicon or model problem. See the rule below. |

**Thin-phone rule.** A phone is flagged when all of these hold:

- it has fewer than `thin_tokens` first-pass tokens (default 50);
- it appears in at least `thin_min_carriers` utterances, in any outcome
  (default 3); and
- the share of those utterances that did not align (rejected or not recovered)
  is at least `thin_rate_ratio` (default 2.0) times the run's share of
  utterances that did not align.

A run with no failed utterances flags no thin phones. Accepted retries do not
count as failures here. A phone that aligns only through retries is flagged as
retry-only instead. Flagged thin phones are listed worst first, by the share of
their utterances that failed. The parameters are arguments of
`alignment_coverage`. They are not config fields, and `pstrain align` uses the
defaults.

## Size

The text report stays bounded. Each flag list names up to 60 units and counts
the rest. `to_dict()` names every flagged unit and has the full per-phone,
per-triphone and per-speaker tables. Per-utterance outcomes are in
`AlignmentCoverage.outcomes`.
