# Input formats

Pstrain accepts the following corpus inputs. Text files are UTF-8.

## Pronunciation dictionary

Each entry has a word, optional numbered pronunciation sense, whitespace, and
one or more phones:

```text
word[(sense)] phone1 phone2 ...
```

Spaces or tabs may separate fields. Alternative pronunciations use Sphinx
numbering beginning with `(2)`:

```text
record R EH K ER D
record(2) R IH K AO R D
```

`#` introduces a whole-line or inline comment. Words and phones are
case-sensitive, and the parser does not impose a particular phoneset. An expanded
pstrain-native dictionary format is under discussion for the future; the
CMU/Sphinx format above is the format accepted now.

## Transcripts

Two word-level transcript forms are accepted:

```text
[<s>] words [</s>] (fileid)
fileid words
```

In the Sphinx form, `<s>` and `</s>` are optional silence markers, independently
of one another, following the conventions used by the arpabo language-model
tools. For example, all of these are valid:

```text
<s> THE QUICK BROWN FOX </s> (arctic_a0001)
THE QUICK BROWN FOX (arctic_a0001)
<s> THE QUICK BROWN FOX (arctic_a0001)
THE QUICK BROWN FOX </s> (arctic_a0001)
arctic_a0001 THE QUICK BROWN FOX
```

A trailing `(fileid)` token selects the Sphinx form; only that trailing token
is the ID, so parentheses may appear earlier in the transcript text. Otherwise,
the first whitespace-delimited token is the file ID and the remainder is text.

## Phoneset

A phoneset file is optional and contains one phone per line. When supplied,
pstrain copies it to `shared/phoneset.txt`, honors it as the model's phone
inventory, and validates the dictionary against it. Every phone in the
dictionary must appear in the phoneset or training stops.

When no phoneset is supplied, pstrain derives one from the union of the main
dictionary and filler-dictionary pronunciations. The default filler dictionary
supplies `SIL`.

## Audio

Audio must be uncompressed mono PCM WAV. The expected and default format is
16 kHz mono, 16-bit little-endian PCM. Other sample rates are accepted without
resampling, but every WAV in a corpus must use the same sample rate. The reader
also retains its existing support for converting 8-bit PCM samples to 16-bit
samples; 16-bit little-endian PCM remains the default input width.
