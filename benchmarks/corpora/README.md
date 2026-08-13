# Additional corpus archives

This parallel manifest covers corpora whose layout is not CMU Arctic's
`wav/` plus `etc/txt.done.data` shape. Archive content checks are data, so a
new corpus or multi-archive corpus can be added without a corpus-name branch
in the fetcher. Arctic retains its established manifest because its benchmark
also consumes voice-specific metadata directly.

Fetch and verify a corpus with:

```console
python scripts/fetch_corpus.py cmu_us_kal_diphone /path/to/cache
```

An existing archive is always hashed and checked against the pin. Downloads
use a temporary `.part` file and an atomic rename, so a failed transfer cannot
replace a good cached archive. Corpus data and extraction products are not
committed.

## CMU US KAL Diphone

KAL contains 1,349 US English nonsense-word recordings designed for uniform
phone-phone transition coverage. It is therefore a phone-sequence alignment
corpus rather than a natural-word/ordinary-lexicon corpus; its `dic/` support
file is not a natural-word pronunciation lexicon. The full archive contains
1,349 hand-corrected `lab/` labels and 1,349 bootstrap/automatic
`prompt-lab/` labels. The latter have regular synthetic 100 ms boundaries;
the former carry the corrected boundaries. The separate Festvox example
walkthrough uses automatic labelling and is not the hand-corrected database.

The full archive contains 1,349 `pm/` pitchmark files, one `pm_lab/` label,
one `dic/` file, and three `etc/` files. Although the reference page says the
database includes laryngograph (EGG) signals and word labels, both current
packed archives contain empty `lar/` and `wrd/` directories. The page's older
`cmu_us_kal_lar.tar.bz2` link returns 404. These empty directories are checked
explicitly so a future upstream replacement cannot silently change the pinned
shape.

Festvox also publishes `cmu_us_kal_diphone_base.tar.bz2` (36M). It is not pinned because all 4,073 of
its regular files are byte-for-byte present in the pinned full archive; the
full archive additionally supplies 8,094 derived and bootstrap files,
including `prompt-lab/`. Pinning both would duplicate data without adding
content.

The corpus license is the permissive Alan W Black and Kevin Lenzo license in
the root `COPYING` file, also available from the
[Festvox database](http://festvox.org/databases/cmu_us_kal_diphone/COPYING).
