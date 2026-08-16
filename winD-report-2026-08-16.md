# Wave D report — August 16, 2026

## Linux libc cross-check and kmeans_init sanitizer cleanup

- Made the POSIX `drand48()` cross-check deterministic by explicitly installing the in-repo generator's `0x1234abcd330e` Xi state with `seed48()` before comparing streams. The golden vector remains seeded with zero and still anchors the exact 48-bit states. Seeded `srand48()`, `seed48()`, `drand48()`, and `lrand48()` comparisons remain exhaustive; no assertion depends on a libc's unspecified initial process-global state.
- Released the feature configuration, output and dump model definitions, dictionaries, global command-line configuration, optional tied-state range file, trained Gaussian parameter arrays, per-state labels, and optional mixture weights in `kmeans_init`. Recoverable initialization failures now release partially constructed objects as well. Corrected the triangular k-means neighbor-map destructor to free its two-dimensional allocation, including its error path.
- Local CTest passed all 10 tests, including RNG test 6 and continuation tests 7–9. A temporary per-state RNG reset made test 9 fail with the expected reset signature (`2243, 2743`), confirming the continuation regression remains discriminating.
- AppleClang built the ASan/UBSan `kmeans_init`, `test_rng`, and continuation-test targets. The macOS runtime reports that leak detection is unsupported, consistent with the repository's Linux-only sanitizer policy; Linux CI remains the leak-detection oracle.
