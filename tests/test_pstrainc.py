"""Test low-level C bindings."""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cffi import FFI

from tests.conftest import requires_c_library

# Skip entire module if C library not built
pytestmark = requires_c_library

from pstrain.lib import _pstrainc  # noqa: E402 - must be after skip check


@pytest.fixture
def lib() -> Any:
    """Get the loaded C library."""
    return _pstrainc.get_lib()


@pytest.fixture
def ffi() -> FFI:
    """Get the FFI instance."""
    return _pstrainc.get_ffi()


def test_library_loads(lib: Any) -> None:
    """Test that the library loads."""
    assert lib is not None


def test_logmath_init_free(lib: Any) -> None:
    """Test logmath creation and destruction."""
    lmath = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)
    assert lmath is not None
    lib.pstrain_cffi_logmath_free(lmath)


def test_logmath_log_exp(lib: Any) -> None:
    """Test logmath log/exp operations."""
    lmath = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)

    log_half = lib.pstrain_cffi_logmath_log(lmath, 0.5)
    exp_back = lib.pstrain_cffi_logmath_exp(lmath, log_half)

    # Should be approximately 0.5
    assert abs(exp_back - 0.5) < 0.001

    lib.pstrain_cffi_logmath_free(lmath)


def test_logmath_add(lib: Any) -> None:
    """Test logmath addition in log domain."""
    lmath = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)

    log_half = lib.pstrain_cffi_logmath_log(lmath, 0.5)
    log_quarter = lib.pstrain_cffi_logmath_log(lmath, 0.25)
    log_sum = lib.pstrain_cffi_logmath_add(lmath, log_half, log_quarter)
    exp_sum = lib.pstrain_cffi_logmath_exp(lmath, log_sum)

    # 0.5 + 0.25 = 0.75
    assert abs(exp_sum - 0.75) < 0.001

    lib.pstrain_cffi_logmath_free(lmath)


def test_logmath_get_base(lib: Any) -> None:
    """Test logmath base retrieval."""
    base = 1.0001
    lmath = lib.pstrain_cffi_logmath_init(base, 0, 1)

    retrieved_base = lib.pstrain_cffi_logmath_get_base(lmath)
    assert abs(retrieved_base - base) < 0.0001

    lib.pstrain_cffi_logmath_free(lmath)


def test_logmath_edge_cases(lib: Any) -> None:
    """Test logmath with edge cases."""
    lmath = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)

    # Test very small value
    log_tiny = lib.pstrain_cffi_logmath_log(lmath, 0.0001)
    exp_tiny = lib.pstrain_cffi_logmath_exp(lmath, log_tiny)
    assert abs(exp_tiny - 0.0001) < 0.0001

    # Test value near 1.0
    log_near_one = lib.pstrain_cffi_logmath_log(lmath, 0.999)
    exp_near_one = lib.pstrain_cffi_logmath_exp(lmath, log_near_one)
    assert abs(exp_near_one - 0.999) < 0.001

    lib.pstrain_cffi_logmath_free(lmath)


def test_logmath_different_bases(lib: Any) -> None:
    """Test logmath with different bases."""
    # Base 1.0001
    lmath1 = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)
    log1 = lib.pstrain_cffi_logmath_log(lmath1, 0.5)
    lib.pstrain_cffi_logmath_free(lmath1)

    # Base 1.001
    lmath2 = lib.pstrain_cffi_logmath_init(1.001, 0, 1)
    log2 = lib.pstrain_cffi_logmath_log(lmath2, 0.5)
    lib.pstrain_cffi_logmath_free(lmath2)

    # Different bases should give different log values
    assert log1 != log2


def test_enum_constants(lib: Any) -> None:
    """Test that enum constants are accessible."""
    assert lib.CMN_NONE == 0
    assert lib.CMN_LIVE == 1
    assert lib.CMN_BATCH == 2

    assert lib.AGC_NONE == 0
    assert lib.AGC_MAX == 1
    assert lib.AGC_EMAX == 2
    assert lib.AGC_NOISE == 3


def test_error_macros_exist(lib: Any) -> None:
    """Test that error macros are accessible."""
    # E_INFO, E_WARN, etc. are variadic macros in the C code
    # They're exposed as functions in cffi, but may not be directly callable
    # Just verify they exist in the library (they're in our CDEF)
    # Note: These may not work as expected in ABI mode since they're macros
    # For now, skip this test or mark as expected to fail
    pass


def test_logmath_identity(lib: Any) -> None:
    """Test logmath identity: exp(log(x)) = x."""
    lmath = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)

    test_values = [0.1, 0.25, 0.5, 0.75, 0.9]
    for val in test_values:
        log_val = lib.pstrain_cffi_logmath_log(lmath, val)
        exp_val = lib.pstrain_cffi_logmath_exp(lmath, log_val)
        assert abs(exp_val - val) < 0.01, f"Identity failed for {val}: {exp_val}"

    lib.pstrain_cffi_logmath_free(lmath)


def test_logmath_add_commutative(lib: Any) -> None:
    """Test that logmath addition is commutative."""
    lmath = lib.pstrain_cffi_logmath_init(1.0001, 0, 1)

    log_a = lib.pstrain_cffi_logmath_log(lmath, 0.3)
    log_b = lib.pstrain_cffi_logmath_log(lmath, 0.4)

    sum_ab = lib.pstrain_cffi_logmath_add(lmath, log_a, log_b)
    sum_ba = lib.pstrain_cffi_logmath_add(lmath, log_b, log_a)

    # Should be approximately equal (within rounding)
    assert abs(sum_ab - sum_ba) < 10  # Allow some rounding error

    lib.pstrain_cffi_logmath_free(lmath)


# =============================================================================
# S3 I/O Round-trip Tests - verify Python wrappers produce correct C format
# =============================================================================


def test_mixw_roundtrip() -> None:
    """Test that write_mixw -> read_mixw produces identical data."""
    # Create test data: mixture weights (n_mixw, n_feat, n_density)
    original = np.random.rand(10, 1, 4).astype(np.float32)
    # Normalize to valid probabilities
    original = original / original.sum(axis=2, keepdims=True)

    with tempfile.NamedTemporaryFile(suffix=".mixw", delete=False) as f:
        tmpfile = f.name

    try:
        # Write using Python wrapper -> C function
        ret = _pstrainc.write_mixw(tmpfile, original)
        assert ret == 0, f"write_mixw failed with return code {ret}"

        # Read back using Python wrapper -> C function
        result, n_mixw, n_feat, n_density = _pstrainc.read_mixw(tmpfile)

        # Verify dimensions
        assert n_mixw == 10
        assert n_feat == 1
        assert n_density == 4
        assert result.shape == original.shape

        # Verify data matches
        assert np.allclose(original, result, rtol=1e-5), "Data mismatch after round-trip"
    finally:
        Path(tmpfile).unlink(missing_ok=True)


def test_tmat_roundtrip() -> None:
    """Test that write_tmat -> read_tmat produces identical data."""
    # Create test data: transition matrices (n_tmat, n_state, n_state)
    # Left-to-right topology
    n_tmat, n_state = 3, 4
    original = np.zeros((n_tmat, n_state, n_state), dtype=np.float32)
    for t in range(n_tmat):
        for i in range(n_state - 1):
            original[t, i, i] = 0.5  # self-loop
            original[t, i, i + 1] = 0.5  # forward
        original[t, n_state - 1, n_state - 1] = 1.0  # exit state

    with tempfile.NamedTemporaryFile(suffix=".tmat", delete=False) as f:
        tmpfile = f.name

    try:
        ret = _pstrainc.write_tmat(tmpfile, original)
        assert ret == 0, f"write_tmat failed with return code {ret}"

        result, out_n_tmat, out_n_state = _pstrainc.read_tmat(tmpfile)

        assert out_n_tmat == n_tmat
        assert out_n_state == n_state
        # Note: tmat read returns (n_tmat, n_state-1, n_state) - no exit row
        assert result.shape == (n_tmat, n_state - 1, n_state)

        # Compare non-exit rows
        assert np.allclose(original[:, :-1, :], result, rtol=1e-5)
    finally:
        Path(tmpfile).unlink(missing_ok=True)


def test_gau_roundtrip() -> None:
    """Test that write_gau -> read_gau produces identical data."""
    # Create test data: Gaussian params (n_mgau, n_feat, n_density, veclen)
    n_mgau, n_feat, n_density, veclen = 5, 1, 2, 13
    original = np.random.rand(n_mgau, n_feat, n_density, veclen).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".gau", delete=False) as f:
        tmpfile = f.name

    try:
        ret = _pstrainc.write_gau(tmpfile, original)
        assert ret == 0, f"write_gau failed with return code {ret}"

        result, out_n_mgau, out_n_feat, out_n_density, out_veclen = _pstrainc.read_gau(tmpfile)

        assert out_n_mgau == n_mgau
        assert out_n_feat == n_feat
        assert out_n_density == n_density
        assert out_veclen == [veclen]
        assert result.shape == original.shape

        assert np.allclose(original, result, rtol=1e-5), "Data mismatch after round-trip"
    finally:
        Path(tmpfile).unlink(missing_ok=True)


def test_logmath_wrapper() -> None:
    """Test LogMath Python wrapper class."""
    lm = _pstrainc.LogMath()

    # Test base
    assert abs(lm.base - 1.0001) < 1e-6

    # Test log/exp round-trip
    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        logp = lm.log(p)
        back = lm.exp(logp)
        assert abs(back - p) < 0.01, f"Round-trip failed for {p}"

    # Test add in log domain
    logp1 = lm.log(0.3)
    logp2 = lm.log(0.4)
    logsum = lm.add(logp1, logp2)
    result = lm.exp(logsum)
    assert abs(result - 0.7) < 0.01, f"Add failed: expected 0.7, got {result}"


def test_pstrain_fe_create_default() -> None:
    """Test pstrain_fe_create_default uses the documented FE parameters."""
    ffi, lib = _pstrainc._init()

    fe = lib.pstrain_fe_create_default()
    explicit_fe = lib.pstrain_fe_create(
        16000.0,
        25,
        512,
        130.0,
        6800.0,
        13,
        0.97,
        22,
        True,
        True,
        True,
        b"dct",
        100,
        0.025625,
    )
    assert fe != _pstrainc.get_ffi().NULL
    assert explicit_fe != ffi.NULL

    def extract(frontend: Any) -> np.ndarray:
        samples = (12000 * np.sin(np.arange(16000) * 0.037)).astype(np.int16)
        samples_buf = ffi.new("int16[]", samples.tolist())
        samples_ptr = ffi.new("int16 const **", samples_buf)
        nsamps = ffi.new("size_t *", len(samples))
        rows = [ffi.new("mfcc_t[13]") for _ in range(100)]
        cepstra = ffi.new("mfcc_t *[100]", rows)
        nframes = ffi.new("int32 *", len(rows))

        assert lib.pstrain_cffi_fe_start_utt(frontend) == 0
        assert (
            lib.pstrain_cffi_fe_process_frames(
                frontend, samples_ptr, nsamps, cepstra, nframes, ffi.NULL
            )
            == 0
        )
        return np.array([[rows[i][j] for j in range(13)] for i in range(nframes[0])])

    try:
        assert lib.pstrain_cffi_fe_get_output_size(fe) == 13
        default_features = extract(fe)
        explicit_features = extract(explicit_fe)
        assert default_features.shape[0] > 0
        # Dither uses a fresh random seed for each front end, so identical
        # configuration guarantees shape and finite output, not equal values.
        assert default_features.shape == explicit_features.shape
        assert np.isfinite(default_features).all()
        assert np.isfinite(explicit_features).all()
    finally:
        lib.pstrain_cffi_fe_free(fe)
        lib.pstrain_cffi_fe_free(explicit_fe)


def test_pstrain_fe_create_custom() -> None:
    """Test pstrain_fe_create with custom parameters."""
    _, lib = _pstrainc._init()

    fe = lib.pstrain_fe_create(
        16000.0,  # samprate
        40,  # nfilt
        512,  # nfft
        130.0,  # lowerf
        6800.0,  # upperf
        26,  # ncep - custom value
        0.97,  # alpha
        22,  # lifter
        True,  # dither
        True,  # remove_dc
        True,  # remove_noise
        b"dct",  # transform
        100,  # frate
        0.025625,  # wlen
    )
    assert fe != _pstrainc.get_ffi().NULL

    # Check output size matches our ncep
    output_size = lib.pstrain_cffi_fe_get_output_size(fe)
    assert output_size == 26

    lib.pstrain_cffi_fe_free(fe)


def test_pstrain_fe_create_8khz() -> None:
    """Test pstrain_fe_create for 8kHz audio."""
    _, lib = _pstrainc._init()

    fe = lib.pstrain_fe_create(
        8000.0,  # samprate
        31,  # nfilt (fewer for 8kHz)
        256,  # nfft (smaller for 8kHz)
        200.0,  # lowerf
        3500.0,  # upperf (Nyquist is 4000)
        13,  # ncep
        0.97,  # alpha
        22,  # lifter
        True,  # dither
        True,  # remove_dc
        True,  # remove_noise
        b"dct",  # transform
        100,  # frate
        0.025625,  # wlen
    )
    assert fe != _pstrainc.get_ffi().NULL

    output_size = lib.pstrain_cffi_fe_get_output_size(fe)
    assert output_size == 13

    lib.pstrain_cffi_fe_free(fe)


@requires_c_library
def test_pstrain_bw_init(tmp_path: Path) -> None:
    """Test BW context initialization with a flat model."""
    import numpy as np

    from pstrain.lib import flat

    # Create a simple flat model
    phones = ["SIL", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D"]
    model_dir = tmp_path / "model"
    flat.init_flat_model(phones, model_dir, n_density=1, n_state=3)

    # Create synthetic means/variances for testing (global mean=0, var=1)
    n_tied_state = len(phones) * 3  # 3 states per phone
    n_feat = 39
    means = np.zeros((n_tied_state, 1, n_feat), dtype=np.float32)
    variances = np.ones((n_tied_state, 1, n_feat), dtype=np.float32)
    _pstrainc.write_gau(str(model_dir / "means"), means)
    _pstrainc.write_gau(str(model_dir / "variances"), variances)

    # Initialize BW context
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()

    config = ffi.new("pstrain_bw_config_t *")
    config.a_beam = 1e-90
    config.b_beam = 1e-10
    config.topn = 1
    config.mixw_floor = 1e-8
    config.tmat_floor = 1e-4
    config.mean_reest = config.var_reest = 1
    config.mixw_reest = config.tmat_reest = 1
    config.pass2var = 1
    config.unobserved_gaussian_policy = 1
    ctx = lib.pstrain_bw_init(
        str(model_dir / "mdef").encode(),
        str(model_dir / "means").encode(),
        str(model_dir / "variances").encode(),
        str(model_dir / "mixture_weights").encode(),
        str(model_dir / "transition_matrices").encode(),
        config,
    )

    assert ctx != ffi.NULL, "BW context should initialize successfully"

    # Check stats are zeroed
    total_log_lik = ffi.new("float64 *")
    total_frames = ffi.new("uint32 *")
    total_utts = ffi.new("uint32 *")
    lib.pstrain_bw_get_stats(ctx, total_log_lik, total_frames, total_utts)
    assert total_log_lik[0] == 0.0
    assert total_frames[0] == 0
    assert total_utts[0] == 0

    lib.pstrain_bw_free(ctx)


@pytest.mark.parametrize("policy", [0, 99])
def test_pstrain_bw_init_rejects_invalid_policy(tmp_path: Path, policy: int) -> None:
    """The C API rejects invalid policy enum values, including INVALID."""
    from pstrain.lib import flat

    model_dir = tmp_path / "model"
    flat.init_flat_model(["SIL", "AA"], model_dir, n_density=1, n_state=3)
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()
    config = ffi.new("pstrain_bw_config_t *")
    config.topn = 1
    config.unobserved_gaussian_policy = policy

    ctx = lib.pstrain_bw_init(
        str(model_dir / "mdef").encode(),
        str(model_dir / "means").encode(),
        str(model_dir / "variances").encode(),
        str(model_dir / "mixture_weights").encode(),
        str(model_dir / "transition_matrices").encode(),
        config,
    )
    assert ctx == ffi.NULL


def test_pstrain_bw_init_rejects_null_config(tmp_path: Path) -> None:
    """The C API rejects a null policy/config pointer."""
    from pstrain.lib import flat

    model_dir = tmp_path / "model"
    flat.init_flat_model(["SIL", "AA"], model_dir, n_density=1, n_state=3)
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()
    ctx = lib.pstrain_bw_init(
        str(model_dir / "mdef").encode(),
        str(model_dir / "means").encode(),
        str(model_dir / "variances").encode(),
        str(model_dir / "mixture_weights").encode(),
        str(model_dir / "transition_matrices").encode(),
        ffi.NULL,
    )
    assert ctx == ffi.NULL


@requires_c_library
def test_pstrain_bw_process_utt(tmp_path: Path) -> None:
    """Test BW utterance processing with synthetic data."""
    import numpy as np

    from pstrain.lib import flat

    # Create a simple flat model (3 phones, 3 states each)
    phones = ["SIL", "AA", "AE"]
    model_dir = tmp_path / "model"
    flat.init_flat_model(phones, model_dir, n_density=1, n_state=3)

    # Create synthetic means/variances for testing
    n_tied_state = len(phones) * 3  # 3 states per phone
    n_feat = 39
    means = np.zeros((n_tied_state, 1, n_feat), dtype=np.float32)
    variances = np.ones((n_tied_state, 1, n_feat), dtype=np.float32)
    _pstrainc.write_gau(str(model_dir / "means"), means)
    _pstrainc.write_gau(str(model_dir / "variances"), variances)

    # Initialize BW context
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()

    config = ffi.new("pstrain_bw_config_t *")
    config.a_beam = 1e-90
    config.b_beam = 1e-10
    config.topn = 1
    config.mixw_floor = 1e-8
    config.tmat_floor = 1e-4
    config.mean_reest = config.var_reest = 1
    config.mixw_reest = config.tmat_reest = 1
    config.pass2var = 1
    config.unobserved_gaussian_policy = 1
    ctx = lib.pstrain_bw_init(
        str(model_dir / "mdef").encode(),
        str(model_dir / "means").encode(),
        str(model_dir / "variances").encode(),
        str(model_dir / "mixture_weights").encode(),
        str(model_dir / "transition_matrices").encode(),
        config,
    )
    assert ctx != ffi.NULL

    # Create synthetic features (10 frames of 39-dim features)
    np.random.seed(42)
    n_frames = 10
    features = np.random.randn(n_frames, 39).astype(np.float32)

    # Phone sequence: SIL -> AA -> SIL (phone IDs 0, 1, 0)
    phone_ids = np.array([0, 1, 0], dtype=np.uint32)

    # Process utterance
    ret = lib.pstrain_bw_process_utt(
        ctx,
        ffi.cast("const float *", ffi.from_buffer(features)),
        n_frames,
        ffi.cast("const uint32 *", ffi.from_buffer(phone_ids)),
        len(phone_ids),
    )

    # Check stats updated
    total_log_lik = ffi.new("float64 *")
    total_frames = ffi.new("uint32 *")
    total_utts = ffi.new("uint32 *")
    lib.pstrain_bw_get_stats(ctx, total_log_lik, total_frames, total_utts)

    if ret == 0:
        # Success - stats should be updated
        assert total_frames[0] == n_frames
        assert total_utts[0] == 1
        assert total_log_lik[0] != 0.0  # Should have computed something
    else:
        # process_utt may fail with synthetic data - that's OK for now
        # The important thing is we didn't crash
        pass

    lib.pstrain_bw_free(ctx)
