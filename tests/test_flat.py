"""Tests for flat model initialization via CFFI."""

from __future__ import annotations

import logging
import os
import struct
import tempfile
from pathlib import Path

import pytest

from pstrain.lib import _pstrainc
from pstrain.lib.flat import (
    create_mdef,
    create_mixture_weights,
    create_topology_file,
    create_transition_matrices,
    init_flat_model,
    init_gaussians,
)
from tests.conftest import requires_c_library

# Skip all tests if C library not available
pytestmark = requires_c_library


def _write_feature_file(path: Path, n_frame: int = 10, ceplen: int = 13) -> None:
    """Write a well formed, all-zero Sphinx feature file."""
    with path.open("wb") as f:
        f.write(struct.pack("<i", n_frame * ceplen))
        f.write(b"\0" * (n_frame * ceplen * 4))


class TestCreateMdef:
    """Tests for model definition creation."""

    def test_create_mdef_basic(self) -> None:
        """Test basic mdef creation."""
        phones = ["SIL", "AA", "AE", "AH"]

        with tempfile.TemporaryDirectory() as tmpdir:
            mdef_path = Path(tmpdir) / "mdef"
            result = create_mdef(phones, output_path=mdef_path)

            assert result["n_phones"] == 4
            assert result["n_state"] == 3
            assert result["n_tied_state"] == 12  # 4 phones * 3 states
            assert result["phones"] == phones

    def test_create_mdef_writes_file(self) -> None:
        """Test that mdef is written to file."""
        phones = ["SIL", "AA", "AE"]

        with tempfile.TemporaryDirectory() as tmpdir:
            mdef_path = Path(tmpdir) / "mdef"
            create_mdef(phones, output_path=mdef_path)

            assert mdef_path.exists()
            content = mdef_path.read_text()
            assert "0.3" in content  # version
            for phone in phones:
                assert phone in content

    def test_create_mdef_custom_states(self) -> None:
        """Test mdef with custom state count."""
        phones = ["SIL", "AA"]

        with tempfile.TemporaryDirectory() as tmpdir:
            mdef_path = Path(tmpdir) / "mdef"
            result = create_mdef(phones, n_state=5, output_path=mdef_path)

            assert result["n_state"] == 5
            assert result["n_tied_state"] == 10  # 2 phones * 5 states


class TestCreateTopologyFile:
    """Tests for topology file creation."""

    def test_create_topology_basic(self) -> None:
        """Test basic topology file creation."""
        content = create_topology_file(n_state=3)

        lines = content.strip().split("\n")
        assert lines[0] == "0.1"  # version
        assert lines[1] == "4"  # n_state + 1 (emitting + exit)
        assert len(lines) == 5  # version + n_state + 3 transition rows

    def test_create_topology_writes_file(self) -> None:
        """Test that topology file is written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topo_path = Path(tmpdir) / "topo"
            create_topology_file(n_state=3, output_path=topo_path)

            assert topo_path.exists()
            content = topo_path.read_text()
            assert "0.1" in content


class TestCreateTransitionMatrices:
    """Tests for transition matrix creation via CFFI."""

    def test_create_tmat_via_cffi(self) -> None:
        """Test transition matrix creation through CFFI.

        SphinxTrain convention is one transition matrix per phone, so for
        a phoneset of N phones we expect n_tmat == N.
        """
        phones = ["SIL", "AA", "AE"]

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Create mdef and topo
            create_mdef(phones, n_state=3, output_path=tmpdir / "mdef")
            create_topology_file(n_state=3, output_path=tmpdir / "topo")

            # Create tmat via CFFI
            tmat_path = tmpdir / "tmat"
            create_transition_matrices(
                tmpdir / "mdef",
                tmpdir / "topo",
                tmat_path,
            )

            assert tmat_path.exists()
            assert tmat_path.stat().st_size > 0

            # Read back via CFFI
            tmat, n_tmat, n_state = _pstrainc.read_tmat_counts(str(tmat_path))
            assert n_tmat == len(phones)  # one tmat per phone
            assert n_state == 4  # 3 emitting + 1 exit


class TestCreateMixtureWeights:
    """Tests for mixture weights creation via CFFI."""

    def test_create_mixw_via_cffi(self) -> None:
        """Test mixture weights creation through CFFI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mixw_path = Path(tmpdir) / "mixw"
            n_tied_state = 30
            n_density = 4

            create_mixture_weights(
                n_tied_state, n_stream=1, n_density=n_density, output_path=mixw_path
            )

            assert mixw_path.exists()

            # Read back via CFFI
            mixw, out_n_mixw, out_n_feat, out_n_density = _pstrainc.read_mixw_counts(str(mixw_path))
            assert out_n_mixw == n_tied_state
            assert out_n_density == n_density

    def test_create_mixw_uniform_distribution(self) -> None:
        """Test that mixture weights are uniform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mixw_path = Path(tmpdir) / "mixw"
            n_tied_state = 10
            n_density = 8

            create_mixture_weights(
                n_tied_state, n_stream=1, n_density=n_density, output_path=mixw_path
            )

            # Read back
            mixw, _, _, _ = _pstrainc.read_mixw_counts(str(mixw_path))

            # Check uniform
            import numpy as np

            expected = 1.0 / n_density
            assert np.allclose(mixw, expected, rtol=1e-5)


class TestInitFlatModel:
    """Tests for full flat model initialization."""

    @pytest.fixture
    def phones(self) -> list[str]:
        """Sample phone set."""
        return ["SIL", "AA", "AE", "AH", "AO", "AW", "AY"]

    def test_init_gaussians_logs_and_counts_missing_features(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A feature file that is not there is counted as a skip, not a success."""
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        ctl = tmp_path / "train.ctl"
        ctl.write_text("present\nmissing\n")
        _write_feature_file(cep_dir / "present.mfc")

        with caplog.at_level(logging.WARNING):
            skipped = init_gaussians(
                ctl,
                cep_dir,
                tmp_path / "accum",
                max_skip_fraction=0.5,
            )

        assert skipped == 1
        assert "Flat initialization skipped 1/2 utterances (50.00%)" in caplog.text

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX permission bits do not gate reads on Windows"
    )
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root reads files regardless of permission bits",
    )
    def test_init_gaussians_logs_and_counts_unreadable_features(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A feature file that exists but cannot be opened is also a skip.

        Distinct from the missing-file case above: this file is present and
        well formed, and only its permission bits keep it out.
        """
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        ctl = tmp_path / "train.ctl"
        ctl.write_text("readable\nunreadable\n")
        _write_feature_file(cep_dir / "readable.mfc")
        unreadable = cep_dir / "unreadable.mfc"
        _write_feature_file(unreadable)
        unreadable.chmod(0o000)

        try:
            with caplog.at_level(logging.WARNING):
                skipped = init_gaussians(
                    ctl,
                    cep_dir,
                    tmp_path / "accum",
                    max_skip_fraction=0.5,
                )
        finally:
            unreadable.chmod(0o644)

        assert unreadable.exists()
        assert skipped == 1
        assert "Flat initialization skipped 1/2 utterances (50.00%)" in caplog.text

    def test_init_gaussians_applies_training_skip_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctl = tmp_path / "train.ctl"
        ctl.write_text("present\nmissing\n")
        monkeypatch.setattr("pstrain.lib.flat._init_gaussians_native", lambda *args: 1)

        with pytest.raises(RuntimeError, match=r"skipped 1/2.*above configured limit 49.00%"):
            init_gaussians(
                ctl,
                tmp_path,
                tmp_path / "accum",
                max_skip_fraction=0.49,
            )

    def test_init_flat_model_creates_files(self, phones: list[str]) -> None:
        """Test that init_flat_model creates all required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = init_flat_model(phones, output_dir, n_density=1)

            assert (output_dir / "mdef").exists()
            assert (output_dir / "topo").exists()
            assert (output_dir / "mixture_weights").exists()
            assert (output_dir / "transition_matrices").exists()

            # Check return value
            assert result["mdef"] == output_dir / "mdef"
            assert result["transition_matrices"] == output_dir / "transition_matrices"

    def test_init_flat_model_readable(self, phones: list[str]) -> None:
        """Test that created files can be read back via CFFI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            init_flat_model(phones, output_dir, n_density=1)

            # Read back using CFFI
            mixw, n_mixw, n_feat, n_density = _pstrainc.read_mixw_counts(
                str(output_dir / "mixture_weights")
            )
            assert n_mixw > 0
            assert n_density == 1

            tmat, n_tmat, n_state = _pstrainc.read_tmat_counts(
                str(output_dir / "transition_matrices")
            )
            assert n_tmat > 0
            assert n_state > 0

    def test_init_flat_model_different_densities(self, phones: list[str]) -> None:
        """Test flat model with different Gaussian densities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for n_density in [1, 2, 4]:
                subdir = Path(tmpdir) / f"density_{n_density}"

                init_flat_model(phones, subdir, n_density=n_density)

                mixw, n_mixw, n_feat_out, out_density = _pstrainc.read_mixw_counts(
                    str(subdir / "mixture_weights")
                )
                assert out_density == n_density
