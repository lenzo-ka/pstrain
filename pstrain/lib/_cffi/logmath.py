"""Log-domain math wrapper.

Uses C library for numerical stability in probability computations.
"""

from __future__ import annotations

from pstrain.lib._cffi.core import _init


class LogMath:
    """Log-domain math using C library for numerical stability.

    This wraps the C logmath functions which use integer log representations
    for fast, stable computation of probabilities in log domain.
    """

    def __init__(self, base: float = 1.0001, shift: int = 0, use_table: bool = True) -> None:
        """Initialize log math.

        Args:
            base: Log base (default 1.0001 for high precision)
            shift: Bit shift for table lookup
            use_table: Whether to use lookup table for speed
        """
        ffi, lib = _init()
        self._ffi = ffi
        self._lib = lib
        self._lmath = lib.logmath_init(base, shift, 1 if use_table else 0)
        if self._lmath == ffi.NULL:
            raise RuntimeError("Failed to initialize logmath")

    def __del__(self) -> None:
        """Free C resources."""
        if hasattr(self, "_lmath") and self._lmath is not None:
            self._lib.logmath_free(self._lmath)

    def log(self, p: float) -> int:
        """Convert probability to log domain.

        Args:
            p: Probability value (0 < p <= 1)

        Returns:
            Log-domain integer representation
        """
        result: int = self._lib.logmath_log(self._lmath, p)
        return result

    def exp(self, logp: int) -> float:
        """Convert log-domain value back to probability.

        Args:
            logp: Log-domain integer

        Returns:
            Probability value
        """
        result: float = self._lib.logmath_exp(self._lmath, logp)
        return result

    def add(self, logp: int, logq: int) -> int:
        """Add two probabilities in log domain.

        Computes log(exp(logp) + exp(logq)) efficiently.

        Args:
            logp: First log probability
            logq: Second log probability

        Returns:
            log(p + q)
        """
        result: int = self._lib.logmath_add(self._lmath, logp, logq)
        return result

    @property
    def base(self) -> float:
        """Get the log base."""
        result: float = self._lib.logmath_get_base(self._lmath)
        return result
