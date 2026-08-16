"""Path discovery for pstrain components.

Provides functions to find pstrain binaries, libraries, and data directories
across different installation scenarios (development, pip install, system).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PstrainPaths", "get_paths", "get_bin_dir", "get_lib_path", "get_include_dir"]


@dataclass
class PstrainPaths:
    """Collection of pstrain installation paths."""

    bin_dir: Path | None
    """Directory containing pstrain C binaries (bw, norm, etc.)"""

    lib_path: Path | None
    """Path to libpstrainc shared library"""

    include_dir: Path | None
    """Directory containing pstrain C headers"""

    project_root: Path | None
    """Project root (development only)"""

    data_dir: Path
    """Directory containing package data files"""

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON output."""
        return {
            "bin_dir": str(self.bin_dir) if self.bin_dir else None,
            "lib_path": str(self.lib_path) if self.lib_path else None,
            "include_dir": str(self.include_dir) if self.include_dir else None,
            "project_root": str(self.project_root) if self.project_root else None,
            "data_dir": str(self.data_dir),
        }


def _get_project_root() -> Path | None:
    """Get project root if running from development checkout."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).parent
    for _ in range(5):  # Don't go too far up
        if (current / "pyproject.toml").exists() and (current / "csrc").exists():
            return current
        current = current.parent
    return None


def _get_bundled_lib_dir() -> Path | None:
    """Get the bundled _lib directory if it exists (wheel install)."""
    lib_dir = Path(__file__).parent.parent / "_lib"
    if lib_dir.is_dir():
        return lib_dir
    return None


def _find_bin_dir() -> Path | None:
    """Find directory containing pstrain binaries.

    Search order:
    1. PSTRAIN_BIN_DIR environment variable
    2. Bundled in wheel (pstrain/_lib/bin/)
    3. Development build directory (build/bin/)
    4. System libexec (PREFIX/libexec/pstrainc/)
    5. Homebrew libexec
    """
    # 1. Environment variable override
    if "PSTRAIN_BIN_DIR" in os.environ:
        bin_dir = Path(os.environ["PSTRAIN_BIN_DIR"])
        if bin_dir.is_dir():
            return bin_dir

    # 2. Bundled in wheel
    bundled = _get_bundled_lib_dir()
    if bundled:
        bin_dir = bundled / "bin"
        if bin_dir.is_dir() and any(bin_dir.iterdir()):
            return bin_dir

    # 3. Development build. Multi-config Windows generators place both the
    #    DLL and CLI executables in a configuration subdirectory.
    project_root = _get_project_root()
    if project_root:
        dev_bin = project_root / "build" / "bin"
        candidates = []
        if sys.platform == "win32":
            candidates.extend(
                dev_bin / configuration
                for configuration in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel")
            )
        candidates.append(dev_bin)
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate

    # 4. System libexec locations
    prefixes = [
        Path(sys.prefix),  # Virtual env or system Python prefix
        Path("/usr/local"),
        Path("/usr"),
    ]

    # Add Homebrew prefix on macOS
    if sys.platform == "darwin":
        homebrew_prefixes = [
            Path("/opt/homebrew"),  # Apple Silicon
            Path("/usr/local"),  # Intel
        ]
        prefixes = homebrew_prefixes + prefixes

    for prefix in prefixes:
        libexec_dir = prefix / "libexec" / "pstrainc"
        if libexec_dir.is_dir():
            return libexec_dir
        # Also check bin/ in case installed there
        bin_dir = prefix / "bin"
        if (bin_dir / "bw").exists() or (bin_dir / "norm").exists():
            return bin_dir

    return None


def _find_lib_path() -> Path | None:
    """Find the pstrainc shared library.

    Search order:
    1. PSTRAIN_LIB_PATH environment variable
    2. Bundled in wheel (pstrain/_lib/lib/)
    3. Development build directory
    4. LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
    5. System library paths
    """
    if sys.platform == "win32":
        lib_names = ["pstrainc.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libpstrainc.dylib"]
    else:
        lib_names = ["libpstrainc.so"]

    # 1. Environment variable override
    if "PSTRAIN_LIB_PATH" in os.environ:
        lib_path = Path(os.environ["PSTRAIN_LIB_PATH"])
        if lib_path.is_file():
            return lib_path

    # 2. Bundled in wheel. manylinux (RPM-based) installs to lib64 via
    #    GNUInstallDirs, macOS/others to lib — check both.
    bundled = _get_bundled_lib_dir()
    if bundled:
        for libdir in ("lib", "lib64", "bin"):
            for name in lib_names:
                candidate = bundled / libdir / name
                if candidate.exists():
                    return candidate

    # 2b. Wheel repaired by auditwheel (Linux) or delocate (macOS): the bundled
    #     shared library may be relocated out of pstrain/_lib/lib into a vendored
    #     sibling directory under a hashed name (e.g. pstrain.libs/libpstrainc-abc.so).
    #     Match it by glob and return the full path (dlopen by path handles the
    #     hashed name).
    pkg_dir = Path(__file__).resolve().parent.parent  # .../pstrain
    vendored_dirs = [
        pkg_dir.parent / "pstrain.libs",  # auditwheel
        pkg_dir / ".dylibs",  # delocate
    ]
    for vendored in vendored_dirs:
        if vendored.is_dir():
            if sys.platform == "win32":
                patterns = ["pstrainc*.dll"]
            elif sys.platform == "darwin":
                patterns = ["libpstrainc*.dylib"]
            else:
                patterns = ["libpstrainc*.so"]
            for pattern in patterns:
                hits = sorted(vendored.glob(pattern))
                if hits:
                    return hits[0]

    # 3. Development build. On Windows the shared library is a RUNTIME
    #    artifact. Multi-config generators append the selected configuration
    #    to CMAKE_RUNTIME_OUTPUT_DIRECTORY, so search those directories before
    #    the unsuffixed runtime directory.
    project_root = _get_project_root()
    if project_root:
        subdirs = ["build/lib"]
        if sys.platform == "win32":
            subdirs.extend(
                f"build/bin/{configuration}"
                for configuration in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel")
            )
        subdirs.extend(["build/bin", "build"])
        for subdir in subdirs:
            for name in lib_names:
                candidate = project_root / subdir / name
                if candidate.exists():
                    return candidate

    # 4. Library path environment variables
    for env_var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
        if env_var in os.environ:
            for path_str in os.environ[env_var].split(":"):
                path = Path(path_str)
                for name in lib_names:
                    candidate = path / name
                    if candidate.exists():
                        return candidate

    # 5. System library paths
    system_lib_dirs = [
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        Path("/opt/homebrew/lib"),
    ]

    for lib_dir in system_lib_dirs:
        for name in lib_names:
            candidate = lib_dir / name
            if candidate.exists():
                return candidate

    return None


def _find_include_dir() -> Path | None:
    """Find directory containing pstrain C headers.

    Search order:
    1. PSTRAIN_INCLUDE_DIR environment variable
    2. Bundled in wheel (pstrain/_lib/include/)
    3. Development source directory (csrc/include/)
    4. System include paths (PREFIX/include/pstrain/)
    """
    # 1. Environment variable override
    if "PSTRAIN_INCLUDE_DIR" in os.environ:
        inc_dir = Path(os.environ["PSTRAIN_INCLUDE_DIR"])
        if inc_dir.is_dir():
            return inc_dir

    # 2. Bundled in wheel
    bundled = _get_bundled_lib_dir()
    if bundled:
        inc_dir = bundled / "include"
        if inc_dir.is_dir():
            return inc_dir

    # 3. Development source
    project_root = _get_project_root()
    if project_root:
        dev_include = project_root / "csrc" / "include"
        if dev_include.is_dir():
            return dev_include

    # 4. System include locations
    prefixes = [
        Path(sys.prefix),
        Path("/usr/local"),
        Path("/usr"),
    ]

    if sys.platform == "darwin":
        homebrew_prefixes = [
            Path("/opt/homebrew"),
            Path("/usr/local"),
        ]
        prefixes = homebrew_prefixes + prefixes

    for prefix in prefixes:
        inc_dir = prefix / "include" / "pstrain"
        if inc_dir.is_dir():
            return inc_dir

    return None


def _get_data_dir() -> Path:
    """Get the package data directory."""
    return Path(__file__).parent.parent / "data"


def get_paths() -> PstrainPaths:
    """Get all pstrain paths.

    Returns:
        PstrainPaths with discovered locations (None if not found)
    """
    return PstrainPaths(
        bin_dir=_find_bin_dir(),
        lib_path=_find_lib_path(),
        include_dir=_find_include_dir(),
        project_root=_get_project_root(),
        data_dir=_get_data_dir(),
    )


def get_bin_dir() -> Path | None:
    """Get the directory containing pstrain binaries.

    Returns:
        Path to binary directory, or None if not found

    Example:
        >>> bin_dir = get_bin_dir()
        >>> if bin_dir:
        ...     bw_path = bin_dir / "bw"
    """
    return _find_bin_dir()


def get_lib_path() -> Path | None:
    """Get the path to the pstrainc shared library.

    Returns:
        Path to libpstrainc, or None if not found
    """
    return _find_lib_path()


def get_include_dir() -> Path | None:
    """Get the directory containing pstrain C headers.

    Returns:
        Path to include directory, or None if not found

    Example:
        >>> inc_dir = get_include_dir()
        >>> if inc_dir:
        ...     # Use with compiler: -I{inc_dir}
        ...     pass
    """
    return _find_include_dir()
