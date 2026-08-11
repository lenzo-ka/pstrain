"""Model definition (mdef) generation.

Generates mdef files for different training stages:
- CI mdef: context-independent phones only
- All-triphones mdef: all possible triphones from dictionary
- Untied mdef: triphones observed in transcripts (pruned by threshold)
"""

from __future__ import annotations

from pathlib import Path

from pstrain.lib import _pstrainc, native_worker
from pstrain.lib.transcription import parse_transcription_file


def generate_ci_mdef(
    phone_list: Path,
    output: Path,
    n_state: int = 3,
) -> None:
    """Generate CI (context-independent) mdef from phone list.

    Args:
        phone_list: Path to phone list file (one phone per line)
        output: Output mdef file path
        n_state: Number of emitting states per phone (typically 3)

    This operation is routed through the persistent native worker, so a
    missing or malformed phone list cannot terminate the calling interpreter.

    Raises:
        PstrainNativeError: If generation fails. ``PstrainNativeCrashError``
            when the native code died on a signal, ``PstrainNativeFatalError``
            when it reported a diagnosed failure.
    """
    native_worker.call(
        "mdef_gen_ci",
        (str(phone_list), str(output), n_state),
        (phone_list,),
    )


def generate_alltriphones_mdef(
    phone_list: Path,
    dict_path: Path,
    output: Path,
    filler_dict: Path | None = None,
    n_state: int = 3,
    ignore_wpos: bool = False,
) -> None:
    """Generate all-triphones mdef from dictionary.

    Creates mdef with all possible triphones from dictionary entries.

    Args:
        phone_list: Path to CI phone list
        dict_path: Path to pronunciation dictionary
        output: Output mdef file path
        filler_dict: Path to filler dictionary (optional)
        n_state: Number of emitting states per phone
        ignore_wpos: If True, ignore word position in triphones

    Raises:
        RuntimeError: If generation fails
    """
    lib = _pstrainc.get_lib()
    ret = lib.pstrain_mdef_gen_alltriphones(
        str(phone_list).encode(),
        str(dict_path).encode(),
        _pstrainc.path_or_null(filler_dict),
        str(output).encode(),
        n_state,
        1 if ignore_wpos else 0,
    )
    if ret != 0:
        raise RuntimeError(f"Failed to generate all-triphones mdef: {output}")


def generate_untied_mdef(
    phone_list: Path,
    dict_path: Path,
    transcript: Path,
    output: Path,
    filler_dict: Path | None = None,
    n_state: int = 3,
    ignore_wpos: bool = False,
    multipron: bool = True,
    inventory_policy: str | None = None,
) -> None:
    """Generate untied mdef from transcripts.

    The default policy preserves current behavior: all dictionary-producible
    triphones in multipron mode and occurrence pruning in linear mode.

    Args:
        phone_list: Path to CI phone list
        dict_path: Path to pronunciation dictionary
        transcript: Path to transcript file
        output: Output mdef file path
        filler_dict: Path to filler dictionary (optional)
        n_state: Number of emitting states per phone
        ignore_wpos: If True, ignore word position
        multipron: Whether the downstream trainer uses pronunciation graphs
        inventory_policy: ``all-triphone``, ``transcript-reachable``, or
            ``linear``. If omitted, derives the current policy from multipron.

    Raises:
        RuntimeError: If generation fails
    """
    if inventory_policy is None:
        inventory_policy = "all-triphone" if multipron else "linear"
    policies = {"linear": 0, "all-triphone": 1, "transcript-reachable": 2}
    if inventory_policy not in policies:
        choices = ", ".join(sorted(policies))
        raise ValueError(f"unknown untied inventory policy {inventory_policy!r}; choose {choices}")

    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()
    parsed_utterances: list[object] = []
    utterances_c = ffi.NULL
    n_utterances = 0
    if inventory_policy == "transcript-reachable":
        parsed_utterances = [
            ffi.new("char[]", text.encode())
            for text in parse_transcription_file(transcript).values()
        ]
        utterances_c = ffi.new("char*[]", parsed_utterances)
        n_utterances = len(parsed_utterances)
    ret = lib.pstrain_mdef_gen_untied(
        str(phone_list).encode(),
        str(dict_path).encode(),
        _pstrainc.path_or_null(filler_dict),
        str(transcript).encode(),
        str(output).encode(),
        n_state,
        1 if ignore_wpos else 0,
        policies[inventory_policy],
        1 if multipron else 0,
        utterances_c,
        n_utterances,
    )
    if ret != 0:
        raise RuntimeError(f"Failed to generate untied mdef: {output}")


def count_triphones(
    phone_list: Path,
    dict_path: Path,
    transcript: Path,
    output: Path,
    filler_dict: Path | None = None,
    ignore_wpos: bool = False,
) -> None:
    """Count triphones in transcripts.

    Args:
        phone_list: Path to CI phone list
        dict_path: Path to pronunciation dictionary
        transcript: Path to transcript file
        output: Output counts file path
        filler_dict: Path to filler dictionary (optional)
        ignore_wpos: If True, ignore word position

    Raises:
        RuntimeError: If counting fails
    """
    lib = _pstrainc.get_lib()
    ret = lib.pstrain_mdef_count_triphones(
        str(phone_list).encode(),
        str(dict_path).encode(),
        _pstrainc.path_or_null(filler_dict),
        str(transcript).encode(),
        str(output).encode(),
        1 if ignore_wpos else 0,
    )
    if ret != 0:
        raise RuntimeError(f"Failed to count triphones: {output}")
