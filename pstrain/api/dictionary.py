"""Public dictionary-source API."""

from pstrain.lib.dictionary.cmudict_source import (
    CMUDictSource,
    default_cmudict_cache,
    fetch_cmudict,
)

__all__ = ["CMUDictSource", "default_cmudict_cache", "fetch_cmudict"]
