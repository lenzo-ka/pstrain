"""Dictionary utilities.

Core classes:
- Dictionary: Base class for pronunciation dictionaries (phoneset-agnostic)
- CMUDict: Subclass with ARPABET stress handling

For Phoneset and phone mapping, see pstrain.lib.phoneset.
"""

from pstrain.lib.dictionary.cmudict import (
    ARPABET_CONSONANTS,
    ARPABET_PHONES,
    ARPABET_VOWELS,
    CMUDict,
    get_stress,
    is_vowel,
    strip_dictionary_stress,
    strip_phoneset_stress,
    strip_stress,
)
from pstrain.lib.dictionary.dictionary import Dictionary
from pstrain.lib.dictionary.filler import (
    create_standard_filler_dict,
    get_filler_phones,
    get_standard_fillers,
    get_standard_fillers_text,
)

__all__ = [
    "Dictionary",
    # Filler
    "create_standard_filler_dict",
    "get_filler_phones",
    "get_standard_fillers",
    "get_standard_fillers_text",
    # CMUDict / ARPABET
    "ARPABET_CONSONANTS",
    "ARPABET_PHONES",
    "ARPABET_VOWELS",
    "CMUDict",
    "get_stress",
    "is_vowel",
    "strip_dictionary_stress",
    "strip_phoneset_stress",
    "strip_stress",
]
