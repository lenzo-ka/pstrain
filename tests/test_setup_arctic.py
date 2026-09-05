from pathlib import Path

import pytest

from scripts import setup_arctic

# More than ten distinct out-of-vocabulary words, so that a manifest holding
# only the first ten -- the behavior this replaced -- cannot pass. Two of the
# words appear in more than one utterance, so the per-word utterance lists are
# not all singletons either.
TRANSCRIPTS = {
    "utt1": "Known oova oovb oovc oovd",
    "utt2": "Oova known oove oovf oovg oovh",
    "utt3": "Oovb oovi oovj oovk oovl known",
}
EXPECTED_MISSING = {
    "OOVA": ["utt1", "utt2"],
    "OOVB": ["utt1", "utt3"],
    "OOVC": ["utt1"],
    "OOVD": ["utt1"],
    "OOVE": ["utt2"],
    "OOVF": ["utt2"],
    "OOVG": ["utt2"],
    "OOVH": ["utt2"],
    "OOVI": ["utt3"],
    "OOVJ": ["utt3"],
    "OOVK": ["utt3"],
    "OOVL": ["utt3"],
}


def test_setup_records_all_missing_vocabulary_and_keeps_utterances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arctic = tmp_path / "arctic"
    (arctic / "etc").mkdir(parents=True)
    (arctic / "wav").mkdir()
    (arctic / "etc" / "txt.done.data").write_text(
        "".join(f'( {utt_id} "{text}" )\n' for utt_id, text in TRANSCRIPTS.items())
    )
    for utt_id in TRANSCRIPTS:
        (arctic / "wav" / f"{utt_id}.wav").write_bytes(b"wav")

    cmudict = tmp_path / "cmudict.dict"
    cmudict.write_text("KNOWN N OW N\n")
    experiment = tmp_path / "experiments" / "default"
    monkeypatch.setattr(setup_arctic, "ARCTIC_DIR", arctic)
    monkeypatch.setattr(setup_arctic, "CMUDICT_PATH", cmudict)
    monkeypatch.setattr(setup_arctic, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(setup_arctic, "SHARED_DIR", tmp_path / "shared")
    monkeypatch.setattr(setup_arctic, "EXPERIMENT_DIR", experiment)

    setup_arctic.main()

    missing_path = experiment / "etc" / "missing_vocabulary.txt"
    recorded = {
        line.split("\t")[0]: line.split("\t")[1:] for line in missing_path.read_text().splitlines()
    }
    # Every missing word, with the utterances that contain it -- not a prefix.
    assert recorded == EXPECTED_MISSING

    output = capsys.readouterr().out
    assert (
        f"missing_vocabulary\twords={len(EXPECTED_MISSING)}\t"
        f"utterances_kept={len(TRANSCRIPTS)}\tpath={missing_path}"
    ) in output
    generated_fileids = {
        *(experiment / "etc" / "train.fileids").read_text().splitlines(),
        *(experiment / "etc" / "test.fileids").read_text().splitlines(),
    }
    assert generated_fileids == set(TRANSCRIPTS)
