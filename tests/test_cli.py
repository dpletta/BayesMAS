"""CLI smoke tests (no GPU, no network)."""

from __future__ import annotations

import json

import pytest

from bayesmas.cli import main


def test_simulate_json_emits_the_four_protocols(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--setting", "low", "--trials", "3", "--seed", "0", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    names = {row["protocol"] for row in rows}
    assert names == {"fixed", "dylan", "bayesmas", "bayesmas-f"}
    assert all(row["setting"] == "low" for row in rows)


def test_puzzles_json_emits_hanoi_rows(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "puzzles",
                "--puzzle",
                "hanoi",
                "--n",
                "2",
                "--trials",
                "2",
                "--seed",
                "0",
                "--json",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert rows
    assert all(row["puzzle"] == "hanoi" for row in rows)
    assert all(row["n"] == 2 for row in rows)


def test_unknown_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-command"])
