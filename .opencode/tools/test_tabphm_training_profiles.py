import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import PICID_training as training


def _write_profiles(repo_root: Path, reference_body: str = "") -> None:
    profiles_path = repo_root / ".opencode" / "reference" / "hparam-profiles.yaml"
    profiles_path.parent.mkdir(parents=True)
    profiles_path.write_text(
        f"""
profiles:
  reference:
    by_experiment:
{reference_body or "      {}"}
    by_scope_model: {{}}
    by_scope: {{}}
    by_dataset: {{}}
  paper:
    overrides: {{}}
""".lstrip(),
        encoding="utf-8",
    )


FIXED_BATCH_OVERRIDES = [
    "datamodule.train_batch_size=512",
    "datamodule.val_batch_size=1024",
    "datamodule.test_batch_size=1024",
]


def test_reference_profile_without_keyed_entry_returns_fixed_validation_batch(tmp_path):
    _write_profiles(tmp_path)

    overrides = training.profile_overrides(
        tmp_path,
        "reference",
        "nb14/prognostics/nb14_autoencoder_pretrain",
        "feedforward",
        None,
    )

    assert overrides == FIXED_BATCH_OVERRIDES


def test_reference_profile_keeps_explicit_runtime_overrides_without_lookup(tmp_path):
    overrides = training.profile_overrides(
        tmp_path,
        "reference",
        "nb14/prognostics/nb14_autoencoder_pretrain",
        "feedforward",
        ["optimization.lr=0.001"],
    )

    assert overrides == ["optimization.lr=0.001", *FIXED_BATCH_OVERRIDES]


def test_reference_profile_still_uses_keyed_entries_when_present(tmp_path):
    _write_profiles(
        tmp_path,
        """
      nb14/prognostics/existing_model:
        overrides:
          optimization.lr: 0.0001
          trainer.max_epochs: 100
""",
    )

    overrides = training.profile_overrides(
        tmp_path,
        "reference",
        "nb14/prognostics/existing_model",
        "feedforward",
        None,
    )

    assert overrides == ["optimization.lr=0.0001", "trainer.max_epochs=100", *FIXED_BATCH_OVERRIDES]


def test_reference_profile_fixed_batch_wins_over_conflicting_runtime_batch(tmp_path):
    overrides = training.profile_overrides(
        tmp_path,
        "reference",
        "nb14/prognostics/nb14_autoencoder_pretrain",
        "feedforward",
        ["datamodule.train_batch_size=16", "optimization.lr=0.001"],
    )

    assert overrides == ["optimization.lr=0.001", *FIXED_BATCH_OVERRIDES]


def test_paper_profile_also_gets_fixed_validation_batch(tmp_path):
    _write_profiles(tmp_path)

    overrides = training.profile_overrides(
        tmp_path,
        "paper",
        "nb14/prognostics/nb14_autoencoder_pretrain",
        "feedforward",
        None,
    )

    assert overrides == FIXED_BATCH_OVERRIDES

    conflicting_overrides = training.profile_overrides(
        tmp_path,
        "paper",
        "nb14/prognostics/nb14_autoencoder_pretrain",
        "feedforward",
        ["datamodule.train_batch_size=16", "optimization.lr=0.001"],
    )

    assert conflicting_overrides == ["optimization.lr=0.001", *FIXED_BATCH_OVERRIDES]


def test_reference_profile_status_distinguishes_composed_vs_keyed():
    assert (
        training.hp_profile_status("reference", [])
        == "FIXED_VALIDATION_BATCH_WITH_PICID_SCHEDULER_AND_PAPER_OR_IMPUTED_MODEL_HPS"
    )
    assert (
        training.hp_profile_status("reference", ["optimization.lr=0.001"])
        == "KEYED_REFERENCE_OVERRIDES_WITH_FIXED_VALIDATION_BATCH"
    )
    assert (
        training.hp_profile_status("paper", [])
        == "PAPER_OR_DEFAULT_PROFILE_WITH_FIXED_VALIDATION_BATCH"
    )


def test_invalid_hp_overrides_still_writes_failure_log(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    result = training.run_training(
        {
            "vault_dir": str(vault_dir),
            "repo_root": str(tmp_path),
            "experiment_config": "nb14/prognostics/nb14_autoencoder_pretrain",
            "hp_profile": "reference",
            "hp_overrides": "not-a-list",
            "epoch_budget_rationale": "x" * 80,
        }
    )

    assert result["status"] == "FAILED"
    assert result["traceback_excerpt"] == (
        "hp_overrides must be a list of Hydra override strings or a mapping"
    )
    assert (vault_dir / "07-training-log.md").is_file()
    records = [
        json.loads(line)
        for line in (vault_dir / "07-training-log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert records[-1]["hp_profile"] == "reference"
