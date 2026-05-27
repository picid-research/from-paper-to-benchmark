import json
import sqlite3
import sys

sys.path.insert(0, ".opencode/tools")

import write_session_stats as stats


def test_run_duration_uses_state_start_and_end():
    payload = {
        "started_at": "2026-04-26T10:00:00+00:00",
        "ended_at": "2026-04-26T12:03:04+00:00",
        "phases": {},
    }

    duration = stats.run_duration_from_state(payload)

    assert duration["seconds"] == 7384
    assert duration["hh_mm_ss"] == "02:03:04"


def test_token_usage_from_assistant_messages_sums_input_output_and_total():
    messages = [
        {"role": "user", "tokens": {"total": 999}},
        {
            "role": "assistant",
            "tokens": {
                "input": 10,
                "output": 3,
                "reasoning": 2,
                "cache": {"read": 5, "write": 7},
                "total": 27,
            },
        },
        {
            "role": "assistant",
            "tokens": {
                "input": 20,
                "output": 4,
                "reasoning": 1,
                "cache": {"read": 6, "write": 0},
                "total": 31,
            },
        },
    ]

    usage = stats.token_usage_from_messages(messages)

    assert usage["assistant_messages_counted"] == 2
    assert usage["input_tokens"] == 30
    assert usage["output_tokens"] == 7
    assert usage["reasoning_tokens"] == 3
    assert usage["cache_read_tokens"] == 11
    assert usage["cache_write_tokens"] == 7
    assert usage["cache_tokens"] == 18
    assert usage["total_tokens"] == 58


def test_cli_falls_back_to_sqlite_and_writes_session_stats(tmp_path):
    repo_root = tmp_path / "repo"
    vault_dir = repo_root / "vault" / "paper"
    repo_root.mkdir()
    vault_dir.mkdir(parents=True)
    (vault_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "overall_status": "completed",
                "final_status": "evaluation_complete",
                "current_phase": "done",
                "started_at": "2026-04-26T10:00:00+00:00",
                "ended_at": "2026-04-26T10:00:05+00:00",
                "phases": {},
            }
        ),
        encoding="utf-8",
    )

    db_path = tmp_path / "opencode.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "create table session (id text primary key, directory text not null, title text not null, time_created integer not null, time_updated integer not null)"
        )
        conn.execute(
            "create table message (id text primary key, session_id text not null, time_created integer not null, time_updated integer not null, data text not null)"
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?)",
            ("ses_test", str(repo_root), "Test session", 1777197600000, 1777197605000),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_1",
                "ses_test",
                1777197601000,
                1777197602000,
                json.dumps(
                    {
                        "role": "assistant",
                        "tokens": {"input": 100, "output": 20, "reasoning": 5, "total": 125},
                    }
                ),
            ),
        )

    output_path = tmp_path / "session_stats.json"
    exit_code = stats.main(
        [
            "--repo-root",
            str(repo_root),
            "--vault-dir",
            str(vault_dir),
            "--output",
            str(output_path),
            "--opencode-db",
            str(db_path),
            "--skip-opencode-command",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selected_session_id"] == "ses_test"
    assert payload["token_usage"]["input_tokens"] == 100
    assert payload["token_usage"]["output_tokens"] == 20
    assert payload["token_usage"]["total_tokens"] == 125
    assert payload["run_duration"]["hh_mm_ss"] == "00:00:05"
