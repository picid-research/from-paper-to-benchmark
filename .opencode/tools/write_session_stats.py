"""Write OpenCode session token and validate-paper runtime stats.

The preferred source for session metadata is:

    opencode session list --format json
    opencode export <session_id>

If those commands fail, the script falls back to OpenCode's local SQLite
database and sums token counters from assistant messages for the selected
session.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SESSION_STATS_FILE_NAME = "session_stats.json"


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def parse_iso_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw /= 1000.0
        return datetime.fromtimestamp(raw, UTC)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def datetime_to_epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1000)


def format_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_command(args: list[str], cwd: Path, timeout: int = 60) -> CommandResult:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(args=args, returncode=127, stdout="", stderr=str(exc))
    return CommandResult(
        args=args,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def iter_session_objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("sessions", "data", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(key in payload for key in ("id", "session_id", "sessionID")):
        return [payload]
    return []


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def session_id_of(session: dict[str, Any]) -> str | None:
    for key in ("id", "session_id", "sessionID", "sessionId"):
        value = session.get(key)
        if value:
            return str(value)
    return None


def session_directory_of(session: dict[str, Any]) -> str | None:
    for keys in (
        ("directory",),
        ("cwd",),
        ("path", "cwd"),
        ("project", "directory"),
        ("project", "path"),
        ("workspace", "directory"),
    ):
        value = get_nested(session, *keys)
        if value:
            return str(value)
    return None


def session_time_ms(session: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = session.get(name)
        if value is not None:
            parsed = parse_iso_datetime(value)
            if parsed:
                return datetime_to_epoch_ms(parsed)
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    time_obj = session.get("time")
    if isinstance(time_obj, dict):
        for name in names:
            short = name.replace("time_", "")
            value = time_obj.get(short)
            if value is not None:
                parsed = parse_iso_datetime(value)
                if parsed:
                    return datetime_to_epoch_ms(parsed)
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def choose_session(
    sessions: list[dict[str, Any]],
    *,
    requested_session_id: str | None,
    repo_root: Path,
    started_ms: int | None,
    ended_ms: int | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not sessions:
        return None, ["No OpenCode sessions were found."]

    env_session_id = (
        requested_session_id
        or os.environ.get("OPENCODE_SESSION_ID")
        or os.environ.get("OPENCODE_SESSION")
        or os.environ.get("SESSION_ID")
    )
    if env_session_id:
        matches = [session for session in sessions if session_id_of(session) == env_session_id]
        if matches:
            return matches[0], warnings
        warnings.append(f"Requested session id was not found: {env_session_id}")

    repo_str = str(repo_root.resolve())

    def score(session: dict[str, Any]) -> tuple[int, int]:
        score_value = 0
        directory = session_directory_of(session)
        created = session_time_ms(session, ("time_created", "created", "created_at"))
        updated = session_time_ms(session, ("time_updated", "updated", "updated_at"))
        if directory:
            try:
                resolved_directory = str(Path(directory).resolve())
            except OSError:
                resolved_directory = directory
            if resolved_directory == repo_str:
                score_value += 40
            elif resolved_directory.startswith(f"{repo_str}{os.sep}"):
                score_value += 20
        if started_ms is not None:
            start_window = 30 * 60 * 1000
            if created is not None and abs(created - started_ms) <= start_window:
                score_value += 35
            if created is not None and updated is not None and created <= started_ms <= updated + start_window:
                score_value += 30
        if ended_ms is not None and updated is not None:
            end_window = 30 * 60 * 1000
            if abs(updated - ended_ms) <= end_window:
                score_value += 25
        return score_value, updated or created or 0

    ranked = sorted(sessions, key=score, reverse=True)
    best = ranked[0]
    if len(ranked) > 1 and score(ranked[0])[0] == score(ranked[1])[0]:
        warnings.append("Multiple sessions matched equally; selected the most recently updated one.")
    return best, warnings


def normalize_token_usage(tokens: dict[str, Any] | None) -> dict[str, int | None]:
    tokens = tokens or {}

    def number(*keys: str) -> int | None:
        for key in keys:
            value = tokens.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        return None

    cache = tokens.get("cache")
    cache_read = None
    cache_write = None
    cache_total = None
    if isinstance(cache, dict):
        cache_read = number_from(cache.get("read"))
        cache_write = number_from(cache.get("write"))
        if cache_read is not None or cache_write is not None:
            cache_total = (cache_read or 0) + (cache_write or 0)
    elif isinstance(cache, (int, float)) and not isinstance(cache, bool):
        cache_total = int(cache)

    input_tokens = number("input", "input_tokens", "prompt", "prompt_tokens")
    output_tokens = number("output", "output_tokens", "completion", "completion_tokens")
    reasoning_tokens = number("reasoning", "reasoning_tokens")
    total_tokens = number("total", "total_tokens")
    if total_tokens is None:
        parts = [input_tokens, output_tokens, reasoning_tokens, cache_total]
        if any(part is not None for part in parts):
            total_tokens = sum(part or 0 for part in parts)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "cache_tokens": cache_total,
        "total_tokens": total_tokens,
    }


def number_from(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def add_token_dict(total: dict[str, int], tokens: dict[str, Any]) -> None:
    normalized = normalize_token_usage(tokens)
    for key, value in normalized.items():
        if value is not None:
            total[key] = total.get(key, 0) + int(value)


def token_usage_from_messages(messages: list[dict[str, Any]]) -> dict[str, int | None]:
    total: dict[str, int] = {}
    counted = 0
    for message in messages:
        if str(message.get("role", "")).lower() != "assistant":
            continue
        tokens = message.get("tokens")
        if isinstance(tokens, dict):
            add_token_dict(total, tokens)
            counted += 1
    payload = {key: total.get(key) for key in normalize_token_usage({})}
    payload["assistant_messages_counted"] = counted
    return payload


def token_usage_from_export(export_payload: Any) -> tuple[dict[str, int | None] | None, str | None]:
    if not isinstance(export_payload, dict):
        return None, None
    session_tokens = export_payload.get("tokens")
    if isinstance(session_tokens, dict):
        return normalize_token_usage(session_tokens), "opencode_export.session.tokens"

    messages = export_payload.get("messages")
    if isinstance(messages, list):
        message_dicts = [message for message in messages if isinstance(message, dict)]
        usage = token_usage_from_messages(message_dicts)
        if usage.get("assistant_messages_counted"):
            return usage, "opencode_export.messages"

    parts = export_payload.get("parts")
    if isinstance(parts, list):
        total: dict[str, int] = {}
        counted = 0
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("tokens"), dict):
                add_token_dict(total, part["tokens"])
                counted += 1
        if counted:
            payload = {key: total.get(key) for key in normalize_token_usage({})}
            payload["parts_counted"] = counted
            return payload, "opencode_export.parts"
    return None, None


def default_opencode_db_path() -> Path:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "opencode" / "opencode.db"
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def read_sessions_from_db(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.is_file():
        return []
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=20) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select id, directory, title, time_created, time_updated
            from session
            order by time_updated desc
            """
        ).fetchall()
    return [dict(row) for row in rows]


def read_messages_from_db(db_path: Path, session_id: str) -> list[dict[str, Any]]:
    if not db_path.is_file():
        return []
    uri = f"file:{db_path}?mode=ro"
    messages: list[dict[str, Any]] = []
    with sqlite3.connect(uri, uri=True, timeout=20) as conn:
        rows = conn.execute(
            """
            select data
            from message
            where session_id = ?
            order by time_created, id
            """,
            (session_id,),
        ).fetchall()
    for (raw_data,) in rows:
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            messages.append(data)
    return messages


def run_duration_from_state(state: dict[str, Any]) -> dict[str, Any]:
    phases = state.get("phases") if isinstance(state.get("phases"), dict) else {}
    started_at = state.get("started_at")
    if not started_at:
        started_candidates = [
            phase.get("started_at")
            for phase in phases.values()
            if isinstance(phase, dict) and phase.get("started_at")
        ]
        started_at = min(started_candidates) if started_candidates else None

    ended_at = state.get("ended_at")
    if not ended_at:
        completed_candidates = [
            phase.get("completed_at")
            for phase in phases.values()
            if isinstance(phase, dict) and phase.get("completed_at")
        ]
        ended_at = max(completed_candidates) if completed_candidates else None

    start_dt = parse_iso_datetime(started_at)
    end_dt = parse_iso_datetime(ended_at)
    seconds = int((end_dt - start_dt).total_seconds()) if start_dt and end_dt else None
    return {
        "started_at": start_dt.isoformat() if start_dt else started_at,
        "ended_at": end_dt.isoformat() if end_dt else ended_at,
        "seconds": seconds,
        "hh_mm_ss": format_duration(seconds),
        "source": "run_state.started_at_to_ended_at",
    }


def load_session_list(
    *,
    opencode_bin: str,
    repo_root: Path,
    max_count: int,
) -> tuple[list[dict[str, Any]], Any | None, CommandResult | None]:
    result = run_command(
        [opencode_bin, "session", "list", "--format", "json", "--max-count", str(max_count)],
        cwd=repo_root,
    )
    if result.returncode != 0:
        return [], None, result
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [], None, result
    return iter_session_objects(payload), payload, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="PICID repository root.")
    parser.add_argument("--vault-dir", default="vault/paper", help="Vault directory containing run_state.json.")
    parser.add_argument("--output", default=None, help="Stats JSON path. Defaults to <vault_dir>/session_stats.json.")
    parser.add_argument("--mirror-output", action="append", default=[], help="Additional path(s) to write the same stats JSON.")
    parser.add_argument("--session-id", default=None, help="Explicit OpenCode session id.")
    parser.add_argument("--opencode-bin", default="opencode", help="OpenCode executable.")
    parser.add_argument("--opencode-db", default=None, help="OpenCode sqlite database path.")
    parser.add_argument("--max-count", type=int, default=100, help="Max sessions to request from opencode session list.")
    parser.add_argument("--skip-opencode-command", action="store_true", help="Use the SQLite fallback directly.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    vault_dir = Path(args.vault_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else vault_dir / SESSION_STATS_FILE_NAME
    db_path = Path(args.opencode_db).expanduser().resolve() if args.opencode_db else default_opencode_db_path()
    raw_dir = output_path.parent
    state_path = vault_dir / "run_state.json"

    warnings: list[str] = []
    errors: list[str] = []
    state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = load_json_file(state_path)
        except json.JSONDecodeError as exc:
            errors.append(f"Failed to parse run_state.json: {exc}")
    else:
        warnings.append(f"run_state.json not found: {state_path}")

    duration = run_duration_from_state(state)
    started_ms = datetime_to_epoch_ms(parse_iso_datetime(duration.get("started_at")))
    ended_ms = datetime_to_epoch_ms(parse_iso_datetime(duration.get("ended_at")))

    sessions: list[dict[str, Any]] = []
    session_list_payload: Any | None = None
    session_source = "unavailable"
    command_status: dict[str, Any] | None = None
    if not args.skip_opencode_command:
        sessions, session_list_payload, command_result = load_session_list(
            opencode_bin=args.opencode_bin,
            repo_root=repo_root,
            max_count=args.max_count,
        )
        if command_result is not None:
            command_status = {
                "args": command_result.args,
                "returncode": command_result.returncode,
                "stderr": command_result.stderr.strip(),
            }
            if command_result.returncode != 0:
                errors.append(
                    "opencode session list failed; falling back to SQLite database: "
                    f"{command_result.stderr.strip() or command_result.stdout.strip()}"
                )
        if session_list_payload is not None:
            session_source = "opencode_session_list"
            write_json_file(raw_dir / "opencode_session_list.json", session_list_payload)

    if not sessions:
        db_sessions = read_sessions_from_db(db_path)
        if db_sessions:
            sessions = db_sessions
            session_source = "opencode_sqlite"
        elif session_source == "unavailable":
            errors.append(f"No sessions available from OpenCode SQLite database: {db_path}")

    selected_session, selection_warnings = choose_session(
        sessions,
        requested_session_id=args.session_id,
        repo_root=repo_root,
        started_ms=started_ms,
        ended_ms=ended_ms,
    )
    warnings.extend(selection_warnings)

    token_usage: dict[str, Any] = {key: None for key in normalize_token_usage({})}
    token_usage_source = "unavailable"
    selected_session_id = session_id_of(selected_session) if selected_session else None
    export_payload: Any | None = None

    if selected_session_id and not args.skip_opencode_command:
        export_result = run_command(
            [args.opencode_bin, "export", selected_session_id],
            cwd=repo_root,
            timeout=120,
        )
        if export_result.returncode == 0:
            try:
                export_payload = json.loads(export_result.stdout)
            except json.JSONDecodeError as exc:
                errors.append(f"opencode export returned invalid JSON: {exc}")
            if export_payload is not None:
                write_json_file(raw_dir / "opencode_session_export.json", export_payload)
                usage, source = token_usage_from_export(export_payload)
                if usage is not None:
                    token_usage = usage
                    token_usage_source = source or "opencode_export"
        else:
            errors.append(
                "opencode export failed; falling back to SQLite messages: "
                f"{export_result.stderr.strip() or export_result.stdout.strip()}"
            )

    if selected_session_id and token_usage_source == "unavailable":
        messages = read_messages_from_db(db_path, selected_session_id)
        usage = token_usage_from_messages(messages)
        if usage.get("assistant_messages_counted"):
            token_usage = usage
            token_usage_source = "opencode_sqlite.message.tokens"
            write_json_file(raw_dir / "opencode_session_selected.json", selected_session)
        else:
            warnings.append(f"No assistant message token counters found for session {selected_session_id}.")

    stats = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "vault_dir": str(vault_dir),
        "run_state_path": str(state_path),
        "run_id": state.get("run_id"),
        "overall_status": state.get("overall_status"),
        "final_status": state.get("final_status"),
        "current_phase": state.get("current_phase"),
        "run_duration": duration,
        "session_source": session_source,
        "selected_session": selected_session,
        "selected_session_id": selected_session_id,
        "token_usage_source": token_usage_source,
        "token_usage": token_usage,
        "artifacts": {
            "session_stats": str(output_path),
            "opencode_session_list": str(raw_dir / "opencode_session_list.json")
            if session_list_payload is not None
            else None,
            "opencode_session_export": str(raw_dir / "opencode_session_export.json")
            if export_payload is not None
            else None,
            "opencode_session_selected": str(raw_dir / "opencode_session_selected.json")
            if selected_session is not None
            else None,
        },
        "opencode_command": command_status,
        "warnings": warnings,
        "errors": errors,
    }

    write_json_file(output_path, stats)
    for mirror in args.mirror_output:
        write_json_file(Path(mirror).resolve(), stats)
    print(json.dumps({"status": "WROTE", "path": str(output_path), "selected_session_id": selected_session_id}, sort_keys=True))
    return 0 if token_usage_source != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
