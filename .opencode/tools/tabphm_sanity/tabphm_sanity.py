"""CLI entrypoint for PICID sanity tools.

OpenCode custom tools call this script and pass a JSON payload through
``--input-json``. The script prints one marker-prefixed JSON result so callers can
ignore framework logging that may appear on stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PICID_sanity_core import (
    DATASET_EXECUTION_FAILED,
    DATASET_UNAVAILABLE,
    PRECHECK_TIMEOUT,
    TOOL_INVOCATION_FAILURE,
    CheckResult,
    append_report,
    check_gradient_flow,
    check_init_loss,
    check_overfit_batch,
    check_subset_convergence,
    check_zero_input,
    ensure_project_root,
    json_default,
    ladder_result,
    preflight_config,
    render_markdown,
    skip_gradient_check,
)


RESULT_MARKER = "PICID_SANITY_RESULT_JSON="


def payload_arg(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    value = payload.get(key, default)
    return default if value is None else value


def common(payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = ensure_project_root(payload_arg(payload, "repo_root", "."))
    return {
        "repo_root": repo_root,
        "experiment_config": payload_arg(payload, "experiment_config"),
        "vault_dir": payload.get("vault_dir"),
        "task_type": payload_arg(payload, "task_type", "regression"),
        "dataset_candidate": payload.get("dataset_candidate"),
        "extra_overrides": list(payload_arg(payload, "extra_overrides", [])),
        "timeout_sec": payload.get("timeout_sec"),
        "write_log": bool(payload_arg(payload, "write_log", False)),
        "zero_keys": payload.get("zero_keys"),
        "micro_batch_size": int(payload_arg(payload, "micro_batch_size", 4)),
        "overfit_max_steps": int(payload_arg(payload, "overfit_max_steps", 400)),
        "include_subset_convergence": bool(
            payload_arg(payload, "include_subset_convergence", False)
        ),
    }


def finish(result: dict[str, Any]) -> None:
    print(RESULT_MARKER + json.dumps(result, default=json_default, sort_keys=True))


def single_result(
    check: CheckResult,
    payload: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    result = ladder_result(
        checks=[check],
        experiment_config=values["experiment_config"],
        task_type=values["task_type"],
        dataset_candidate=values["dataset_candidate"],
        vault_dir=values["vault_dir"],
        write_log=values["write_log"],
        compact=True,
    )
    result["tool"] = payload.get("command")
    return result


def run_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    values = common(payload)
    experiment_config = values["experiment_config"]
    if not experiment_config:
        raise ValueError("experiment_config is required")

    task_type = values["task_type"]
    if command == "init_loss":
        check = check_init_loss(
            repo_root=values["repo_root"],
            experiment_config=experiment_config,
            task_type=task_type,
            num_classes=payload.get("num_classes"),
            expected_init_loss=payload.get("expected_init_loss", "auto"),
            extra_overrides=values["extra_overrides"],
            timeout_sec=values["timeout_sec"],
        )
        return single_result(check, payload, values)

    if command == "zero_input":
        if task_type == "fit_predict":
            check = CheckResult(
                name="zero_input",
                status="SKIPPED",
                diagnostic="fit-predict zero-input check is not implemented by the tool-side manual trainer",
            )
        else:
            check = check_zero_input(
                repo_root=values["repo_root"],
                experiment_config=experiment_config,
                extra_overrides=values["extra_overrides"],
                zero_keys=values["zero_keys"],
                timeout_sec=values["timeout_sec"],
            )
        return single_result(check, payload, values)

    if command == "overfit_batch":
        if task_type == "fit_predict":
            check = skip_gradient_check("overfit_batch")
        else:
            check = check_overfit_batch(
                repo_root=values["repo_root"],
                experiment_config=experiment_config,
                task_type=task_type,
                extra_overrides=values["extra_overrides"],
                timeout_sec=values["timeout_sec"],
                micro_batch_size=values["micro_batch_size"],
                max_steps=values["overfit_max_steps"],
            )
        return single_result(check, payload, values)

    if command == "gradient_flow":
        if task_type == "fit_predict":
            check = skip_gradient_check("gradient_flow")
        else:
            check = check_gradient_flow(
                repo_root=values["repo_root"],
                experiment_config=experiment_config,
                extra_overrides=values["extra_overrides"],
                zero_keys=values["zero_keys"],
                timeout_sec=values["timeout_sec"],
            )
        return single_result(check, payload, values)

    if command == "subset_convergence":
        if task_type == "fit_predict":
            check = skip_gradient_check("subset_convergence")
        else:
            check = check_subset_convergence(
                repo_root=values["repo_root"],
                experiment_config=experiment_config,
                extra_overrides=values["extra_overrides"],
                timeout_sec=values["timeout_sec"],
            )
        return single_result(check, payload, values)

    if command == "ladder":
        checks: list[CheckResult] = []
        preflight = preflight_config(
            values["repo_root"], experiment_config, values["extra_overrides"]
        )
        checks.append(preflight)
        if preflight.status != "PASS":
            return ladder_result(
                checks=checks,
                experiment_config=experiment_config,
                task_type=task_type,
                dataset_candidate=values["dataset_candidate"],
                vault_dir=values["vault_dir"],
                write_log=bool(payload_arg(payload, "write_log", True)),
                compact=True,
            )
        if task_type == "fit_predict":
            checks.extend(
                [
                    skip_gradient_check("init_loss"),
                    skip_gradient_check("gradient_flow"),
                    skip_gradient_check("overfit_batch"),
                ]
            )
            if values["include_subset_convergence"]:
                checks.append(skip_gradient_check("subset_convergence"))
        else:
            ladder_steps = [
                lambda: check_init_loss(
                    repo_root=values["repo_root"],
                    experiment_config=experiment_config,
                    task_type=task_type,
                    num_classes=payload.get("num_classes"),
                    expected_init_loss=payload.get("expected_init_loss", "auto"),
                    extra_overrides=values["extra_overrides"],
                    timeout_sec=values["timeout_sec"],
                ),
                lambda: check_gradient_flow(
                    repo_root=values["repo_root"],
                    experiment_config=experiment_config,
                    extra_overrides=values["extra_overrides"],
                    zero_keys=values["zero_keys"],
                    timeout_sec=values["timeout_sec"],
                ),
                lambda: check_overfit_batch(
                    repo_root=values["repo_root"],
                    experiment_config=experiment_config,
                    task_type=task_type,
                    extra_overrides=values["extra_overrides"],
                    timeout_sec=values["timeout_sec"],
                    micro_batch_size=values["micro_batch_size"],
                    max_steps=values["overfit_max_steps"],
                ),
            ]
            if values["include_subset_convergence"]:
                ladder_steps.append(
                    lambda: check_subset_convergence(
                        repo_root=values["repo_root"],
                        experiment_config=experiment_config,
                        extra_overrides=values["extra_overrides"],
                        timeout_sec=values["timeout_sec"],
                    )
                )
            for check_fn in ladder_steps:
                check = check_fn()
                checks.append(check)
                if check.status in (
                    DATASET_UNAVAILABLE,
                    DATASET_EXECUTION_FAILED,
                    TOOL_INVOCATION_FAILURE,
                    PRECHECK_TIMEOUT,
                ):
                    break
        return ladder_result(
            checks=checks,
            experiment_config=experiment_config,
            task_type=task_type,
            dataset_candidate=values["dataset_candidate"],
            vault_dir=values["vault_dir"],
            write_log=bool(payload_arg(payload, "write_log", True)),
            compact=True,
        )

    if command == "report":
        checks = [CheckResult(**item) for item in payload_arg(payload, "checks", [])]
        result = ladder_result(
            checks=checks,
            experiment_config=experiment_config,
            task_type=task_type,
            dataset_candidate=values["dataset_candidate"],
            vault_dir=values["vault_dir"],
            write_log=False,
        )
        markdown = render_markdown(
            result,
            values["vault_dir"],
            experiment_config,
            task_type,
            values["dataset_candidate"],
        )
        result["markdown_appended_path"] = append_report(markdown, values["vault_dir"])
        return result

    raise ValueError(f"Unknown command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PICID sanity checks.")
    parser.add_argument(
        "command",
        choices=[
            "ladder",
            "init_loss",
            "zero_input",
            "overfit_batch",
            "gradient_flow",
            "subset_convergence",
            "report",
        ],
    )
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    payload = json.loads(args.input_json)
    payload["command"] = args.command
    result = run_command(args.command, payload)
    finish(result)


if __name__ == "__main__":
    main()
