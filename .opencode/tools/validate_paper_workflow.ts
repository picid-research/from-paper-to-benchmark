import { tool, type ToolContext } from "@opencode-ai/plugin"
import path from "path"

const z = tool.schema
const RESULT_MARKER = "VALIDATE_PAPER_WORKFLOW_RESULT_JSON="

async function invoke(command: string, args: Record<string, unknown>, context: ToolContext) {
  const repoRoot = String(args.repo_root || context.worktree || context.directory)
  const script = path.join(repoRoot, ".opencode", "tools", "validate_paper_workflow.py")
  const payload = JSON.stringify({ ...args, repo_root: repoRoot })
  const proc = Bun.spawn(["uv", "run", "python", script, command, "--input-json", payload], {
    cwd: repoRoot,
    stdout: "pipe",
    stderr: "inherit",
    env: {
      ...process.env,
      PROJECT_ROOT: repoRoot,
    },
  })
  const stdout = await new Response(proc.stdout).text()
  const exitCode = await proc.exited
  if (exitCode !== 0) {
    const markerIndex = stdout.lastIndexOf(RESULT_MARKER)
    if (markerIndex >= 0) {
      const raw = stdout.slice(markerIndex + RESULT_MARKER.length).trim()
      throw new Error(raw)
    }
    throw new Error(`validate_paper_workflow failed with exit code ${exitCode}\nSTDOUT:\n${stdout}`)
  }
  const markerIndex = stdout.lastIndexOf(RESULT_MARKER)
  if (markerIndex < 0) {
    throw new Error(`validate_paper_workflow did not emit ${RESULT_MARKER}\nSTDOUT:\n${stdout}`)
  }
  const raw = stdout.slice(markerIndex + RESULT_MARKER.length).trim()
  const parsed = JSON.parse(raw)
  return JSON.stringify(parsed, null, 2)
}

const commonArgs = {
  repo_root: z.string().optional().describe("Repository root. Defaults to the OpenCode worktree."),
  vault_dir: z.string().describe("Vault directory for the validate-paper run."),
}

const hparamRow = z.object({
  value: z.any(),
  source_location: z.any(),
  category: z.any(),
  framework_default_available: z.any(),
})

const trainingHyperparameters = z.object({
  optimizer: hparamRow,
  learning_rate: hparamRow,
  lr_schedule: hparamRow,
  weight_decay: hparamRow,
  grad_clip: hparamRow,
  warmup: hparamRow,
  max_epochs: hparamRow,
  batch_size: hparamRow,
  training_protocol_notes: hparamRow,
}).catchall(hparamRow)

export const init_run = tool({
  description: "Create or load the canonical run_state.json for a validate-paper workflow.",
  args: {
    ...commonArgs,
    paper_dir: z.string().describe("Paper directory containing the PDF or processed_paper output."),
    run_mode: z.enum(["blueprint-only", "quick", "full"]).default("full"),
    hp_mode: z
      .enum(["reference_only", "reference_plus_paper"])
      .default("reference_only")
      .describe("Hyperparameter comparison mode. All validation profiles force datamodule batch sizes to train=512, val=1024, test=1024. reference_only preserves PICID scheduler cadence and uses paper/imputed model HPs; reference_plus_paper also runs the paper/default non-batch profile as auxiliary context."),
  },
  async execute(args, context) {
    return invoke("init_run", args, context)
  },
})

export const get_status = tool({
  description: "Read run_state.json and return the current workflow status and next phase.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("get_status", args, context)
  },
})

export const start_phase = tool({
  description: "Mark a workflow phase as running and increment its attempt counter. Enforces configured attempt ceilings for phases that have bounded retry budgets.",
  args: {
    ...commonArgs,
    phase: z.string().describe("Workflow phase name."),
    max_attempts: z.number().int().positive().optional().describe("Optional explicit total-attempt ceiling for this phase. When omitted, the control plane uses its built-in defaults."),
  },
  async execute(args, context) {
    return invoke("start_phase", args, context)
  },
})

export const complete_phase = tool({
  description: "Mark a workflow phase as passed after validating its required artifacts.",
  args: {
    ...commonArgs,
    phase: z.string().describe("Workflow phase name."),
    artifact_paths: z.array(z.string()).optional().describe("Optional explicit artifact paths for this phase."),
    metadata: z
      .record(z.string(), z.any())
      .optional()
      .describe("Optional phase metadata to persist in run_state.json."),
    dataset_transition: z
      .record(z.string(), z.any())
      .optional()
      .describe("Optional same-dataset recovery or selection event. Cross-dataset fallback is disabled."),
  },
  async execute(args, context) {
    return invoke("complete_phase", args, context)
  },
})

export const fail_phase = tool({
  description: "Mark a workflow phase as failed. Set retryable=true when the workflow should keep working from that phase instead of globally blocking. If the phase has exhausted its configured attempt ceiling, the control plane converts the failure into a blocking one.",
  args: {
    ...commonArgs,
    phase: z.string().describe("Workflow phase name."),
    reason_code: z.string().optional().describe("Stable machine-readable failure code."),
    message: z.string().optional().describe("Human-readable failure detail."),
    retryable: z.boolean().optional().describe("Whether this failure should leave the workflow active at the same phase for autonomous repair/retry."),
    max_attempts: z.number().int().positive().optional().describe("Optional explicit total-attempt ceiling for this phase. When omitted, the control plane uses its built-in defaults."),
    metadata: z
      .record(z.string(), z.any())
      .optional()
      .describe("Optional phase metadata to persist in run_state.json."),
  },
  async execute(args, context) {
    return invoke("fail_phase", args, context)
  },
})

export const sync_from_artifacts = tool({
  description: "Rebuild or refresh run_state.json by inspecting the vault for known workflow artifacts.",
  args: {
    ...commonArgs,
    paper_dir: z.string().describe("Paper directory containing the PDF or processed_paper output."),
    run_mode: z.enum(["blueprint-only", "quick", "full"]).default("full"),
    hp_mode: z.enum(["reference_only", "reference_plus_paper"]).default("reference_only"),
  },
  async execute(args, context) {
    return invoke("sync_from_artifacts", args, context)
  },
})

export const write_blueprint_sidecar = tool({
  description: "Write canonical 04-implementation-blueprint.json and render 04-implementation-blueprint.md from the same validated payload.",
  args: {
    ...commonArgs,
    schema_version: z.any(),
    paper_dataset: z.any(),
    evaluation_targets: z.any(),
    excluded_paper_datasets: z.any(),
    framework_dataset_used: z.any(),
    datasets_are_same: z.any(),
    comparison_mode: z.any(),
    fallback_allowed: z.any(),
    dataset_fallback_candidates: z.any(),
    experiment_config: z.any(),
    task_type: z.any(),
    required_new_files: z.any(),
    verification_protocol: z.any(),
    paper_hyperparameters: trainingHyperparameters,
    dataset_contract: z.any(),
    model_io_contract: z.any(),
    split_contract: z.any(),
    validation_run_matrix: z.any().optional(),
    markdown_sections: z.record(z.string(), z.any()).optional().describe("Optional rendered markdown sections. Stored only in the markdown artifact, not in the JSON sidecar."),
  },
  async execute(args, context) {
    return invoke("write_blueprint_sidecar", args, context)
  },
})

export const write_conceptual_sidecar = tool({
  description: "Write canonical 02-conceptual-analysis.json and render 02-conceptual-analysis.md from the same validated payload.",
  args: {
    ...commonArgs,
    schema_version: z.any(),
    paper_summary: z.any(),
    structure_map: z.any(),
    novel_components: z.any(),
    reused_components: z.any(),
    configured_components: z.any(),
    dataset_mapping: z.any(),
    direct_evaluation_targets: z.any(),
    excluded_paper_datasets: z.any(),
    integration_roadmap: z.any(),
    markdown_sections: z.record(z.string(), z.any()).optional().describe("Optional rendered markdown sections. Stored only in the markdown artifact, not in the JSON sidecar."),
  },
  async execute(args, context) {
    return invoke("write_conceptual_sidecar", args, context)
  },
})

export const write_algorithmic_sidecar = tool({
  description: "Write canonical 03-algorithmic-spec.json, validate the nine required HP rows, and render 03-algorithmic-spec.md.",
  args: {
    ...commonArgs,
    schema_version: z.any(),
    algorithms: z.any(),
    equations: z.any(),
    architectures: z.any(),
    losses: z.any(),
    training_hyperparameters: trainingHyperparameters,
    data_processing: z.any(),
    reference_implementations: z.any(),
    markdown_sections: z.record(z.string(), z.any()).optional().describe("Optional rendered markdown sections. Stored only in the markdown artifact, not in the JSON sidecar."),
  },
  async execute(args, context) {
    return invoke("write_algorithmic_sidecar", args, context)
  },
})

export const append_sanity_result = tool({
  description: "Append one structured sanity attempt to 06-sanity-ladder-log.jsonl.",
  args: {
    ...commonArgs,
    verdict: z.any(),
    attempt: z.any(),
    passed_checks: z.any(),
    failed_checks: z.any(),
    skipped_checks: z.any(),
    check_diagnostics: z.any(),
    dataset_candidate: z.any(),
    dataset_status: z.any(),
    fallback_trigger: z.any(),
    experiment_config: z.any(),
    timestamp: z.any(),
  },
  async execute(args, context) {
    return invoke("append_sanity_result", args, context)
  },
})

export const append_training_result = tool({
  description: "Append one structured training attempt to 07-training-log.jsonl.",
  args: {
    ...commonArgs,
    status: z.any(),
    experiment_config: z.any(),
    dataset_candidate: z.any(),
    hp_profile: z.any().optional(),
    hp_profile_status: z.any().optional(),
    is_primary_comparison: z.any().optional(),
    hp_overrides: z.any().optional(),
    hp_reference_source: z.any().optional(),
    run_mode: z.any(),
    training_stage: z.any().optional(),
    max_epochs_ceiling: z.any(),
    epochs_run: z.any(),
    early_stopping_triggered: z.any(),
    early_stopping_epoch: z.any(),
    batch_size_paper: z.any(),
    batch_size_chosen: z.any(),
    batch_probe_source: z.any(),
    batch_fit_check_source: z.any().optional(),
    batch_size_configured: z.any().optional(),
    lr_paper: z.any(),
    lr_scaled: z.any(),
    lr_scaling_rule: z.any(),
    optimizer_family: z.any(),
    resolved_config_hash: z.any(),
    hardware_id: z.any(),
    epoch_budget_rationale: z.any(),
    monitor_status: z.any().optional(),
    stop_reason: z.any().optional(),
    monitor_events: z.any().optional(),
    monitor_probe_summary: z.any().optional(),
    checkpoint_created: z.any().optional(),
    test_stage_completed: z.any().optional(),
    output_dir: z.any().optional(),
    metrics_csv: z.any().optional(),
    traceback_excerpt: z.any(),
    timestamp: z.any(),
  },
  async execute(args, context) {
    return invoke("append_training_result", args, context)
  },
})

export const write_evaluation_sidecar = tool({
  description: "Write 08-evaluation-report.json after validating the required machine-readable fields.",
  args: {
    ...commonArgs,
    status: z.any(),
    artifact_status: z.any().optional(),
    technical_status: z.any().optional(),
    scientific_status: z.any().optional(),
    blocking: z.any().optional(),
    observations: z.any().optional(),
    comparison_mode: z.any(),
    dataset_used: z.any(),
    paper_dataset: z.any(),
    hypothesis_status: z.any(),
    paper_claims_summary: z.any(),
    timestamp: z.any(),
  },
  async execute(args, context) {
    return invoke("write_evaluation_sidecar", args, context)
  },
})
