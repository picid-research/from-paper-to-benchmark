import { tool, type ToolContext } from "@opencode-ai/plugin"
import path from "path"

const z = tool.schema
const RESULT_MARKER = "PICID_TRAINING_RESULT_JSON="

async function invoke(command: string, args: Record<string, unknown>, context: ToolContext) {
  const repoRoot = String(args.repo_root || context.worktree || context.directory)
  const script = path.join(repoRoot, ".opencode", "tools", "PICID_training.py")
  const payload = JSON.stringify({ ...args, repo_root: repoRoot })
  const proc = Bun.spawn(["uv", "run", "python", script, command, "--input-json", payload], {
    cwd: repoRoot,
    stdout: "pipe",
    stderr: "inherit",
    env: {
      ...process.env,
      PROJECT_ROOT: repoRoot,
      TABPFN_DISABLE_TELEMETRY: "1",
    },
  })
  const stdout = await new Response(proc.stdout).text()
  const exitCode = await proc.exited
  const markerIndex = stdout.lastIndexOf(RESULT_MARKER)
  if (markerIndex < 0) {
    throw new Error(`PICID training tool did not emit ${RESULT_MARKER}\nSTDOUT:\n${stdout}`)
  }
  const raw = stdout.slice(markerIndex + RESULT_MARKER.length).trim()
  if (exitCode !== 0) {
    throw new Error(raw)
  }
  return JSON.stringify(JSON.parse(raw), null, 2)
}

export const run = tool({
  description: "Execute a monitored PICID training run with sanity/fixed-batch fit gates, then write stage-specific 07-training-log artifacts.",
  args: {
    repo_root: z.string().optional().describe("Repository root. Defaults to the OpenCode worktree."),
    vault_dir: z.string().describe("Vault directory for the validate-paper run."),
    experiment_config: z.string().describe("Hydra experiment config, e.g. pronostia/prognostics/raw/my_model"),
    model_paradigm: z.enum(["feedforward", "training", "fit_predict"]).default("feedforward"),
    expected_epochs: z.any().optional().describe("Expected epoch ceiling for full runs, or NOT_SPECIFIED to use the resolved trainer default."),
    expected_metrics: z
      .record(z.string(), z.number())
      .optional()
      .describe("Optional expected metric values for reporting."),
    gpu_available: z.boolean().default(false).describe("Whether GPU execution is available for this run."),
    run_mode: z.enum(["quick", "full"]).default("full"),
    training_stage: z
      .enum(["preflight_1epoch", "full"])
      .optional()
      .describe("preflight_1epoch exercises train/val/test/checkpoint before a full paper-scale run."),
    hp_profile: z.enum(["reference", "paper"]).default("reference").describe("Hyperparameter profile to run. All validation profiles force datamodule batch sizes to train=512, val=1024, test=1024. reference also preserves PICID scheduler cadence and uses paper/imputed model HPs; paper uses experiment config paper/default non-batch HPs."),
    hp_overrides: z
      .array(z.string())
      .optional()
      .describe("Optional keyed non-batch Hydra overrides for the selected HP profile. Validation runs always append fixed batch overrides after these values."),
    hp_reference_source: z
      .string()
      .optional()
      .describe("Run script/report path or note proving where hp_overrides came from."),
    dataset_candidate: z.string().optional().describe("Current dataset candidate name/config for reporting."),
    epoch_budget_rationale: z.string().describe("Required explanation for the chosen epoch budget."),
    monitor_interval_sec: z.number().optional().describe("Polling cadence for live full-run health checks."),
    plateau_patience_epochs: z.number().int().positive().optional().describe("Epoch window used for graceful plateau stops when early stopping is absent or not firing."),
  },
  async execute(args, context) {
    return invoke("run", args, context)
  },
})
