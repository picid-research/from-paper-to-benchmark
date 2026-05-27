import { tool, type ToolContext } from "@opencode-ai/plugin"
import path from "path"

const z = tool.schema
const RESULT_MARKER = "PICID_BASELINES_RESULT_JSON="

const taskTypes = [
  "prognostics",
  "diagnostics",
  "anomaly_detection",
  "forecasting",
  "state_forecasting",
] as const

const commonArgs = {
  repo_root: z.string().optional().describe("Repository root. Defaults to the OpenCode worktree."),
  report_output_root: z
    .string()
    .optional()
    .describe("Override for the report_output directory; defaults to <repo_root>/report_output."),
}

async function invoke(command: string, args: Record<string, unknown>, context: ToolContext) {
  const repoRoot = String(args.repo_root || context.worktree || context.directory)
  const script = path.join(repoRoot, ".opencode", "tools", "PICID_baselines", "PICID_baselines.py")
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
    throw new Error(`PICID baselines tool failed with exit code ${exitCode}\nSTDOUT:\n${stdout}`)
  }
  const markerIndex = stdout.lastIndexOf(RESULT_MARKER)
  if (markerIndex < 0) {
    throw new Error(`PICID baselines tool did not emit ${RESULT_MARKER}\nSTDOUT:\n${stdout}`)
  }
  const raw = stdout.slice(markerIndex + RESULT_MARKER.length).trim()
  const parsed = JSON.parse(raw)
  return JSON.stringify(parsed, null, 2)
}

export const load_baselines = tool({
  description:
    "Load PICID's frozen internal baselines for a given dataset/task from report_output/<folder>/results.nc. Returns per-(model, metric_key) mean/std/n and the list of available metric keys. Status OK | FOLDER_NOT_FOUND | METRIC_MISSING | CORRUPT_NETCDF — never throws for missing data, only for tool crashes.",
  args: {
    ...commonArgs,
    dataset: z
      .string()
      .describe("Dataset key, e.g. 'nb14', 'hsf15', 'concepts_n_cmapss'. Consulted against dataset_map.yaml first."),
    task_type: z.enum(taskTypes).describe("Task type used for folder disambiguation."),
    subtask: z
      .string()
      .optional()
      .describe("Optional subtask for datasets with multiple folders (e.g. hsf15 pump/cooler/valve/accumulator; concepts_n_cmapss ds02/default)."),
    metric_keys: z
      .array(z.string())
      .describe("Metric keys from the NetCDF metric_key dimension, e.g. ['test/rmse_denormalized','test/mae_denormalized']."),
    models_filter: z
      .string()
      .optional()
      .describe("Optional substring filter on the raw dotted model path; narrow the returned rows."),
  },
  async execute(args, context) {
    return invoke("load", args, context)
  },
})

export const resolve_folder = tool({
  description:
    "Resolve (dataset, task_type[, subtask]) to a report_output folder name without loading the NetCDF. Useful for quick routing checks.",
  args: {
    ...commonArgs,
    dataset: z.string(),
    task_type: z.enum(taskTypes),
    subtask: z.string().optional(),
  },
  async execute(args, context) {
    return invoke("resolve_folder", args, context)
  },
})

export const list_datasets = tool({
  description: "List report_output folder names that contain a results.nc file.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("list_datasets", args, context)
  },
})

export const short_name = tool({
  description:
    "Derive the stable short name the baseline tool uses for a given model class path (useful when mapping paper baselines to PICID baselines).",
  args: {
    ...commonArgs,
    model: z.string(),
  },
  async execute(args, context) {
    return invoke("short_name", args, context)
  },
})
