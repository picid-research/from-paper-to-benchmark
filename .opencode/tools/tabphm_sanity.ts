import { tool, type ToolContext } from "@opencode-ai/plugin"
import path from "path"

const z = tool.schema
const RESULT_MARKER = "PICID_SANITY_RESULT_JSON="

const commonArgs = {
  experiment_config: z.string().describe("Hydra experiment config, e.g. pronostia/prognostics/raw/my_model"),
  vault_dir: z.string().optional().describe("Vault directory containing 04-implementation-blueprint.md"),
  repo_root: z.string().optional().describe("Repository root. Defaults to the OpenCode worktree."),
  dataset_candidate: z.string().optional().describe("Current dataset candidate name for fallback reporting"),
  task_type: z.enum(["regression", "classification", "fit_predict"]).default("regression"),
  num_classes: z.number().optional().describe("Number of classes for classification init-loss checks"),
  expected_init_loss: z.union([z.number(), z.literal("auto")]).default("auto"),
  extra_overrides: z.array(z.string()).optional().describe("Additional Hydra overrides appended to sanity commands"),
  timeout_sec: z.number().optional().describe("Per-check wall-clock timeout in seconds (default 600s for in-process checks)"),
  write_log: z.boolean().optional().describe("Append Markdown to vault_dir/06-sanity-ladder-log.md"),
  zero_keys: z.array(z.string()).optional().describe("Batch keys to zero for input-independence and gradient checks"),
  micro_batch_size: z.number().optional().describe("Sample count for the micro-batch memorization probe used by overfit_batch (default 4)"),
  overfit_max_steps: z.number().optional().describe("Maximum optimizer steps for the micro-batch memorization probe (default 400)"),
  include_subset_convergence: z.boolean().optional().describe("Opt in to the expensive 30-epoch subset_convergence check (off by default)"),
}

async function invoke(command: string, args: Record<string, unknown>, context: ToolContext) {
  const repoRoot = String(args.repo_root || context.worktree || context.directory)
  const script = path.join(repoRoot, ".opencode", "tools", "PICID_sanity", "PICID_sanity.py")
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
  if (exitCode !== 0) {
    throw new Error(`PICID sanity tool failed with exit code ${exitCode}\nSTDOUT:\n${stdout}`)
  }
  const markerIndex = stdout.lastIndexOf(RESULT_MARKER)
  if (markerIndex < 0) {
    throw new Error(`PICID sanity tool did not emit ${RESULT_MARKER}\nSTDOUT:\n${stdout}`)
  }
  const raw = stdout.slice(markerIndex + RESULT_MARKER.length).trim()
  const parsed = JSON.parse(raw)
  return JSON.stringify(parsed, null, 2)
}

export const ladder = tool({
  description: "Run the trimmed 3-check PICID sanity ladder (init_loss → gradient_flow → overfit_batch) in a single in-process framework build, with [heartbeat] progress on stderr and a 600s default per-check timeout. The overfit_batch step is a micro-batch memorization probe. Set include_subset_convergence:true only when explicitly needed.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("ladder", { ...args, write_log: args.write_log ?? true }, context)
  },
})

export const init_loss = tool({
  description: "Run PICID sanity Check 1: loss at initialization.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("init_loss", args, context)
  },
})

export const zero_input = tool({
  description: "Run PICID sanity Check 2: real inputs versus zeroed inputs.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("zero_input", args, context)
  },
})

export const overfit_batch = tool({
  description: "Run PICID sanity Check 3: memorize a tiny sliced micro-batch and classify the result as PASS / INVESTIGATE / FAIL.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("overfit_batch", args, context)
  },
})

export const gradient_flow = tool({
  description: "Run PICID sanity Check 4: gradient-flow and batch-isolation audit.",
  args: commonArgs,
  async execute(args, context) {
    return invoke("gradient_flow", args, context)
  },
})

export const subset_convergence = tool({
  description: "Run PICID sanity small-subset convergence check (opt-in only; expensive 30-epoch CLI run, not part of the default ladder).",
  args: commonArgs,
  async execute(args, context) {
    return invoke("subset_convergence", args, context)
  },
})

export const report = tool({
  description: "Append a PICID sanity Markdown report from structured check results.",
  args: {
    ...commonArgs,
    checks: z.array(z.any()).describe("Check result objects returned by other PICID_sanity tools"),
  },
  async execute(args, context) {
    return invoke("report", args, context)
  },
})
