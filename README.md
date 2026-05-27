# From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence

OpenCode workflow bundle for validating research papers against the PICID Prognostics and Health Management benchmark framework.

The agent implementation is self contained under the `.opencode/` directory. It
contains custom OpenCode agents, skills, tools, reference contracts, and the
`/validate-paper` command. The workflow reads a paper, extracts its conceptual
and algorithmic claims, builds an implementation blueprint, implements the
required PICID extensions, runs static and sanity checks, trains/evaluates the
result, and writes structured validation artifacts into a vault directory.

## What It Contains

- `.opencode/commands/validate-paper.md` - the top-level OpenCode command that
  starts the validation workflow.
- `.opencode/agents/` - orchestration and specialist agents for paper analysis,
  blueprinting, implementation, training, and evaluation.
- `.opencode/skills/` - task-level workflows for paper preprocessing,
  implementation, static verification, sanity checks, training, and result
  diagnosis.
- `.opencode/tools/` - deterministic helper tools used by the agents for state
  management, paper indexing, sanity checks, baseline parsing, training, plots,
  and session accounting.
- `.opencode/baseline_prompts/` - prompt templates for running external
  baseline-agent implementations against the same paper, either as a standalone
  repository build or inside the PICID framework.
- `.opencode/reference/` - PICID/PICID contracts, inventory, configuration
  patterns, and validation policies.

## Requirements

You need:

- A working PICID checkout. Run the validation command from that checkout
  root, because the workflow treats the current working directory as the PICID
  repository.
- Python dependencies for PICID, installed in the way that PICID normally
  expects.
- `uv`, because the workflow runs helper commands with `uv run`.
- `marker_single`, available through the active `uv` environment, if the input
  paper is a PDF and has not already been converted to markdown.
- OpenCode CLI with an authenticated model provider.

## Install OpenCode

OpenCode's official install script is:

```bash
curl -fsSL https://opencode.ai/install | bash
```

You can also install it with Node.js:

```bash
npm install -g opencode-ai
```

Then configure an LLM provider:

```bash
opencode auth login
```

Verify the CLI is available:

```bash
opencode --version
```

OpenCode loads project commands from `.opencode/commands/`, so this bundle's
`.opencode/commands/validate-paper.md` becomes the `/validate-paper` command
when OpenCode is started in the project root.

## Install This Workflow In PICID

From this repository, copy the `.opencode` directory into the root of the
PICID/PICID checkout:

```bash
rsync -a .opencode/ /path/to/PICID/.opencode/
```

The target PICID root should then contain:

```text
/path/to/PICID/
  picid/
  configs/
  .opencode/
    commands/validate-paper.md
    agents/
    baseline_prompts/
    skills/
    tools/
    reference/
```

## Prepare A Paper

Create a vault directory in the PICID checkout and place exactly one paper PDF
there:

```bash
cd /path/to/PICID
mkdir -p vault/paper
cp /path/to/paper.pdf vault/paper/
```

If the paper has already been converted by Marker, the workflow can also consume
existing processed output under:

```text
vault/paper/processed_paper/<pdf_stem>/
```

## Run The Validation Workflow

Start OpenCode from the PICID root:

```bash
cd /path/to/PICID
opencode
```

In the OpenCode TUI, invoke:

```text
/validate-paper vault/paper vault/paper quick reference_only
```

For a full unattended validation run:

```text
/validate-paper vault/paper vault/paper full reference_only
```

The command arguments are:

1. `paper_dir` - directory containing the PDF or processed markdown. Default:
   `vault/paper`.
2. `vault_dir` - output directory for workflow state and artifacts. Default:
   same as `paper_dir`.
3. `run_mode` - `blueprint-only`, `quick`, or `full`. Default: `full`.
4. `hp_mode` - `reference_only` or `reference_plus_paper`. Default:
   `reference_only`.

## Run Baseline Agent Prompts

The files in `.opencode/baseline_prompts/` are not OpenCode commands. They are copy/paste prompt templates for separate baseline-agent runs, used when you want an independent comparison against the main `/validate-paper` workflow.

Use `.opencode/baseline_prompts/baseline-agent-with-framework.md` when the baseline agent should work inside an existing PICID checkout. Start the agent from the PICID root, replace `Paper: path_to_paper.pdf` with the actual paper path, and keep the instruction to run experiments with `paths=agent` so datasets and caches resolve through the shared agent paths profile.

Use `.opencode/baseline_prompts/baseline-agent-no-framework.md` when the baseline agent should implement the paper from scratch in an empty repository.
Set up that repository with the paper at `paper/paper.pdf`, make the dataset root available at `/workspace/datasets`, then paste the prompt as the initial agent instruction.

## Run Modes

- `blueprint-only` - parse the paper and produce the analysis/specification and
  implementation blueprint without running training.
- `quick` - run a one-epoch smoke validation path for each direct supported
  PICID paper dataset.
- `full` - run the repair-looped preflight and monitored paper-scale validation
  runs without wall-clock timeout interruption.

## Outputs

The workflow writes state and validation artifacts into `vault_dir`, including:

- `run_state.json`
- `02-conceptual-analysis.json`
- `03-algorithmic-spec.json`
- `04-implementation-blueprint.json`
- `05-paper-hypothesis.md`
- `06-sanity-ladder-log.jsonl`
- `batch_fit_check.json`
- `07-training-log.jsonl`
- `08-evaluation-report.json`
- `session_stats.json`

The default success condition is benchmark validation evidence, not perfect
paper-claim reproduction. Scientific caveats such as metric-scale mismatch,
missing overlapping baselines, disputed claims, or benchmark-only comparison are
recorded in the vault artifacts instead of being hidden.

## Notes

- The workflow is unattended by default. It proceeds autonomously through
  recoverable failures and stops only on named blockers after bounded retries.
- Cross-dataset fallback is disabled. A selected validation dataset must be a
  direct PICID-supported dataset used by the paper, or the workflow reports a
  named dataset blocker.
- `.opencode/` is workflow infrastructure. During paper validation, agents should
  not edit `.opencode/`; per-paper outputs belong in the vault and generated
  PICID implementation/config files.

## References

- OpenCode install docs: https://opencode.ai/docs/
- OpenCode custom command docs: https://opencode.ai/docs/commands/
