# SETUP.md — operator runbook (Codex CLI, local on the M5 Pro)

Everything runs on your Mac: Codex, MLX on Metal, the PyTorch reference on CPU, the benchmark. No cloud lanes, no runners, no RunPod. Your one-time setup is about fifteen minutes; after that your job is to reply `continue` when Codex ends a turn and to read two files once.

## 1. Repository

1. Create `~/robot/smolvla-mlx` (sibling of `~/robot/so101`, never inside it): `mkdir -p ~/robot/smolvla-mlx && cd ~/robot/smolvla-mlx && git init`.
2. Put `AGENTS.md`, `BRIEF.md`, and this `SETUP.md` at the root. Add the `.codex/config.toml` from Section 2. Commit.
3. Optional but recommended: create a private GitHub repo and add it as `origin`, so the agent can push hourly as backup. Run `gh auth login` once so pushes work without prompts. Make it public at v0.1.0.

## 2. Codex sandbox configuration

Create `.codex/config.toml` inside the repo with this content:

```toml
# Project-scoped Codex settings for smolvla-mlx.
# workspace-write is enforced by the OS: everything outside this repo is
# read-only, which protects ~/robot/so101 by construction, not by instruction.
sandbox_mode = "workspace-write"
approval_policy = "never"
model_reasoning_effort = "high"

[sandbox_workspace_write]
# Package installs and Hugging Face downloads need the network.
network_access = true
# Caches are routed inside the repo (see AGENTS.md), so no extra writable roots
# are required. Add your own here only if a tool insists on writing elsewhere.
writable_roots = []
```

Notes:
- Project-scoped config only applies to trusted projects. The first time you launch Codex in this directory it will ask you to trust it; accept.
- `approval_policy = "never"` means Codex never pauses to ask. Combined with `workspace-write`, a command that tries to write outside the repo simply fails, and the agent is instructed to handle that by routing the write inside the repo, not by escalating.
- Check `codex --help` and `/permissions` inside a session if your Codex version names these keys differently; the intent is: writes confined to the repo, network on, no approval prompts.

## 3. Launch

From `~/robot/smolvla-mlx`, run `codex`, accept the trust prompt, and paste the kickoff message from `BRIEF.md` Section 0. `AGENTS.md` is read automatically and holds the rules that must survive context resets.

When Codex ends a turn before Definition of Done, reply `continue`. To pick up a session later, use `codex resume`. If you want it to run unattended for hours, a bounded loop works:

```
for i in $(seq 1 20); do
  codex exec "Continue from the last PROGRESS.md entry, per AGENTS.md and BRIEF.md. Stop only at a stop condition or Definition of Done." || break
  grep -q "DEFINITION OF DONE MET" STATUS.md 2>/dev/null && break
done
```

Keep the Mac awake while it runs (`caffeinate -i` in a spare terminal, or disable sleep on power).

## 4. Your two touchpoints

1. **After Phase 1** (expect it within the first few hours): read `ARCHITECTURE.md` and `REUSE_DECISIONS.md`. Fifteen minutes. This is where a wrong turn is cheapest to catch. If something looks off, say so in the session; otherwise say nothing.
2. **`HUMAN_TASKS.md`** if it ever changes. Locally there should be almost nothing in it; the likely entries are "a tool needs a writable path outside the repo" (add it to `writable_roots`) or "GitHub push failed" (run `gh auth login`).

Stall detection: no commits for a few hours means the turn ended or the agent is stuck. Open the session, ask for `STATUS.md`, reply `continue`.

## 5. What the Mac does while this runs

- The PyTorch reference runs on CPU in fp32 to generate goldens. Expect a minute or two per regeneration; it happens rarely.
- MLX tests run on Metal. They are short; you can keep using the machine.
- The benchmark (`make bench`) wants a quiet machine for a few minutes. If you are mid-video-call when it runs, the p95 numbers will be noisy; the agent will rerun on request.

## 6. RunPod

Not needed. The reference model is 450M parameters and CPU handles everything this port requires. Keep the key for the benchmark project's fine-tunes, and for a future π0.5 or Wall-OSS port where 3B-parameter reference passes at statistical sample sizes get slow on CPU. When that day comes, the clean pattern is: you start the pod, register it as a self-hosted GitHub Actions runner labeled `gpu`, and the agent targets it from CI. The API key never goes to the agent.

## 7. Publishing v0.1

When `PROGRESS.md` says Definition of Done is met: push, make the repo public, tag `v0.1.0`. The README the agent wrote contains the benchmark table and the correctness summary; that is your announcement material. If the repo is public, let the agent add the optional GitHub Actions workflow from BRIEF Phase 6 so a `macos-15` runner keeps the suite green for outside contributors.
