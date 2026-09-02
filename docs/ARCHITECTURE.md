# Architecture

This document is generated from the pinned CPU/fp32 reference by
`scripts/inspect_reference.py`. It resolves every hypothesis in
`history/BRIEF.md`
Section 3 using checkpoint metadata, installed LeRobot source, and one real
SO-101 observation. The installed PyPI wheel does not embed a source-git SHA;
the immutable package/checkpoint/dataset/VLM revisions below are the reference
identity used for every golden and conversion test.

## Reference pins

- LeRobot `0.6.1`; PyTorch `2.11.0`;
  Transformers `5.5.4`.
- SmolVLA policy source: `.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`.
- SmolVLA configuration source: `.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/configuration_smolvla.py`.
- SmolVLA VLM-with-expert source: `.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py`.
- Policy checkpoint: `lerobot/smolvla_base` at `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- Base VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Golden dataset: `lerobot/svla_so101_pickplace` at `f641879e22172be7e8161d5e6c1503c2d2feb657`.
- Checkpoint inventory: 500 tensors and
  450,046,176 parameters.
- Dataset: 2 real cameras (`observation.images.side`, `observation.images.up`),
  state `[6]`, action `[6]`,
  language task table `True`.

## Fine-tuning freeze policy

T3B inspected the installed LeRobot `0.6.1` dataclass defaults and the code
that applies them before freezing its adapter topology. The configuration
source SHA-256 is
`2fb637cb428fa2fdf1d114646dcffaf4728216bfe5b7039d5d0cac4857ffc4e0`;
the VLM-with-expert implementation source SHA-256 is
`996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf`.
The installed defaults are `freeze_vision_encoder=True`,
`train_expert_only=True`, and `train_state_proj=True`. In the reference code,
`train_expert_only` freezes the complete VLM/prefix side and leaves the action
expert trainable; the vision flag freezes the vision tower explicitly. The
state projection is a separately controlled trainable path in the reference.

`history/BRIEF_T3B.md` narrows the adapter topology further and takes precedence for
the MLX run: only the 16 action-expert layers' four attention projections and
three MLP projections receive LoRA. Vision, connector, language/prefix layers,
state projection, and the action/time input and output projections remain
frozen. At rank 8 this is exactly **112 adapters**, **224 fp32 trainable
tensors**, and **1,708,032 trainable scalars**. The historical 229-adapter
scope remains available only as the explicit `legacy_full` mode so the original
T3 artifacts and configuration digest remain reproducible.

## Verified architecture

```text
two camera frames ──► top/left-pad + [-1, 1] ──► SigLIP vision (12 layers)
                                                    │ 1024 tokens/camera
                                                    ▼
                                      pixel shuffle ×4 + projection
                                                    │ 64 × 960 tokens/camera
task ──► newline + tokenizer (48 tokens) ──────────┤
state (6 → zero-padded 32) ──► linear projection ──┤
                                                    ▼
                   16-layer SmolVLM prefix prefill + per-layer KV cache
                                                    │
noise actions (50 × 32) + timestep ───────────────► 16-layer action expert
                                                    │  self-attn even layers; cross-attn odd layers
                                                    ▼
                    50 × 32 velocity → Euler (10 steps, t=1 → 0) → slice → 50 × 6 actions
```

| Field | Verified value |
| --- | --- |
| Total parameters | 450,046,176 |
| VLM / action expert | 350,165,184 / 98,245,840 |
| Used VLM / expert layers | 16 / 16 |
| VLM / expert hidden width | 960 / 720 |
| Expert MLP intermediate width | 2048 |
| Attention | 15 query heads, 5 KV heads, head dim 64 |
| Expert layer alignment | self-attention [0, 2, 4, 6, 8, 10, 12, 14]; cross-attention [1, 3, 5, 7, 9, 11, 13, 15] |
| State placement | VLM prefix after image and language tokens |
| RoPE base | 10000 |

The expert is **720-wide**, not 576-wide: the checkpoint's 0.75 width multiplier
is applied to the 960-wide text model. Its attention still uses 15 × 64 query
channels and 5 × 64 KV channels; cross-attention layers project cached VLM KV
channels into expert KV channels.

**Layer-depth correction.** The base-VLM configuration declares 32 text layers,
but this SmolVLA safetensors checkpoint contains exactly 144 language-layer
tensors: nine tensors for each of layers `0` through `15`, and none for
`16` through `31`. The native language tree therefore has 16 layers so strict
converted-weight loading proves the actual checkpoint boundary rather than
inventing unused parameters.

## Boundary tensors and cache

| Field | Verified value |
| --- | --- |
| Prepared camera batch | [2, 3, 512, 512] |
| Vision encoder output | [2, 1024, 768] |
| Connector output | [2, 64, 960] |
| Language IDs | [1, 48] |
| Padded state / state embedding | [1, 32] / [1, 1, 960] |
| Prefix / valid prefix tokens | [1, 177, 960] / 139 |
| Action suffix | [1, 50, 720] |
| First layer cache K/V | [1, 5, 177, 64] / [1, 5, 177, 64] |
| Cache layers | 16 |

The two real cameras produce 128 prefix image tokens. With 48 language slots and
one state token, the fixed prefix shape is `[1, 177, 960]`; padded language slots
remain in that shape but are invalid in its pad mask.

## Vision and preprocessing

| Field | Verified value |
| --- | --- |
| Vision family | SmolVLM2 SigLIP-style encoder |
| Vision configuration | 12 layers, hidden 768, 12 heads, MLP 3072 |
| Image / patch size | 512 / 16 |
| Vision activation | gelu_pytorch_tanh |
| Pixel-shuffle scale | 4 |
| Resize | 512×512 bilinear, align_corners=False |
| Padding | left, top; pad value 0.0 before pixel transform |
| Pixel normalization | x * 2 - 1 |
| Language | newline appended; right padded to 48 tokens (max_length) |
| Configured / effective state norm | MEAN_STD / identity |
| Configured / effective action output norm | MEAN_STD / identity |
| Saved normalization-stat keys | `so100-blue.buffer.action`, `so100-red.buffer.action`, `so100.buffer.action` |
| Configured / injected empty cameras | 3 / 0 |

**Normalization correction.** The checkpoint declares mean/std normalization for
state and action, but its saved processor statistics exist only under the robot-
prefixed action keys shown above. The actual SO-101 runtime input uses
`observation.state` and output uses `action`, neither of which matches those
keys. The pinned processor therefore leaves this reference's state unchanged and
also leaves its output action unchanged. The MLX port preserves that effective
identity behavior for the base checkpoint and now reads exact `observation.state`
and `action` mean/std tensors when a fine-tuned checkpoint actually carries
matching keys. It never applies unrelated robot-prefixed statistics.

## Stats-active and public fine-tune checkpoints

The release generality ladder adds two independently generated reference sets.
The first is the unchanged base model with the pinned
`lerobot/svla_so101_pickplace` revision's complete `meta/stats.json` attached
through explicit LeRobot processor state-dict surgery. The model bytes remain
SHA-256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`;
only the saved processor state changes. Its artifact manifest is generated by
`scripts/make_stats_active_reference.py` under
`reference/artifacts/stats-active-base`.

The second target is the public fine-tune
`soonweihong0857/swhfypv3_smolvla_multitask_model` at revision
`5e2491c809ec892427f54db1eb23bf8c4bbbf770`, evaluated against
`soonweihong0857/smolvla_multitask_data` at revision
`ec0062a53e0ae88d46a4341ab0695dfa9f03111b`. It retains the audited 500-tensor,
450,046,176-parameter architecture, declares active six-element mean/std
statistics, and changes the input contract to these two streams:

- `observation.images.wrist_camera`: `(3, 480, 640)`
- `observation.images.top_camera`: `(3, 480, 640)`
- `observation.state`: `(6,)`

The native config parser reads those keys and shapes from each checkpoint.
Architecture mismatches report that complete input contract rather than failing
later in a matrix operation.

## Runtime execution modes

`SmolVLAMLX.from_pretrained` freezes one of two device contracts into each
policy instance. `production` is the public default: every public inference
call owns an MLX Metal context, even if its caller temporarily selected CPU.
`strict` owns an MLX CPU context and enables the compatibility RMSNorm, RoPE,
softmax, and SiLU arithmetic used by the immutable PyTorch CPU parity ladder.
Callers select it with `execution_mode="strict"`; the CLI mirrors that choice
with `--execution-mode strict`.

The modes share one checkpoint conversion, parameter tree, pre/postprocessor,
prefix cache, flow schedule, and action queue. They differ only in the owned
device context and therefore the MLX kernels chosen inside the modules. The
known Metal Vision and Connector reduction differences remain bounded negative
evidence: production fp32 fails the strict eight-case deterministic threshold,
production bf16 passes its fixed threshold, and both pass the 50-frame
statistical threshold. Exact values and timing are in `BENCHMARK.md`.

### Camera-slot behavior

Installed LeRobot 0.6.1 iterates the configured image features and uses every
key present in the observation. Missing configured keys are skipped. Only the
separate `empty_cameras` count appends synthetic `-1` image tensors with a false
mask. Therefore the base checkpoint's three configured keys plus
`empty_cameras=0` produce three real streams when all three are supplied, two
real streams when only the two dataset cameras are supplied, and no implicit
third-camera padding. The native implementation and its reference-conformance
test reproduce this exact behavior. At least one configured camera remains
required.

## Attention, masking, and flow matching

| Field | Verified value |
| --- | --- |
| Prefix content bidirectional | True |
| State attends valid prefix | True |
| Image/language cannot attend state | True |
| Action suffix is causal | True |
| Flow steps / dt | 10 / -0.1 |
| Velocity / returned action width | 32 / 6 |
| Timesteps | 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 |
| Update | x_t = x_t + dt * v_t |
| Noise / action endpoint | t=1 / t=0 |
| Unit-velocity terminal value | -1.0000001192092896 |
| Action queue | refill 50 actions when empty; expose first 50 one at a time |

The prefix's 2-D mask is created from cumulative autoregressive flags: image and
language flags are 0, the state flag is 1. The suffix has 50 flags of 1, making
it causal. During prefix prefill every VLM layer stores post-RoPE K/V. During
each Euler denoising step, even expert layers self-attend over actions while odd
layers cross-attend into that fixed VLM K/V cache; suffix K/V appended by self-
attention are cropped back to the prefix length before the next step.

## LeRobot 0.6.1 async-inference wire contract

P1-4 was audited from the installed, pinned package rather than from a recalled
or redesigned protocol. The protobuf file descriptor has SHA-256
`e116fbf44dd1fc65b67ff255c04857000c28e69055211af5ef3df85ac8d81f8d`,
package `transport`, syntax proto3, and this complete service:

| RPC | Request | Response | Streaming |
| --- | --- | --- | --- |
| `Ready` | `Empty` | `Empty` | unary |
| `SendPolicyInstructions` | `PolicySetup { bytes data = 1; }` | `Empty` | unary |
| `SendObservations` | `Observation { TransferState transfer_state = 1; bytes data = 2; }` | `Empty` | client stream |
| `GetActions` | `Empty` | `Actions { bytes data = 1; }` | unary |

`TransferState` is exactly `UNKNOWN=0`, `BEGIN=1`, `MIDDLE=2`, `END=3`.
The reference client pickles a `RemotePolicyConfig` into `PolicySetup`, pickles
a `TimedObservation`, splits it into 2 MiB messages with a 4 MiB per-message
limit, and unpickles `Actions.data` as `list[TimedAction]`. A single observation
message is marked `END`; longer payloads use `BEGIN`, zero or more `MIDDLE`, and
`END`. Returned actions are CPU Torch tensors. Their first timestamp/timestep
are copied from the observation and subsequent entries advance by `1 / fps`
and one respectively. An empty observation queue returns an empty `Actions`
payload after the configured timeout.

The native server uses that schema and the reference client-side message
classes/chunker directly. It validates the pickled dataclass and feature map,
converts the raw robot feature dictionary with LeRobot's own helper, applies
the old-to-new rename map, removes the reference helper's singleton batch
dimensions, calls native `predict_action_chunk`, applies the checkpoint's
native action unnormalizer, and serializes `TimedAction` objects. A one-item
queue preserves reference newest-observation behavior; `must_go`, already
predicted timestep filtering, similar-state filtering, `Ready` reset, target
inference latency, and action-chunk slicing are preserved. One inference lock
protects MLX state across concurrent RPC workers, while cancellation is polled
during queue, lock, and latency waits. Invalid setup/observation payloads use
explicit gRPC statuses rather than silently becoming an empty action chunk.

The installed source identities are:

| LeRobot 0.6.1 source | SHA-256 |
| --- | --- |
| `async_inference/policy_server.py` | `88ee490394adab0d229e9e2407bec900e89cf72302237c7f128da2c07de37fe7` |
| `async_inference/robot_client.py` | `c557f2f866abe1cc776fe0bd0b167ec3b19978d644876a748eb5efd3afd554ce` |
| `async_inference/helpers.py` | `1457c5bb556b4f10225a9c94ac95553bd53c7b51665f1c918e15bb2acb0ac24e` |
| `async_inference/configs.py` | `9b3514b3c09727dc1336d9586e68bd2c276c401ed34bffb69f9e7f67f0f7f342` |
| `transport/services_pb2.py` | `06efc0dd44bb4a1e532612d6f7994176214028802ecbfd96e9d79376da74657c` |
| `transport/services_pb2_grpc.py` | `3b1cde1f24d00b536e113074385a1bd2f0948d0e8ce2c5cb543ca4debbb377ba` |
| `transport/utils.py` | `3cb63b68d2628996b3999288b33deea67f56be05d6dcc5dae0ec9a4004b3cd8a` |

The `serve` extra is the only dependency edge to gRPC/LeRobot/Torch, and the
CLI imports the server lazily. Base imports therefore remain isolated. The
transport is deliberately insecure and uses pickle because that is the pinned
protocol: binding defaults to `127.0.0.1`; a non-loopback host requires the
explicit `--allow-remote` acknowledgement and remains suitable only for a
trusted private network. The server itself has no robot or serial I/O.

An optional `ServeConfig.latency_log` adds Stage H observation-to-chunk
telemetry without changing the transport. The server exclusively creates a
new mode-0600 JSONL file and refuses an existing path, so evidence from two
sessions cannot be mixed. Receipt time is captured only after the complete
`TimedObservation` payload has been decoded and accepted. Each successful
chunk records the client wall-clock timestamp, server receipt/chunk-ready UTC
times, monotonic server-receipt-to-ready latency, inference latency, timesteps,
action count, and policy configuration. Images, state values, task text, and
action values are never logged. Receipt metadata follows queue replacement,
cancellation restoration, and episode reset under the existing state lock;
the logger flushes and syncs each record before advancing its sequence.

## BRIEF Section 3 verdicts

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| ~450M SmolVLM2 backbone, 64 visual tokens | Confirmed | 450,046,176 parameters; 12-layer 512/16 vision tower and 4× pixel shuffle produce 64 tokens/camera. |
| First 16 VLM layers and cross-attending expert | Confirmed | Checkpoint config and runtime cache both contain 16 layers. |
| ~100M 0.75-wide action expert | Corrected | 98,245,840 parameters, width 720, 16 layers; self-attention even/cross-attention odd. |
| 10-step Euler, 50 actions, 32 padded dimensions | Confirmed | Runtime schedule is 1.0→0.1, dt −0.1, `x_t = x_t + dt*v_t`. |
| Three-camera configuration / state in prefix | Confirmed | Three configured slots; two supplied; `empty_cameras=0`; state is the final VLM prefix token. |
| Prefix bidirectional / suffix causal masks | Confirmed | Runtime 2-D masks exercised on the real frame. |
| 512 pad resize and mean/std stats | Corrected | Resize and `x*2-1` confirmed; declared stats are present but ineffective for this checkpoint's unprefixed input/output keys. |

## Full checkpoint tensor inventory

The inventory below was read from the pinned `model.safetensors` header without
materializing any tensor. It is the conversion contract: every source tensor must
be mapped exactly once in Phase 2.

<details>
<summary>500 tensors</summary>

| Source tensor | dtype | shape |
| --- | --- | --- |
| `model.action_in_proj.bias` | `F32` | `[720]` |
| `model.action_in_proj.weight` | `F32` | `[720, 32]` |
| `model.action_out_proj.bias` | `F32` | `[32]` |
| `model.action_out_proj.weight` | `F32` | `[32, 720]` |
| `model.action_time_mlp_in.bias` | `F32` | `[720]` |
| `model.action_time_mlp_in.weight` | `F32` | `[720, 1440]` |
| `model.action_time_mlp_out.bias` | `F32` | `[720]` |
| `model.action_time_mlp_out.weight` | `F32` | `[720, 720]` |
| `model.state_proj.bias` | `F32` | `[960]` |
| `model.state_proj.weight` | `F32` | `[960, 32]` |
| `model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.0.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.0.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.0.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.0.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.1.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.1.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.1.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.1.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.1.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.1.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.1.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.1.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.1.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.10.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.10.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.10.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.10.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.10.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.10.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.10.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.10.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.10.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.11.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.11.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.11.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.11.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.11.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.11.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.11.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.11.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.11.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.12.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.12.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.12.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.12.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.12.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.12.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.12.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.12.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.12.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.13.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.13.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.13.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.13.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.13.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.13.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.13.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.13.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.13.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.14.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.14.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.14.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.14.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.14.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.14.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.14.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.14.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.14.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.15.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.15.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.15.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.15.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.15.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.15.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.15.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.15.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.15.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.2.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.2.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.2.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.2.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.2.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.2.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.2.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.2.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.2.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.3.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.3.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.3.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.3.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.3.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.3.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.3.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.3.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.3.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.4.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.4.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.4.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.4.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.4.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.4.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.4.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.4.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.4.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.5.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.5.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.5.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.5.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.5.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.5.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.5.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.5.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.5.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.6.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.6.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.6.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.6.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.6.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.6.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.6.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.6.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.6.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.7.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.7.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.7.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.7.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.7.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.7.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.7.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.7.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.7.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.8.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.8.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.8.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.8.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.8.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.8.self_attn.k_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.8.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.8.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.8.self_attn.v_proj.weight` | `BF16` | `[320, 720]` |
| `model.vlm_with_expert.lm_expert.layers.9.input_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.9.mlp.down_proj.weight` | `BF16` | `[720, 2048]` |
| `model.vlm_with_expert.lm_expert.layers.9.mlp.gate_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.9.mlp.up_proj.weight` | `BF16` | `[2048, 720]` |
| `model.vlm_with_expert.lm_expert.layers.9.post_attention_layernorm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.lm_expert.layers.9.self_attn.k_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.layers.9.self_attn.o_proj.weight` | `BF16` | `[720, 960]` |
| `model.vlm_with_expert.lm_expert.layers.9.self_attn.q_proj.weight` | `BF16` | `[960, 720]` |
| `model.vlm_with_expert.lm_expert.layers.9.self_attn.v_proj.weight` | `F32` | `[320, 320]` |
| `model.vlm_with_expert.lm_expert.norm.weight` | `BF16` | `[720]` |
| `model.vlm_with_expert.vlm.lm_head.weight` | `BF16` | `[49280, 960]` |
| `model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight` | `BF16` | `[960, 12288]` |
| `model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight` | `BF16` | `[49280, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.1.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.10.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.11.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.12.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.13.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.14.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.2.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.3.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.4.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.5.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.6.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.7.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.8.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.input_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.mlp.down_proj.weight` | `BF16` | `[960, 2560]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.mlp.gate_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.mlp.up_proj.weight` | `BF16` | `[2560, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.post_attention_layernorm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.self_attn.k_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.self_attn.o_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.self_attn.q_proj.weight` | `BF16` | `[960, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.layers.9.self_attn.v_proj.weight` | `BF16` | `[320, 960]` |
| `model.vlm_with_expert.vlm.model.text_model.norm.weight` | `BF16` | `[960]` |
| `model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight` | `BF16` | `[768, 3, 16, 16]` |
| `model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight` | `BF16` | `[1024, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.1.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.10.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.11.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.3.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.4.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.5.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.6.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.7.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.8.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.layer_norm1.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.layer_norm1.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.layer_norm2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.layer_norm2.weight` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.mlp.fc1.bias` | `BF16` | `[3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.mlp.fc1.weight` | `BF16` | `[3072, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.mlp.fc2.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.mlp.fc2.weight` | `BF16` | `[768, 3072]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.k_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.k_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.out_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.out_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.q_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.q_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.v_proj.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.encoder.layers.9.self_attn.v_proj.weight` | `BF16` | `[768, 768]` |
| `model.vlm_with_expert.vlm.model.vision_model.post_layernorm.bias` | `BF16` | `[768]` |
| `model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight` | `BF16` | `[768]` |

</details>
