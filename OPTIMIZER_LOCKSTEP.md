# SmolVLA 25-Step Optimizer Lockstep

## Decision

**Stage T2 passes.** Native MLX reproduces the first 25 updates of the pinned
LeRobot SmolVLA CPU/fp32 training configuration. All 25 pre-update losses and
all 155 final selected parameter tensors pass the immutable limits.

This is cumulative optimizer evidence beyond T1's step-zero derivative. It
does not claim training quality; Stage T3 owns real held-out improvement,
checkpoint export, and PyTorch round-trip gates.

## Reproduce

```bash
export HF_HOME="$PWD/.cache/hf"
export UV_CACHE_DIR="$PWD/.cache/uv"
export SMOLVLA_MLX_CACHE="$PWD/.cache/smolvla_mlx"
make optimizer-goldens
make optimizer-lockstep
```

The first command runs the actual PyTorch/LeRobot loop and writes
`.cache/training/optimizer_goldens/`. The second runs native MLX and writes
`.cache/training/t2-lockstep.json`. Both artifacts remain local and ignored.

## Checked semantics

| Semantic | Exact behavior checked |
| --- | --- |
| Selected parameters | state projection plus action expert; 155 tensors / 99,880,992 scalars |
| AdamW LR | peak `1e-4` |
| Moments | betas `(0.9, 0.95)`, fp32 state |
| Bias correction | first and second moments; optimizer step starts at one |
| Epsilon | `1e-8`, after the bias-corrected square-root second moment |
| Weight decay | decoupled multiplicative decay before the moment update; coefficient `1e-10` |
| Gradient clipping | PyTorch-order multi-tensor global fp32 L2 norm, maximum 10, denominator epsilon `1e-6` |
| Update order | forward → backward → clip → AdamW → zero gradients → scheduler |
| Scheduler | 1,000-step linear warmup, 30,000-step cosine decay, `2.5e-6` floor |
| Scheduler horizon | first 25 updates of the default 100,000-step training run |

Small cross-framework tests additionally enlarge decay and epsilon and alter
the betas so ordering mistakes cannot hide behind the production preset's tiny
decay. One-step and 25-step results match `torch.optim.AdamW`; all first 25 LR
values match the installed LeRobot scheduler at zero tolerance.

The 100,000-step horizon is deliberate. Configuring a 25-step total run invokes
LeRobot's short-run scaling, truncates warmup to zero, and tests a different
edge case. Here LR begins at `9.990009990009991e-8` and reaches
`2.4975024975025017e-6` on update 24.

## Data and artifact integrity

The optimizer-isolation run repeats the exact T1 real, non-padded processed
batch at episode/frame/absolute index `0/100/100`. A single PyTorch seed
`20260831` generates 25 distinct sequential noise/timestep pairs; MLX consumes
those serialized draws without sampling.

Before reference update 0, capture proves the batch and every selected initial
parameter exactly equal T1. Before MLX update 0, the strict native loader proves
the same 155 fp32 values again. The 330-payload, 399,852,897-byte optimizer
artifact was captured twice with identical manifest SHA-256:

```text
88c3febc7da3e553bcb7c26f261721369ed1f56efd457887b7d43d50a077807c
```

It binds T1 manifest:

```text
b029a0ed66312e785cb8aa3f1db0affb16c9502ad7b5d0fe0feea3177bf8c145
```

The complete MLX report SHA-256 is:

```text
da8cabf5eecf4379065771b3a74407c47290b8aee9c2d0a9756893b6dd87a6a4
```

## Immutable gate result

| Metric | Result | Required gate |
| --- | ---: | ---: |
| Loss steps passing | 25 / 25 | 25 / 25 |
| Maximum per-step loss relative difference | 1.3529624562582406e-6 | ≤ 1e-3 |
| Final parameter tensors passing | 155 / 155 | 155 / 155 |
| Maximum final parameter relative-L2 drift | 2.8499913470883435e-8 | ≤ 5e-3 |

### All 25 pre-update losses

| Step | LR used | PyTorch loss | MLX loss | Relative difference |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 9.990009990009991e-8 | 2.101923942565918 | 2.101925849914551 | 9.074298999059449e-7 |
| 1 | 1.9980019980019304e-7 | 2.180537223815918 | 2.1805384159088135 | 5.466968793230055e-7 |
| 2 | 2.997002997003007e-7 | 1.2219771146774292 | 1.2219771146774292 | 0 |
| 3 | 3.996003996003972e-7 | 0.956856906414032 | 0.9568575024604797 | 6.229211951739804e-7 |
| 4 | 4.995004995004937e-7 | 1.3491101264953613 | 1.3491108417510986 | 5.301685335078884e-7 |
| 5 | 5.994005994006014e-7 | 0.8366235494613647 | 0.8366238474845886 | 3.5622141412207025e-7 |
| 6 | 6.993006993006979e-7 | 1.0368428230285645 | 1.0368435382843018 | 6.898400812723595e-7 |
| 7 | 7.992007992007944e-7 | 1.8503063917160034 | 1.850308895111084 | 1.3529624562582406e-6 |
| 8 | 8.99100899100902e-7 | 1.8900256156921387 | 1.8900264501571655 | 4.4150990331942285e-7 |
| 9 | 9.990009990009984e-7 | 1.7298120260238647 | 1.729814052581787 | 1.1715480594856972e-6 |
| 10 | 1.098901098901095e-6 | 1.7557307481765747 | 1.7557321786880493 | 8.147670000625333e-7 |
| 11 | 1.1988011988011915e-6 | 1.678403377532959 | 1.6784045696258545 | 7.102541090330969e-7 |
| 12 | 1.2987012987012992e-6 | 0.6517868041992188 | 0.6517871022224426 | 4.572403460102305e-7 |
| 13 | 1.3986013986013958e-6 | 1.8158870935440063 | 1.815888524055481 | 7.877755614295895e-7 |
| 14 | 1.4985014985014923e-6 | 1.0061006546020508 | 1.0061017274856567 | 1.0663780020910487e-6 |
| 15 | 1.5984015984015998e-6 | 0.9501298069953918 | 0.95013028383255 | 5.018652763994779e-7 |
| 16 | 1.6983016983016963e-6 | 1.0700381994247437 | 1.0700390338897705 | 7.798460160619313e-7 |
| 17 | 1.7982017982017928e-6 | 1.0001074075698853 | 1.0001083612442017 | 9.535718955662363e-7 |
| 18 | 1.8981018981019006e-6 | 0.6507793664932251 | 0.650780200958252 | 1.282254892855697e-6 |
| 19 | 1.998001998001997e-6 | 0.7287701368331909 | 0.7287707924842834 | 8.996678916871831e-7 |
| 20 | 2.0979020979020936e-6 | 0.8545678853988647 | 0.8545681834220886 | 3.487414270638692e-7 |
| 21 | 2.1978021978022014e-6 | 0.5203338861465454 | 0.5203340649604797 | 3.4365229535677636e-7 |
| 22 | 2.2977022977022977e-6 | 0.5148371458053589 | 0.5148375630378723 | 8.104164915588176e-7 |
| 23 | 2.3976023976023944e-6 | 0.354121595621109 | 0.3541215658187866 | 8.415844375552342e-8 |
| 24 | 2.4975024975025017e-6 | 0.5079650282859802 | 0.5079655647277832 | 1.0560605024102234e-6 |

### Five largest final parameter drifts

| Tensor | Relative L2 | Cosine | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| `expert.layers.1.self_attn.q_proj.weight` | 2.8499913470883435e-8 | 0.9999999999999992 | 2.2351741790771484e-8 |
| `expert.layers.1.self_attn.o_proj.weight` | 1.939221166057099e-8 | 0.9999999999999998 | 1.862645149230957e-8 |
| `expert.layers.5.self_attn.o_proj.weight` | 1.796292933841857e-8 | 0.9999999999999996 | 3.259629011154175e-8 |
| `expert.layers.3.self_attn.q_proj.weight` | 1.785367358266342e-8 | 0.9999999999999998 | 9.313225746154785e-9 |
| `expert.layers.9.self_attn.o_proj.weight` | 1.769763506243787e-8 | 0.9999999999999998 | 4.0512531995773315e-8 |

## Clipping and resource evidence

Clipping is not a no-op: reference step-zero norm is
`43.20928955078125`. Twenty-four steps clip; stochastic step 23 has norm
`6.328668594360352` and correctly uses coefficient 1.0. Across all steps, the
maximum MLX/PyTorch gradient-norm relative difference is
`4.9728077828962536e-5`; maximum clip-coefficient absolute difference is
`3.2007518114496314e-5`.

| Resource metric | Result |
| --- | ---: |
| 25 synchronized MLX updates | 33.535 seconds |
| Total strict load/integrity/identity/update/check | 34.561 seconds |
| Active MLX memory | 3,181,112,565 bytes (2.963 GiB) |
| Peak MLX memory | 3,373,751,277 bytes (3.142 GiB) |
| Disk free before | 582,258,147,328 bytes (542.270 GiB) |
| Disk free after | 582,256,947,200 bytes (542.269 GiB) |

The free-space floor remains 40 GiB. No upload, credential, robot tree, vendor
fork, serial port, or hardware access was involved. `training/__init__.py`
remains empty and base imports remain free of Torch, LeRobot, and Transformers.

## Conclusion

Step-zero gradients and 25-step optimizer evolution are both independently
verified against PyTorch. Stage T2 is complete. Stage T3 remains the next
eligible money stage: real MLX LoRA fine-tuning, held-out improvement, standard
checkpoint export, and PyTorch round-trip proof.
