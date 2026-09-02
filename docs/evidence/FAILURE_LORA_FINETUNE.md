# Stage T3 LoRA outcome failure

## Status

The fixed 3,000-update native MLX LoRA run completed and produced a strict,
merged, standard LeRobot checkpoint. The held-out improvement gate and the
Torch/MLX MAE round-trip gate both passed. The unchanged fp32 stats-active
inference-parity gate failed, so Stage T3 does not earn `TRAINING ALPHA`.

No threshold, population, noise draw, split, run length, checkpoint, or output
selection was changed after observing the result.

## Frozen run and artifact identity

- Updates: **3,000**, effective batch **8**, rank **8**, alpha **16**, dropout
  **0**.
- Training time: **4,956.693033957996 seconds**; peak MLX memory:
  **2,478,803,693 bytes**.
- Final loss: **0.15556271374225616**; final smoothed loss:
  **0.15856251862136114**.
- Adapter SHA-256:
  `814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc`.
- Run-state SHA-256:
  `c7c3b86361c0872e26f2088cbd33ada865cf450b6711a9b737ece933c1868c82`.
- Metrics SHA-256:
  `7f3a8c070f8102d7edc0afe5a9f4e5088321d1cdd21548fc21e9c772dbfafc2c`.
- Export-manifest SHA-256:
  `55ad6834cbb3acb9dd565a57296a274d78e7cdc863aa81c3e6ef25da8b66ba03`.
- Outcome-report SHA-256:
  `8b74faf8f9cc96341090f91cfa795ed874c838026416944e4b77a550ad91bc44`.
- Held-out manifest SHA-256:
  `9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3`.
- Held-out metadata SHA-256:
  `f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107`.
- Frozen base-report SHA-256:
  `211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce`.
- Train-statistics SHA-256:
  `5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e`.
- Audited dataset-revision-tree SHA-256:
  `09c0f368ed112082c8a53fa6c83b286834bd855f2f817a7f281c9bb2ad7d3ee4`.
- Native fp32 conversion/model-map SHA-256 values:
  `76d893b95c739cbd2a02598025e360a596edf3a7f90a8c4b1cb63d23ae54b42a`
  and `1664c39008363b98587a8c8fc54ed3af5e899b7fd092944d146bbfe4efc17902`.
- Final checkpoint metadata/model/optimizer SHA-256 values:
  `5d912a1e94bb1809c8fe72570450dc1c6d5c6c8973e8bcc00b1045eb465431ef`,
  `814e6f4b2a78a46b609aa7b48a28b4509f709d3e851e588dcd9a4bd2ca1408dc`,
  and `c9440be75315e04c1812ba18da0e0daccd2990fb0f6fdb1841d40ef7b01ffb5a`.

The finalization process initially stopped after update 3,000 because MLX
appended a safetensors suffix to a temporary filename that ended in `.tmp`.
The final update checkpoint was already durable. Resume restored update 3,000
without replaying an optimizer step and completed the export after the
temporary filename was corrected. The regenerated final adapter matched the
pre-failure adapter byte-for-byte. Checkpoints 2,800, 2,900, and 3,000 remain
under `.cache/training/t3/checkpoints/`.

## Immutable outcome gates

| Gate | Result | Required |
| --- | ---: | ---: |
| Base MLX held-out physical MAE | `4.639846293521779` | Frozen baseline |
| Fine-tuned MLX held-out physical MAE | `2.1164464077779224` | `<= 4.175861664169601` |
| Fine/base MAE ratio | `0.4561458017979897` | `<= 0.9` |
| Torch held-out physical MAE | `2.101113587617874` | Same 56 cases/draws |
| Torch/MLX MAE ratio | `0.9927553940871358` | `[0.95, 1.05]` |
| Image preprocessing max absolute | `3.5762786865234375e-7` | `<= 1e-5` |
| State preprocessing max absolute | `0.0` | `<= 1e-6` |
| Normalized action-chunk max absolute | `0.17762404680252075` | `<= 0.005` |
| Physical action max absolute | `6.632053375244141` | `<= 0.005` |
| Standardized physical max absolute | `0.17762437462806702` | `<= 0.005` |

All **56** prospectively frozen cases participate in the parity ladder. Raw
physical error is worst at ordinal 47, episode 35, frame 176
(`6.632053375244141`). Normalized and standardized error are worst at ordinal
40, episode 34, frame 130 (`0.17762404680252075` and
`0.17762437462806702`). The gate therefore does not depend on a post-run
one-case-per-episode selection.

The tensor-manifest and baseline hashes were committed before training. The
full metadata digest was added during the post-run hardening audit, so it is not
presented as an independent prospective commitment. Instead, the evaluator
reconstructs every metadata field—including task strings and all 56 case
identities—from the precommitted selection algorithm and the pinned dataset
revision. The cached source files must match that revision's audited HF tree;
the reconstructed metadata must then match exactly as well as by full-file
digest.

The evaluator also pins the train statistics, run, metrics, adapter and adapter
metadata, final model and optimizer checkpoint, export manifest, processor
tensors, and all per-case identities. It recomputes the trajectory-affecting
run-configuration digest, reads the metrics bytes once, verifies the frozen
learning-rate schedule and all 3,000 rows semantically, and reconciles the final
row and sample/draw counters to the run and checkpoint. Nested dataset symlinks
are rejected and the 40 GiB free-space floor is enforced before evaluation.
Before each MLX scoring pass, it also proves that all 500 cached native tensors
derive from the validated export: 499 fp32 tensors match by raw tensor hash and
the patch convolution matches the exact OIHW-to-OHWI transpose. The
audit-manifest-only migration added source hashes without changing any model,
processor, adapter, optimizer, or checkpoint bytes.

## Three tested hypotheses

### 1. Data, split, statistics, or preprocessing identity

Ruled out. The 56 cases were captured before training from eight whole unseen
episodes, and the outcome report binds them to the same train-only statistics,
run, adapter, metrics, and export hashes. Tokens and masks are exact. P0 image
preprocessing differs by at most `3.5762786865234375e-7`; state preprocessing
is byte-exact (`0.0`). Both pass their separate immutable `1e-5` and `1e-6`
limits.

An eight-episode diagnostic subset passes on the original fp32 base checkpoint
at max absolute `8.996715223474894e-6`. On the diagnostic trained case used for
boundary tracing (episode 31, frame 0), replacing the native processed inputs
with the exact Torch arrays still yields final max absolute
`0.09159556031227112`; injecting the exact Torch connector output yields
`0.09160420298576355`. The processor or vision input boundary therefore does
not explain that representative failed trajectory. The immutable outcome
itself uses all 56 cases, not this diagnostic subset.

### 2. Adapter checkpoint, merge, tensor name, or layout error

Ruled out. Rebuilding the bf16 base, loading all 458 adapter tensors, and
performing the original fp32 merge on the MLX GPU reproduces all **500**
exported source-layout tensors byte-for-byte: mismatch count **0**, maximum
absolute difference **0.0**. The export strictly loads in both MLX and the
pinned Torch/LeRobot policy and contains **450,046,176** fp32 parameters.

The adapter produced during exact update-3,000 resume is also byte-identical to
the adapter archived from the failed finalization attempt. There is no evidence
of checkpoint loss, stale weights, transposition, or partial export.

### 3. Framework input/noise identity or a localized model defect

No localized defect was demonstrated. Both frameworks consume the exact same
`1x50x32` stored noise. On the episode-31/frame-0 diagnostic trace, injecting
Torch's exact prefix K/V cache into the MLX expert reduces the final difference
from about `0.0916` to
`0.008952289819717407`, locating most of the accumulated difference in the
trained prefix decoder and a smaller remainder in the expert.

At the first denoising evaluation, the suffix-embedding difference is only
`2.9802322387695312e-6`. The first expert-layer output differs by max absolute
`0.0016430020332336426`; after all 16 expert layers the first velocity differs
by `0.017260074615478516`. The Euler trajectory then amplifies the field
difference:

| Euler step | Velocity max absolute | State after step max absolute |
| ---: | ---: | ---: |
| 0 | `0.017260074615478516` | `0.0017260313034057617` |
| 1 | `0.019919991493225098` | `0.0036236047744750977` |
| 2 | `0.02443671226501465` | `0.0059930384159088135` |
| 3 | `0.023112405091524124` | `0.007715433835983276` |
| 4 | `0.028079688549041748` | `0.009673595428466797` |
| 5 | `0.03498363494873047` | `0.012366250157356262` |
| 6 | `0.04876816272735596` | `0.015206113457679749` |
| 7 | `0.07906341552734375` | `0.019207850098609924` |
| 8 | `0.2628980875015259` | `0.04549767076969147` |
| 9 | `0.7516704201698303` | `0.0915878415107727` |

Teacher-forcing each trained layer with the exact Torch input finds distributed
framework reduction-order drift rather than one wrong operation. Prefix-layer
output relative-L2 is approximately `1e-5` to `4.3e-5`, while projected K/V
relative-L2 reaches roughly `0.001` to `0.003`. Expert-layer output relative-L2
is roughly `0.00026` to `0.00049`. The trained velocity field is numerically
sensitive enough to amplify those small, distributed differences in its last
Euler evaluations. This is an inference from the measured boundary trace, not
a claim that the immutable gate should change.

## Consequence

The brief permits fixes only for demonstrated implementation defects and
forbids threshold changes, held-out selection, run extension, or result-based
checkpoint selection. None of the three required investigations found such a
defect, so the failed `0.005` gate is preserved.

Stage T3 is failure-documented. `TRAINING ALPHA` is not written. Stages T4 and
T5 remain dependency-blocked on a passing T3 outcome. A future attempt must be
a new, prospectively frozen experiment—for example, a numerically conditioned
training design or a parity-preserving CPU linear/attention kernel—not a
reinterpretation of this run.
