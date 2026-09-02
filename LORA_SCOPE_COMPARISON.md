# Which layers to adapt: T3 versus T3B

## Decision

For this checkpoint and dataset, expert-only LoRA is the better default scope.
It preserves every fixed training outcome gate while reducing the worst
MLX-versus-PyTorch normalized action divergence by **13.62x** and the worst raw
physical divergence by **50.44x** relative to the original full-scope T3 LoRA
run. It also trains **28.46% faster** with **56.51% fewer** trainable scalars.

The tradeoff is measurable: T3B's held-out physical MAE is **6.55% higher**
than T3's, and its measured peak MLX memory is **16.98% higher**. Both trained
models easily pass the fixed held-out-improvement and Torch/MLX round-trip
gates. T3B still misses the separately derived deterministic bound, so this is
a `TRAINING ALPHA (STATISTICAL)` result rather than strict deterministic alpha.

## Frozen side-by-side result

Both runs use rank 8, alpha 16, dropout 0, 3,000 optimizer updates, effective
batch 8, the same learning-rate schedule and seed, the same eight held-out
episodes, the same 56 cases, and the same stored `1x50x32` noise for each case.
No timing was rerun for this comparison; the table uses each run's already
captured resource evidence.

| Measurement | T3: legacy full scope | T3B: expert only |
| --- | ---: | ---: |
| LoRA adapters / tensors | `229` / `458` | `112` / `224` |
| Trainable scalars | `3,927,680` | `1,708,032` |
| Training wall-clock | `4,956.693033957996 s` | `3,545.863376834008 s` |
| Peak MLX memory | `2,478,803,693 B` | `2,899,690,676 B` |
| Base MLX held-out physical MAE | `4.639846293521779` | `4.639846293521779` |
| Fine-tuned MLX held-out physical MAE | `2.1164464077779224` | `2.2550044155546596` |
| Fine/base MAE ratio, required `<= 0.9` | `0.4561458017979897` — pass | `0.486008430646319` — pass |
| Torch held-out physical MAE | `2.101113587617874` | `2.2548798021106493` |
| Torch/MLX MAE ratio, required `[0.95, 1.05]` | `0.9927553940871358` — pass | `0.9999447391574267` — pass |
| PyTorch self-consistency floor `F` | `0.00003549918286283038` | `0.00002467632293701172` |
| PyTorch float64 sensitivity `F64` | `0.00003549918286283038` | `0.000022446572480461224` |
| Derived threshold `max(0.005, 3F)` | diagnostic only | `0.005` |
| Image preprocessing max absolute | `3.5762786865234375e-7` — pass | `3.5762786865234375e-7` — pass |
| State preprocessing max absolute | `0.0` — pass | `0.0` — pass |
| MLX/reference normalized max absolute | `0.17762404680252075` | `0.013038858771324158` |
| MLX/reference raw physical max absolute | `6.632053375244141` | `0.13149452209472656` |
| Derived deterministic gate | original fixed gate failed | fail at `2.6078x` threshold |

T3's floor is retrospective and diagnostic only. T3B's floor was computed,
written, hashed, and recorded before the comparison start marker and before the
first MLX-versus-PyTorch comparison of that checkpoint.

## Euler amplification

The T3 curve is the previously recorded episode-31/frame-0 boundary trace. The
T3B curve is a post-verdict diagnostic on the prospectively frozen case with
the largest normalized action difference: ordinal 24, episode 28, frame 87,
absolute index 6307. Both traces use the stored noise, ten fp32 Euler steps, and
`dt = -0.1`. Values are maximum absolute MLX-versus-PyTorch differences over
the padded velocity/state tensors; for T3B the maxima are in the six physical
action dimensions at every step.

| Euler step | T3 velocity max | T3 state after step | T3B velocity max | T3B state after step |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.017260074615478516` | `0.0017260313034057617` | `0.0035195350646972656` | `0.0003519505262374878` |
| 1 | `0.019919991493225098` | `0.0036236047744750977` | `0.004750430583953857` | `0.0008269846439361572` |
| 2 | `0.02443671226501465` | `0.0059930384159088135` | `0.00719451904296875` | `0.0015464425086975098` |
| 3 | `0.023112405091524124` | `0.007715433835983276` | `0.01281505823135376` | `0.002642720937728882` |
| 4 | `0.028079688549041748` | `0.009673595428466797` | `0.02077704668045044` | `0.004720434546470642` |
| 5 | `0.03498363494873047` | `0.012366250157356262` | `0.024213097989559174` | `0.006602033972740173` |
| 6 | `0.04876816272735596` | `0.015206113457679749` | `0.030286550521850586` | `0.008135437965393066` |
| 7 | `0.07906341552734375` | `0.019207850098609924` | `0.016949772834777832` | `0.00983041524887085` |
| 8 | `0.2628980875015259` | `0.04549767076969147` | `0.012452125549316406` | `0.011075630784034729` |
| 9 | `0.7516704201698303` | `0.0915878415107727` | `0.035959720611572266` | `0.013038858771324158` |

The original scope adapts the trained prefix decoder as well as the expert;
T3's earlier boundary investigation found distributed prefix K/V and expert
reduction-order drift that the learned field sharply amplified in its last two
Euler evaluations. T3B leaves the prefix decoder frozen and adapts only expert
attention and MLP linears. The new curve has no comparable late-step blow-up,
although its accumulated final difference remains above the prospective
`0.005` deterministic threshold.

This supports expert-only LoRA as the release training default. It does not
prove a localized framework defect and does not justify changing any parity
tolerance.

## Evidence identity

- Original T3 failure SHA-256:
  `d6654131c4acf86de13206f210f1ea1a82e3aad18871e5b64428bdf1dbeed7c6`.
- T3B run-state SHA-256:
  `2af527bea4691862e89eb6daa674e6d99309668ac45f57397786976aab3c301e`.
- T3B adapter SHA-256:
  `cce4eed18a7311594950f6d4da33a44dd337f66fbc29162d686c5338ec044826`.
- T3B merged model SHA-256:
  `858704fa572501d9e5a048076f8da692693b90c463feda29201a72f3f0b18883`.
- T3B floor SHA-256:
  `28d83926a70e507671bfd694e032f81b71093d475075aad627b3c24c5b334efc`.
- T3B comparison SHA-256:
  `6aa8e3771bbbd81ecd9599ec9605a4e1efb804fa9ec66c4f82d2d6aea3eb00c6`.
- T3B authoritative evaluation SHA-256:
  `1e337f0bb87aa66a4270c526dd918bd18807aa6aa5291a59b119780080ea9eca`.

