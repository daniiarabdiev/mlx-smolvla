# v0.1.1 distribution manifest

Documentation patch correcting the pre-publication instructions embedded in
PyPI's description. Source tag `v0.1.1` resolves to `e6f32bbff310773a8e3bc7d8acf35528e859e3d1`.
Runtime behavior is unchanged; package and citation version fields are 0.1.1.

Built from a clean tracked-source export with the release's seven changed
files overlaid, then built all wheels from that source distribution. Earlier
candidates were preserved separately and rejected from publication.

## Verification

- Full suite: 802 passed and one old version assertion failed in 769.52 seconds.
  After synchronizing version fields and assertions, the final public-release
  and distribution rerun passed all 18 tests in 12.65 seconds. No tests were
  skipped and no numerical tolerances changed.
- Four archive audits and Twine checks passed. Each final embedded description
  matches the corrected README and contains no pre-publication instructions.
- Python 3.11/3.12/3.13 wheel installations passed import/version and doctor
  checks outside the repository. Python 3.12 source installation also passed.
- Dependency lock and whitespace checks passed.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mlx_smolvla-0.1.1-cp311-cp311-macosx_14_0_arm64.whl` | 379655 | `c2004cbf3dd11fbaf2ad44cbcbaa126e8619abfcc5028bf70dcbcaaafecf81c4` |
| `mlx_smolvla-0.1.1-cp312-cp312-macosx_14_0_arm64.whl` | 378678 | `4ee1ede450b572d159a3143c85c2d268efe6f07fd9cb47d5847c57fb0e0aae9d` |
| `mlx_smolvla-0.1.1-cp313-cp313-macosx_14_0_arm64.whl` | 378713 | `92b43e850793f2f1a8dfcaf2cec4af24f1fc6ecc80588b6d6b567c844339f3e1` |
| `mlx_smolvla-0.1.1.tar.gz` | 443587 | `9856106332165aca3a313bf51f5e4a5985d7b91cf0e8f74f7912afa93c11bed2` |

## Publication

Publication is authorized. PyPI upload awaits the operator supplying the token
locally; GitHub Release follows verified PyPI upload. No hardware action.
