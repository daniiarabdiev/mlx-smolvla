"""Exact identities and audit anchors for the captured Stage T3 evidence."""

from reference.discovery import DATASET_ID, DATASET_REVISION

FROZEN_EVALUATION_MANIFEST_SHA256 = (
    "9cabca6cd21e8658a94e42980af3e91ecd8ff5ed5daca5f75eb7a1ebd1d261a3"
)
# Recorded during the post-run audit. The evaluator therefore also reconstructs
# every metadata field from the precommitted selection algorithm and pinned data.
FROZEN_EVALUATION_METADATA_SHA256 = (
    "f49ee54aead7ce3ede7b94d5638864afd2e12ef57ae2622eb6574333820cd107"
)
FROZEN_BASE_REPORT_SHA256 = (
    "211d6778b0530208ca2e81abe6f4002cc683e24d496a09ddbe39c100ebd4f7ce"
)
FROZEN_TRAIN_STATISTICS_SHA256 = (
    "5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e"
)
FROZEN_DATASET_REVISION_TREE_SHA256 = (
    "09c0f368ed112082c8a53fa6c83b286834bd855f2f817a7f281c9bb2ad7d3ee4"
)
FROZEN_CHECKPOINT_REVISION_TREE_SHA256 = (
    "274e35c748a00f4f72889280d6a9cce939c06c63d41871f3373f0777893ebfe3"
)
FROZEN_TOKENIZER_REVISION_TREE_SHA256 = (
    "384e21694bf540b85f741ad3c93be4d36f7054d08b5434409ee64cb308d0dff3"
)


def frozen_export_audit_metadata(run_config_sha256: str) -> dict[str, object]:
    """Return the shared run/evaluation source chain required in every T3 export."""

    if (
        len(run_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in run_config_sha256)
    ):
        raise ValueError("training run configuration is not a lowercase SHA-256 digest")
    return {
        "run_config_sha256": run_config_sha256,
        "evaluation_manifest_sha256": FROZEN_EVALUATION_MANIFEST_SHA256,
        "evaluation_metadata_sha256": FROZEN_EVALUATION_METADATA_SHA256,
        "base_report_sha256": FROZEN_BASE_REPORT_SHA256,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
    }
