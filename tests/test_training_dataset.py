"""Whole-episode split, train-only stats, and Torch-to-NumPy bridge gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_smolvla._lab.training.data import TrainingArtifact


pytestmark = pytest.mark.slow


_DATASET_ROOT = Path(".cache/hf/datasets/svla_so101_pickplace")
_CACHE_DIR = Path(".cache/hf")
_EXPECTED_HOLDOUT = (2, 7, 21, 28, 31, 34, 35, 41)
_EXPECTED_TRAIN = (
    0,
    1,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    29,
    30,
    32,
    33,
    36,
    37,
    38,
    39,
    40,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
)


def test_episode_split_is_exact_disjoint_and_sixteen_percent() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["make_episode_split"])

    first = module.make_episode_split(num_episodes=50, seed=20260901)
    second = module.make_episode_split(num_episodes=50, seed=20260901)

    assert first == second
    assert first.seed == 20260901
    assert first.holdout_fraction == 0.16
    assert first.holdout_episodes == _EXPECTED_HOLDOUT
    assert first.train_episodes == _EXPECTED_TRAIN
    assert len(first.holdout_episodes) == 8
    assert len(first.train_episodes) == 42
    assert set(first.holdout_episodes).isdisjoint(first.train_episodes)
    assert set(first.holdout_episodes) | set(first.train_episodes) == set(range(50))


def test_train_only_statistics_exclude_all_heldout_rows() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["compute_train_statistics"])
    split = module.make_episode_split(num_episodes=50, seed=20260901)

    result = module.compute_train_statistics(_DATASET_ROOT, split.train_episodes)

    assert result.row_count == 10_011
    assert result.excluded_row_count == 1_928
    assert result.total_row_count == 11_939
    assert result.episode_indices == _EXPECTED_TRAIN
    assert result.sha256 == "5aa5ab85e0c71c0adee97782be37907b0918050a8539bb3aab88fe392953948e"
    assert result.stats["observation.state"]["count"] == [10_011]
    assert result.stats["action"]["count"] == [10_011]
    np.testing.assert_allclose(
        result.stats["observation.state"]["mean"],
        [
            7.7335728942111,
            -55.131248744440754,
            66.75018237690006,
            69.09834068655765,
            -53.38197588213268,
            8.229249810138885,
        ],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        result.stats["action"]["std"],
        [
            44.408887141717095,
            36.58955485583073,
            29.045779360525216,
            13.377312902613456,
            17.734983596446423,
            9.116214953109523,
        ],
        rtol=0,
        atol=0,
    )


def test_heldout_manifest_has_seven_full_chunk_cases_per_episode() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["make_heldout_case_specs"])

    specs = module.make_heldout_case_specs(
        _DATASET_ROOT,
        _EXPECTED_HOLDOUT,
        samples_per_episode=7,
        chunk_size=50,
    )

    assert len(specs) == 56
    assert tuple(sorted({spec.episode for spec in specs})) == _EXPECTED_HOLDOUT
    assert all(sum(spec.episode == episode for spec in specs) == 7 for episode in _EXPECTED_HOLDOUT)
    assert len({spec.absolute_index for spec in specs}) == 56
    assert (specs[0].episode, specs[0].frame_index, specs[0].absolute_index) == (2, 0, 569)
    assert (specs[-1].episode, specs[-1].frame_index, specs[-1].absolute_index) == (
        41,
        256,
        9729,
    )


def test_bridge_fixed_case_matches_the_immutable_reference_artifact() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["TrainingDataBridge"])
    artifact = TrainingArtifact(Path(".cache/training/gradient_goldens"))
    bridge = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=(0,),
        sampler_seed=20260901,
    )

    batch = bridge.frame(episode=0, frame_index=100)

    assert (batch.episode, batch.frame_index, batch.absolute_index) == (0, 100, 100)
    assert batch.task == "pink lego brick into the transparent box\n"
    assert batch.physical_action_dim == 6
    for name, actual in (
        ("batch/pixel_values", batch.pixel_values),
        ("batch/pixel_attention_mask", batch.pixel_attention_mask),
        ("batch/input_ids", batch.input_ids),
        ("batch/text_attention_mask", batch.text_attention_mask),
        ("batch/state", batch.state),
        ("batch/actions", batch.actions),
        ("batch/action_is_pad", batch.action_is_pad),
    ):
        np.testing.assert_array_equal(actual, artifact.load(name), err_msg=name)


def test_bridge_sampler_is_reproducible_and_microbatches_are_distinct() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["TrainingDataBridge"])
    split = module.make_episode_split(num_episodes=50, seed=20260901)
    stats = module.compute_train_statistics(_DATASET_ROOT, split.train_episodes)
    first = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=split.train_episodes,
        sampler_seed=20260901,
        stats=stats.processor_stats,
    )
    second = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=split.train_episodes,
        sampler_seed=20260901,
        stats=stats.processor_stats,
    )

    first_evidence = first.semantic_evidence()
    second_evidence = second.semantic_evidence()
    assert first_evidence == second_evidence
    assert first_evidence["format_version"] == 1
    assert len(first_evidence["sha256"]) == 64
    assert set(first_evidence["components"]) == {
        "bridge",
        "config",
        "dataset",
        "loader",
        "metadata",
        "preprocessor",
        "sampler",
    }

    first_batches = [first.next_batch() for _ in range(8)]
    second_batches = [second.next_batch() for _ in range(8)]
    first_ids = [(item.episode, item.frame_index, item.absolute_index) for item in first_batches]
    second_ids = [(item.episode, item.frame_index, item.absolute_index) for item in second_batches]

    assert first_ids == second_ids
    assert len(set(first_ids)) == 8
    assert all(episode in _EXPECTED_TRAIN for episode, _, _ in first_ids)
    assert all(item.pixel_values.dtype == np.float32 for item in first_batches)
    assert all(item.state.shape == (1, 32) for item in first_batches)
    assert all(item.actions.shape == (1, 50, 32) for item in first_batches)
    assert all(item.action_is_pad.shape == (1, 50) for item in first_batches)


def test_bridge_sampler_state_resumes_at_the_exact_next_microbatch() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["TrainingDataBridge"])
    split = module.make_episode_split(num_episodes=50, seed=20260901)
    stats = module.compute_train_statistics(_DATASET_ROOT, split.train_episodes)
    uninterrupted = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=split.train_episodes,
        sampler_seed=20260901,
        stats=stats.processor_stats,
    )
    for _ in range(11):
        uninterrupted.next_batch()
    state = uninterrupted.state_dict()
    expected = uninterrupted.next_batch()

    resumed = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=split.train_episodes,
        sampler_seed=20260901,
        stats=stats.processor_stats,
    )
    resumed.load_state_dict(state)
    actual = resumed.next_batch()

    assert state["samples_consumed"] == 11
    assert (actual.episode, actual.frame_index, actual.absolute_index) == (
        expected.episode,
        expected.frame_index,
        expected.absolute_index,
    )
    np.testing.assert_array_equal(actual.actions, expected.actions)


def test_bridge_semantic_evidence_hashes_materialized_behavior_rows() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["TrainingDataBridge"])
    from datasets import Dataset
    import pyarrow as pa

    first = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=(0,),
        sampler_seed=20260901,
    )
    second = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=(0,),
        sampler_seed=20260901,
    )
    clean = first.semantic_evidence()
    assert second.semantic_evidence() == clean

    source = second.dataset.reader.hf_dataset
    table = source.data.table
    timestamp_index = table.column_names.index("timestamp")
    timestamps = table.column(timestamp_index).combine_chunks().to_numpy().copy()
    timestamps[0] += 1.0
    poisoned_table = table.set_column(
        timestamp_index,
        "timestamp",
        pa.chunked_array([pa.array(timestamps, type=table.schema.field("timestamp").type)]),
    )
    second.dataset.reader.hf_dataset = Dataset(poisoned_table)
    poisoned = second.semantic_evidence()

    assert poisoned["sha256"] != clean["sha256"]
    assert poisoned["components"]["dataset"] != clean["components"]["dataset"]
    assert poisoned["components"]["metadata"] == clean["components"]["metadata"]
    assert poisoned["components"]["config"] == clean["components"]["config"]


def test_bridge_semantic_evidence_rejects_a_replaced_live_iterator() -> None:
    module = __import__("mlx_smolvla._lab.training.dataset", fromlist=["TrainingDataBridge"])
    bridge = module.TrainingDataBridge(
        cache_dir=_CACHE_DIR,
        episodes=(0,),
        sampler_seed=20260901,
    )
    bridge._iterator = iter(())

    try:
        bridge.semantic_evidence()
    except RuntimeError as error:
        assert "iterator" in str(error)
    else:
        raise AssertionError("replaced bridge iterator was accepted")
