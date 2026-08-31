#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace smolvla_mlx::native {

namespace mx = mlx::core;

mx::array rms_norm(
    const mx::array& input,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice stream_or_device = {});

mx::array reference_rope(
    const mx::array& states,
    const mx::array& position_ids,
    mx::StreamOrDevice stream_or_device = {});

mx::array reference_softmax(
    const mx::array& input,
    mx::StreamOrDevice stream_or_device = {});

mx::array reference_silu(
    const mx::array& input,
    mx::StreamOrDevice stream_or_device = {});

} // namespace smolvla_mlx::native
