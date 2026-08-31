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

} // namespace smolvla_mlx::native
