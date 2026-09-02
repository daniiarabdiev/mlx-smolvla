#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx_smolvla/native/rmsnorm.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_rmsnorm_native, module) {
  module.def(
      "rms_norm",
      &mlx_smolvla::native::rms_norm,
      "input"_a,
      "weight"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  module.def(
      "reference_rope",
      &mlx_smolvla::native::reference_rope,
      "states"_a,
      "position_ids"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  module.def(
      "reference_softmax",
      &mlx_smolvla::native::reference_softmax,
      "input"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  module.def(
      "reference_silu",
      &mlx_smolvla::native::reference_silu,
      "input"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
}
