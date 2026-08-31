#include "smolvla_mlx/native/rmsnorm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace smolvla_mlx::native {
namespace {

constexpr int64_t kWidth = 960;
constexpr int64_t kVectorWidth = 4;
constexpr int64_t kInstructionLevelParallelism = 4;
using Vector = std::array<float, kVectorWidth>;

int64_t ceil_log2(int64_t value) {
  int64_t result = 0;
  for (--value; value > 0; value >>= 1) {
    ++result;
  }
  return result;
}

Vector load_vector(const char* data, int64_t stride, int64_t index) {
  const auto* source = reinterpret_cast<const float*>(data + stride * index);
  Vector values{};
  for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
    values[lane] = source[lane];
  }
  return values;
}

std::array<Vector, kInstructionLevelParallelism> cascade_sum(
    const char* input,
    int64_t row_stride,
    int64_t column_stride,
    int64_t size) {
  constexpr int64_t kLevels = 4;
  const int64_t level_power = std::max(int64_t(4), ceil_log2(size) / kLevels);
  const int64_t level_step = 1 << level_power;
  const int64_t level_mask = level_step - 1;

  std::array<std::array<Vector, kInstructionLevelParallelism>, kLevels> accumulators{};
  for (auto& level : accumulators) {
    for (auto& row : level) {
      row.fill(0.0F);
    }
  }

  int64_t index = 0;
  for (; index + level_step <= size;) {
    for (int64_t step = 0; step < level_step; ++step, ++index) {
      const char* sum_base = input + index * row_stride;
      for (int64_t row = 0; row < kInstructionLevelParallelism; ++row) {
        const auto values = load_vector(sum_base, column_stride, row);
        for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
          accumulators[0][row][lane] += values[lane];
        }
      }
    }

    for (int64_t level = 1; level < kLevels; ++level) {
      for (int64_t row = 0; row < kInstructionLevelParallelism; ++row) {
        for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
          accumulators[level][row][lane] += accumulators[level - 1][row][lane];
          accumulators[level - 1][row][lane] = 0.0F;
        }
      }
      const int64_t mask = level_mask << (level * level_power);
      if ((index & mask) != 0) {
        break;
      }
    }
  }

  for (; index < size; ++index) {
    const char* sum_base = input + index * row_stride;
    for (int64_t row = 0; row < kInstructionLevelParallelism; ++row) {
      const auto values = load_vector(sum_base, column_stride, row);
      for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
        accumulators[0][row][lane] += values[lane];
      }
    }
  }

  for (int64_t level = 1; level < kLevels; ++level) {
    for (int64_t row = 0; row < kInstructionLevelParallelism; ++row) {
      for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
        accumulators[0][row][lane] += accumulators[level][row][lane];
      }
    }
  }
  return accumulators[0];
}

float row_sum(const std::array<float, kWidth>& squared) {
  const int64_t vector_count = kWidth / kVectorWidth;
  const int64_t grouped_size = vector_count / kInstructionLevelParallelism;
  auto partial_sums = cascade_sum(
      reinterpret_cast<const char*>(squared.data()),
      sizeof(float) * kVectorWidth * kInstructionLevelParallelism,
      sizeof(float) * kVectorWidth,
      grouped_size);

  for (int64_t index = grouped_size * kInstructionLevelParallelism;
       index < vector_count;
       ++index) {
    const auto values = load_vector(
        reinterpret_cast<const char*>(squared.data()),
        sizeof(float) * kVectorWidth,
        index);
    for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
      partial_sums[0][lane] += values[lane];
    }
  }

  for (int64_t partial = 1; partial < kInstructionLevelParallelism; ++partial) {
    for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
      partial_sums[0][lane] += partial_sums[partial][lane];
    }
  }

  float total = 0.0F;
  for (int64_t lane = 0; lane < kVectorWidth; ++lane) {
    total += partial_sums[0][lane];
  }
  return total;
}

class CPUReferenceRMSNorm final : public mx::Primitive {
 public:
  CPUReferenceRMSNorm(mx::Stream stream, float eps)
      : Primitive(stream), eps_(eps) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& input = inputs.at(0);
    const auto& weight = inputs.at(1);
    auto& output = outputs.at(0);
    output.set_data(mx::allocator::malloc(output.nbytes()));

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(input);
    encoder.set_input_array(weight);
    encoder.set_output_array(output);
    encoder.dispatch([input = mx::array::unsafe_weak_copy(input),
                      weight = mx::array::unsafe_weak_copy(weight),
                      output = mx::array::unsafe_weak_copy(output),
                      eps = eps_]() mutable {
      const auto* input_data = input.data<float>();
      const auto* weight_data = weight.data<float>();
      auto* output_data = output.data<float>();
      const size_t rows = input.size() / kWidth;

      for (size_t row = 0; row < rows; ++row) {
        const auto* input_row = input_data + row * kWidth;
        auto* output_row = output_data + row * kWidth;
        std::array<float, kWidth> squared{};
        for (int64_t column = 0; column < kWidth; ++column) {
          squared[column] = input_row[column] * input_row[column];
        }

        const float mean = row_sum(squared) / static_cast<float>(kWidth);
        const float inverse_rms = 1.0F / std::sqrt(mean + eps);
        for (int64_t column = 0; column < kWidth; ++column) {
          const float normalized = input_row[column] * inverse_rms;
          output_row[column] = normalized * weight_data[column];
        }
      }
    });
  }

  void eval_gpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("CPUReferenceRMSNorm only supports CPU streams");
  }

  const char* name() const override {
    return "CPUReferenceRMSNorm";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& other_norm = static_cast<const CPUReferenceRMSNorm&>(other);
    return eps_ == other_norm.eps_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs)
      override {
    return {inputs.at(0).shape()};
  }

 private:
  float eps_;
};

void validate_inputs(
    const mx::array& input,
    const mx::array& weight,
    float eps,
    const mx::Stream& stream) {
  if (stream.device.type != mx::Device::cpu) {
    throw std::invalid_argument("rms_norm requires an MLX CPU stream");
  }
  if (input.dtype() != mx::float32 || weight.dtype() != mx::float32) {
    throw std::invalid_argument("rms_norm requires float32 input and weight");
  }
  if (input.ndim() == 0 || input.shape(-1) != kWidth) {
    throw std::invalid_argument("rms_norm requires a final dimension of 960");
  }
  if (weight.ndim() != 1 || weight.shape(0) != kWidth) {
    throw std::invalid_argument("rms_norm requires a float32 weight of shape [960]");
  }
  if (!std::isfinite(eps) || eps <= 0.0F) {
    throw std::invalid_argument("rms_norm requires a finite positive epsilon");
  }
}

} // namespace

mx::array rms_norm(
    const mx::array& input,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice stream_or_device) {
  const auto stream = mx::to_stream(stream_or_device);
  validate_inputs(input, weight, eps, stream);
  const auto contiguous_input = mx::contiguous(input, false, stream);
  const auto contiguous_weight = mx::contiguous(weight, false, stream);
  return mx::array(
      contiguous_input.shape(),
      mx::float32,
      std::make_shared<CPUReferenceRMSNorm>(stream, eps),
      {contiguous_input, contiguous_weight});
}

} // namespace smolvla_mlx::native
