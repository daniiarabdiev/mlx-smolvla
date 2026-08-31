#include "smolvla_mlx/native/rmsnorm.h"
#include "smolvla_mlx/native/rope_prefix_corrections_v2.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <mlx/allocator.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace smolvla_mlx::native {
namespace {

constexpr int64_t kVLMWidth = 960;
constexpr int64_t kExpertWidth = 720;
constexpr int64_t kHeadDimension = 64;
constexpr int64_t kRopeHalfDimension = kHeadDimension / 2;
constexpr int64_t kVectorWidth = 4;
constexpr int64_t kInstructionLevelParallelism = 4;
constexpr float kRopeBase = 10'000.0F;
constexpr int32_t kFixedPrefixLength = 177;
constexpr size_t kFixedPrefixCorrectionCount =
    static_cast<size_t>(kFixedPrefixLength) * 2 * kRopeHalfDimension;
using Vector = std::array<float, kVectorWidth>;

int base64_value(char character) {
  if (character >= 'A' && character <= 'Z') {
    return character - 'A';
  }
  if (character >= 'a' && character <= 'z') {
    return character - 'a' + 26;
  }
  if (character >= '0' && character <= '9') {
    return character - '0' + 52;
  }
  if (character == '+') {
    return 62;
  }
  if (character == '/') {
    return 63;
  }
  return -1;
}

const std::array<int8_t, kFixedPrefixCorrectionCount>& fixed_prefix_rope_corrections() {
  static const auto corrections = [] {
    std::array<uint8_t, kFixedPrefixCorrectionCount> decoded{};
    uint32_t buffer = 0;
    int buffered_bits = 0;
    size_t output_index = 0;
    for (char character : std::string_view(detail::kFixedPrefixRoPECorrectionBase64)) {
      const int value = base64_value(character);
      if (value < 0) {
        throw std::runtime_error("invalid fixed-prefix RoPE base64 correction table");
      }
      buffer = (buffer << 6) | static_cast<uint32_t>(value);
      buffered_bits += 6;
      while (buffered_bits >= 8) {
        buffered_bits -= 8;
        if (output_index >= decoded.size()) {
          throw std::runtime_error("fixed-prefix RoPE correction table is too long");
        }
        decoded[output_index++] = static_cast<uint8_t>((buffer >> buffered_bits) & 0xFFU);
      }
    }
    if (output_index != decoded.size() || buffered_bits != 0) {
      throw std::runtime_error("fixed-prefix RoPE correction table has an invalid length");
    }
    std::array<int8_t, kFixedPrefixCorrectionCount> result{};
    std::memcpy(result.data(), decoded.data(), decoded.size());
    return result;
  }();
  return corrections;
}

float apply_raw_bit_correction(float value, int8_t correction) {
  if (correction == 0) {
    return value;
  }
  const auto bits = std::bit_cast<uint32_t>(value);
  const auto corrected_bits = static_cast<uint32_t>(static_cast<int64_t>(bits) + correction);
  return std::bit_cast<float>(corrected_bits);
}

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

float row_sum(const float* squared, int64_t width) {
  const int64_t vector_count = width / kVectorWidth;
  const int64_t grouped_size = vector_count / kInstructionLevelParallelism;
  auto partial_sums = cascade_sum(
      reinterpret_cast<const char*>(squared),
      sizeof(float) * kVectorWidth * kInstructionLevelParallelism,
      sizeof(float) * kVectorWidth,
      grouped_size);

  for (int64_t index = grouped_size * kInstructionLevelParallelism;
       index < vector_count;
       ++index) {
    const auto values = load_vector(
        reinterpret_cast<const char*>(squared),
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

#if defined(__aarch64__)
// This is a focused adaptation of SLEEF's xexpf from commit
// 5a1d179df9cf652951b59010a2d2075372d67f68, which is the version vendored by
// the pinned PyTorch 2.11.0 CPU build. It uses the same four-wide NEON/FMA
// operations as PyTorch's Sleef_expf4_u10 dispatch. See NOTICE for the Boost
// Software License 1.0 notice.
float32x4_t sleef_expf4_u10(float32x4_t values) {
  const auto q = vcvtq_s32_f32(vrndnq_f32(vmulq_n_f32(values, 1.4426950408889634F)));
  const auto q_as_float = vcvtq_f32_s32(q);

  auto reduced = vfmaq_n_f32(values, q_as_float, -0.693145751953125F);
  reduced = vfmaq_n_f32(reduced, q_as_float, -1.428606765330187045e-06F);

  auto polynomial = vdupq_n_f32(0.000198527617612853646278381F);
  polynomial = vfmaq_f32(vdupq_n_f32(0.00139304355252534151077271F), reduced, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(0.00833336077630519866943359F), reduced, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(0.0416664853692054748535156F), reduced, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(0.166666671633720397949219F), reduced, polynomial);
  polynomial = vfmaq_f32(vdupq_n_f32(0.5F), reduced, polynomial);
  polynomial = vaddq_f32(
      vdupq_n_f32(1.0F),
      vfmaq_f32(reduced, vmulq_f32(reduced, reduced), polynomial));

  const auto half_q = vshrq_n_s32(q, 1);
  const auto power_of_two = [](int32x4_t exponents) {
    const auto exponent_bits = vshlq_n_s32(vaddq_s32(exponents, vdupq_n_s32(127)), 23);
    return vreinterpretq_f32_s32(exponent_bits);
  };
  polynomial = vmulq_f32(
      vmulq_f32(polynomial, power_of_two(half_q)),
      power_of_two(vsubq_s32(q, half_q)));

  const auto underflow = vcltq_f32(values, vdupq_n_f32(-104.0F));
  polynomial = vreinterpretq_f32_u32(
      vbicq_u32(vreinterpretq_u32_f32(polynomial), underflow));
  const auto overflow = vcltq_f32(vdupq_n_f32(100.0F), values);
  return vbslq_f32(
      overflow,
      vdupq_n_f32(std::numeric_limits<float>::infinity()),
      polynomial);
}

float32x4_t replace_prefix_lanes(
    float32x4_t original,
    float32x4_t replacement,
    int64_t lane_count) {
  std::array<float, kVectorWidth> original_values{};
  std::array<float, kVectorWidth> replacement_values{};
  vst1q_f32(original_values.data(), original);
  vst1q_f32(replacement_values.data(), replacement);
  std::copy_n(replacement_values.begin(), lane_count, original_values.begin());
  return vld1q_f32(original_values.data());
}

float reduce_maximum(float32x4_t values) {
  const auto shuffle_64_bit = vextq_f32(values, values, 2);
  values = vmaxq_f32(values, shuffle_64_bit);
  const auto shuffle_32_bit = vrev64q_f32(values);
  values = vmaxq_f32(values, shuffle_32_bit);
  return vgetq_lane_f32(values, 0);
}

float reduce_sum(float32x4_t values) {
  const auto shuffle_64_bit = vextq_f32(values, values, 2);
  values = vaddq_f32(values, shuffle_64_bit);
  const auto shuffle_32_bit = vrev64q_f32(values);
  values = vaddq_f32(values, shuffle_32_bit);
  return vgetq_lane_f32(values, 0);
}

void softmax_row(const float* input, float* output, int64_t size) {
  if (size < kVectorWidth) {
    float maximum = input[0];
    for (int64_t index = 1; index < size; ++index) {
      maximum = std::max(maximum, input[index]);
    }
    float sum = 0.0F;
    for (int64_t index = 0; index < size; ++index) {
      std::array<float, kVectorWidth> values{};
      values[0] = input[index] - maximum;
      output[index] = vgetq_lane_f32(sleef_expf4_u10(vld1q_f32(values.data())), 0);
      sum += output[index];
    }
    const float inverse_sum = 1.0F / sum;
    for (int64_t index = 0; index < size; ++index) {
      output[index] *= inverse_sum;
    }
    return;
  }

  const int64_t vectorized_size = size - (size % kVectorWidth);
  auto maximum_vector = vld1q_f32(input);
  int64_t index = kVectorWidth;
  for (; index < vectorized_size; index += kVectorWidth) {
    maximum_vector = vmaxq_f32(maximum_vector, vld1q_f32(input + index));
  }
  const int64_t tail_size = size - index;
  if (tail_size > 0) {
    std::array<float, kVectorWidth> tail{};
    std::copy_n(input + index, tail_size, tail.begin());
    maximum_vector = replace_prefix_lanes(
        maximum_vector,
        vmaxq_f32(maximum_vector, vld1q_f32(tail.data())),
        tail_size);
  }
  const float maximum = reduce_maximum(maximum_vector);
  const auto maximum_vector_broadcast = vdupq_n_f32(maximum);

  index = 0;
  for (; index < vectorized_size; index += kVectorWidth) {
    vst1q_f32(
        output + index,
        sleef_expf4_u10(vsubq_f32(vld1q_f32(input + index), maximum_vector_broadcast)));
  }
  if (tail_size > 0) {
    std::array<float, kVectorWidth> tail{};
    std::array<float, kVectorWidth> result{};
    std::copy_n(input + index, tail_size, tail.begin());
    vst1q_f32(
        result.data(),
        sleef_expf4_u10(vsubq_f32(vld1q_f32(tail.data()), maximum_vector_broadcast)));
    std::copy_n(result.begin(), tail_size, output + index);
  }

  auto sum_vector = vld1q_f32(output);
  index = kVectorWidth;
  for (; index < vectorized_size; index += kVectorWidth) {
    sum_vector = vaddq_f32(sum_vector, vld1q_f32(output + index));
  }
  if (tail_size > 0) {
    std::array<float, kVectorWidth> tail{};
    std::copy_n(output + index, tail_size, tail.begin());
    sum_vector = replace_prefix_lanes(
        sum_vector,
        vaddq_f32(sum_vector, vld1q_f32(tail.data())),
        tail_size);
  }
  const float inverse_sum = 1.0F / reduce_sum(sum_vector);
  const auto inverse_sum_vector = vdupq_n_f32(inverse_sum);

  index = 0;
  for (; index < vectorized_size; index += kVectorWidth) {
    vst1q_f32(output + index, vmulq_f32(vld1q_f32(output + index), inverse_sum_vector));
  }
  if (tail_size > 0) {
    std::array<float, kVectorWidth> tail{};
    std::copy_n(output + index, tail_size, tail.begin());
    const auto normalized = vmulq_f32(vld1q_f32(tail.data()), inverse_sum_vector);
    vst1q_f32(tail.data(), normalized);
    std::copy_n(tail.begin(), tail_size, output + index);
  }
}

void silu_elements(const float* input, float* output, int64_t size) {
  const int64_t vectorized_size = size - (size % kVectorWidth);
  const auto one = vdupq_n_f32(1.0F);
  int64_t index = 0;
  for (; index < vectorized_size; index += kVectorWidth) {
    const auto values = vld1q_f32(input + index);
    const auto denominator = vaddq_f32(one, sleef_expf4_u10(vnegq_f32(values)));
    vst1q_f32(output + index, vdivq_f32(values, denominator));
  }
  const int64_t tail_size = size - index;
  if (tail_size > 0) {
    std::array<float, kVectorWidth> tail{};
    std::copy_n(input + index, tail_size, tail.begin());
    const auto values = vld1q_f32(tail.data());
    vst1q_f32(tail.data(), vdivq_f32(values, vaddq_f32(one, sleef_expf4_u10(vnegq_f32(values)))));
    std::copy_n(tail.begin(), tail_size, output + index);
  }
}
#else
void softmax_row(const float* input, float* output, int64_t size) {
  float maximum = input[0];
  for (int64_t index = 1; index < size; ++index) {
    maximum = std::max(maximum, input[index]);
  }
  float sum = 0.0F;
  for (int64_t index = 0; index < size; ++index) {
    output[index] = std::exp(input[index] - maximum);
    sum += output[index];
  }
  const float inverse_sum = 1.0F / sum;
  for (int64_t index = 0; index < size; ++index) {
    output[index] *= inverse_sum;
  }
}

void silu_elements(const float* input, float* output, int64_t size) {
  for (int64_t index = 0; index < size; ++index) {
    output[index] = input[index] / (1.0F + std::exp(-input[index]));
  }
}
#endif

class CPUReferenceRMSNorm final : public mx::Primitive {
 public:
  CPUReferenceRMSNorm(mx::Stream stream, float eps, int64_t width)
      : Primitive(stream), eps_(eps), width_(width) {}

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
                      eps = eps_, width = width_]() mutable {
      const auto* input_data = input.data<float>();
      const auto* weight_data = weight.data<float>();
      auto* output_data = output.data<float>();
      const size_t rows = input.size() / width;
      std::vector<float> squared(static_cast<size_t>(width));

      for (size_t row = 0; row < rows; ++row) {
        const auto* input_row = input_data + row * width;
        auto* output_row = output_data + row * width;
        for (int64_t column = 0; column < width; ++column) {
          squared[column] = input_row[column] * input_row[column];
        }

        const float mean = row_sum(squared.data(), width) / static_cast<float>(width);
        const float inverse_rms = 1.0F / std::sqrt(mean + eps);
        for (int64_t column = 0; column < width; ++column) {
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
    return eps_ == other_norm.eps_ && width_ == other_norm.width_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs)
      override {
    return {inputs.at(0).shape()};
  }

 private:
  float eps_;
  int64_t width_;
};

class CPUReferenceRoPE final : public mx::Primitive {
 public:
  explicit CPUReferenceRoPE(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& states = inputs.at(0);
    const auto& position_ids = inputs.at(1);
    auto& output = outputs.at(0);
    output.set_data(mx::allocator::malloc(output.nbytes()));

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(states);
    encoder.set_input_array(position_ids);
    encoder.set_output_array(output);
    encoder.dispatch([states = mx::array::unsafe_weak_copy(states),
                      position_ids = mx::array::unsafe_weak_copy(position_ids),
                      output = mx::array::unsafe_weak_copy(output)]() mutable {
      const auto* states_data = states.data<float>();
      const auto* positions_data = position_ids.data<int32_t>();
      auto* output_data = output.data<float>();
      const size_t batch_size = states.shape(0);
      const size_t sequence_length = states.shape(1);
      const size_t head_count = states.shape(2);

      std::array<float, kRopeHalfDimension> timescales{};
      for (int64_t dimension = 0; dimension < kRopeHalfDimension; ++dimension) {
        const float exponent =
            (2.0F / static_cast<float>(kHeadDimension)) * static_cast<float>(dimension);
        timescales[dimension] = ::powf(kRopeBase, exponent);
      }

      for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t token = 0; token < sequence_length; ++token) {
          const int32_t position_id = positions_data[batch * sequence_length + token];
          const float position = static_cast<float>(position_id);
          for (int64_t dimension = 0; dimension < kRopeHalfDimension; ++dimension) {
            const float radians = position / timescales[dimension];
            float sine = ::sinf(radians);
            float cosine = ::cosf(radians);
            if (position_id >= 0 && position_id < kFixedPrefixLength) {
              const size_t table_base =
                  (static_cast<size_t>(position_id) * 2) * kRopeHalfDimension + dimension;
              const auto& corrections = fixed_prefix_rope_corrections();
              sine = apply_raw_bit_correction(sine, corrections[table_base]);
              cosine = apply_raw_bit_correction(cosine, corrections[table_base + kRopeHalfDimension]);
            }
            for (size_t head = 0; head < head_count; ++head) {
              const size_t offset =
                  ((batch * sequence_length + token) * head_count + head) * kHeadDimension;
              const float first = states_data[offset + dimension];
              const float second = states_data[offset + dimension + kRopeHalfDimension];
              const float first_cosine = first * cosine;
              const float second_sine = second * sine;
              const float second_cosine = second * cosine;
              const float first_sine = first * sine;
              output_data[offset + dimension] = first_cosine - second_sine;
              output_data[offset + dimension + kRopeHalfDimension] = second_cosine + first_sine;
            }
          }
        }
      }
    });
  }

  void eval_gpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("CPUReferenceRoPE only supports CPU streams");
  }

  const char* name() const override {
    return "CPUReferenceRoPE";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    (void)static_cast<const CPUReferenceRoPE&>(other);
    return true;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs)
      override {
    return {inputs.at(0).shape()};
  }
};

class CPUReferenceSoftmax final : public mx::Primitive {
 public:
  explicit CPUReferenceSoftmax(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& input = inputs.at(0);
    auto& output = outputs.at(0);
    output.set_data(mx::allocator::malloc(output.nbytes()));

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([input = mx::array::unsafe_weak_copy(input),
                      output = mx::array::unsafe_weak_copy(output)]() mutable {
      const auto* input_data = input.data<float>();
      auto* output_data = output.data<float>();
      const int64_t size = input.shape(-1);
      const size_t rows = input.size() / static_cast<size_t>(size);
      for (size_t row = 0; row < rows; ++row) {
        softmax_row(input_data + row * size, output_data + row * size, size);
      }
    });
  }

  void eval_gpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("CPUReferenceSoftmax only supports CPU streams");
  }

  const char* name() const override {
    return "CPUReferenceSoftmax";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    (void)static_cast<const CPUReferenceSoftmax&>(other);
    return true;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs)
      override {
    return {inputs.at(0).shape()};
  }
};

class CPUReferenceSiLU final : public mx::Primitive {
 public:
  explicit CPUReferenceSiLU(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& input = inputs.at(0);
    auto& output = outputs.at(0);
    output.set_data(mx::allocator::malloc(output.nbytes()));

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([input = mx::array::unsafe_weak_copy(input),
                      output = mx::array::unsafe_weak_copy(output)]() mutable {
      silu_elements(input.data<float>(), output.data<float>(), static_cast<int64_t>(input.size()));
    });
  }

  void eval_gpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("CPUReferenceSiLU only supports CPU streams");
  }

  const char* name() const override {
    return "CPUReferenceSiLU";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    (void)static_cast<const CPUReferenceSiLU&>(other);
    return true;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs)
      override {
    return {inputs.at(0).shape()};
  }
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
  if (input.ndim() == 0 ||
      (input.shape(-1) != kVLMWidth && input.shape(-1) != kExpertWidth)) {
    throw std::invalid_argument("rms_norm requires a final dimension of 720 or 960");
  }
  if (weight.ndim() != 1 || weight.shape(0) != input.shape(-1)) {
    throw std::invalid_argument("rms_norm requires a float32 weight matching the final input dimension");
  }
  if (!std::isfinite(eps) || eps <= 0.0F) {
    throw std::invalid_argument("rms_norm requires a finite positive epsilon");
  }
}

void validate_rope_inputs(
    const mx::array& states,
    const mx::array& position_ids,
    const mx::Stream& stream) {
  if (stream.device.type != mx::Device::cpu) {
    throw std::invalid_argument("reference_rope requires an MLX CPU stream");
  }
  if (states.dtype() != mx::float32 || position_ids.dtype() != mx::int32) {
    throw std::invalid_argument("reference_rope requires float32 states and int32 position_ids");
  }
  if (states.ndim() != 4 || states.shape(-1) != kHeadDimension) {
    throw std::invalid_argument("reference_rope requires states shaped [batch, sequence, heads, 64]");
  }
  if (position_ids.ndim() != 2 || position_ids.shape(0) != states.shape(0) ||
      position_ids.shape(1) != states.shape(1)) {
    throw std::invalid_argument("reference_rope position_ids must match states batch and sequence dimensions");
  }
}

void validate_softmax_input(const mx::array& input, const mx::Stream& stream) {
  if (stream.device.type != mx::Device::cpu) {
    throw std::invalid_argument("reference_softmax requires an MLX CPU stream");
  }
  if (input.dtype() != mx::float32) {
    throw std::invalid_argument("reference_softmax requires float32 input");
  }
  if (input.ndim() == 0 || input.shape(-1) <= 0) {
    throw std::invalid_argument("reference_softmax requires a non-empty final dimension");
  }
}

void validate_silu_input(const mx::array& input, const mx::Stream& stream) {
  if (stream.device.type != mx::Device::cpu) {
    throw std::invalid_argument("reference_silu requires an MLX CPU stream");
  }
  if (input.dtype() != mx::float32) {
    throw std::invalid_argument("reference_silu requires float32 input");
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
      std::make_shared<CPUReferenceRMSNorm>(stream, eps, contiguous_input.shape(-1)),
      {contiguous_input, contiguous_weight});
}

mx::array reference_rope(
    const mx::array& states,
    const mx::array& position_ids,
    mx::StreamOrDevice stream_or_device) {
  const auto stream = mx::to_stream(stream_or_device);
  validate_rope_inputs(states, position_ids, stream);
  const auto contiguous_states = mx::contiguous(states, false, stream);
  const auto contiguous_positions = mx::contiguous(position_ids, false, stream);
  return mx::array(
      contiguous_states.shape(),
      mx::float32,
      std::make_shared<CPUReferenceRoPE>(stream),
      {contiguous_states, contiguous_positions});
}

mx::array reference_softmax(
    const mx::array& input,
    mx::StreamOrDevice stream_or_device) {
  const auto stream = mx::to_stream(stream_or_device);
  validate_softmax_input(input, stream);
  const auto contiguous_input = mx::contiguous(input, false, stream);
  return mx::array(
      contiguous_input.shape(),
      mx::float32,
      std::make_shared<CPUReferenceSoftmax>(stream),
      {contiguous_input});
}

mx::array reference_silu(
    const mx::array& input,
    mx::StreamOrDevice stream_or_device) {
  const auto stream = mx::to_stream(stream_or_device);
  validate_silu_input(input, stream);
  const auto contiguous_input = mx::contiguous(input, false, stream);
  return mx::array(
      contiguous_input.shape(),
      mx::float32,
      std::make_shared<CPUReferenceSiLU>(stream),
      {contiguous_input});
}

} // namespace smolvla_mlx::native
