#pragma once

namespace tie {

enum class OpType {
  kUnknown,
  kFlatten,
  kGemm,
  kRelu,
  kAdd,
  kConv,
  kMaxPool,
};

}  // namespace tie
