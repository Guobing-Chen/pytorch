#pragma once

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

struct TensorInfo {
  std::vector<int64_t> dims;
  c10::ScalarType dtype;
};
c10::optional<TensorInfo> getTensorInfo(BufHandle b);

int64_t normalizeAndCheckIndex(int64_t idx, int64_t list_size);

// Convert boolean to integer, if needed.
ExprHandle boolToInteger(const ExprHandle& x);
ExprHandle promoteToDtype(ExprHandle e, ScalarType dt);
void promoteInputs(
    std::vector<ExprHandle>& inputs,
    const int typeConstraints = kAllTypes);
ExprHandle promoteIntegerToDefaultType(const ExprHandle& e);
ExprHandle promoteHalfToFloat(const ExprHandle& e);
ExprHandle demoteOutput(
    const ExprHandle& e,
    const c10::optional<ScalarType> type);

std::vector<ExprHandle> broadcastShapes(
    std::vector<std::vector<ExprHandle>> shapes);
std::vector<ExprHandle> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b);

std::vector<ExprHandle> valueShape(const ArgValue& v);
ExprHandle tensorOrConstant(
    const ArgValue& v,
    const std::vector<ExprHandle>& axes);
ExprHandle scalarOrConstant(const ArgValue& v);
ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes);
ExprHandle constant(const ArgValue& v);

// Infrastructure to provide indirect-indexing related lowering. It helps
// generating the overall loop-nest for the Op and indirect-indexing related
// logic, while leaving Op-specific logic to be defined by Op with injected
// function. E.x.: take a 2D indices and a 3D target (to be indirect indexing),
// and the 2nd dim is the dim to be indirect-indexing for i : size_i
//   for j : size_j
//     x = indices[i, j]
//     for m : size_m
//       for n : size_n
//         innerStmt by Op with innerStmtFunc(idxingTarget[m, x, n], [i, j, m,
//         n])
StmtPtr computeIndirectIndexing(
    size_t posOfIndices, // Position of indices in inputs
    size_t posOfIdxingTarget, // Position of target (to be indirect indexing) in
                              // inputs
    size_t dimOfIndirectIdxing, // Indirect indexing dim of target
    const std::vector<ArgValue>& inputs, // Original lowering input
    const std::vector<ExprHandle>& outputShape, // Original lowering outputShape
    const std::function<StmtPtr(
        const ExprPtr&, // Loaded IdxingTarget
        const std::vector<ExprPtr>& // Loop-nest Indices
        )>& innerStmtFunc // OP-specific customization for inner-most loop body
                          // generation
);

ExprHandle clamp(
    const ExprHandle& cmin,
    const ExprHandle& cmax,
    const ExprHandle& input);

Tensor computeChunk(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeTranspose(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeExpand(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeReshape(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeFlatten(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeCatWoConditionals(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape);
Tensor computeCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeEmbedding(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeEmbeddingExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
