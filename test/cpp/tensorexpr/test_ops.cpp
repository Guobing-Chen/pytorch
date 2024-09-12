#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

using Tensors = std::vector<Tensor>;
using Args = std::vector<CodeGen::BufferArg>;
std::unique_ptr<SimpleIREvaluator> compile(
    const Args& inputs,
    const Tensors& outputs) {
  LoopNest nest({outputs});
  nest.prepareForCodegen();
  nest.simplify();
  auto join = inputs;
  join.insert(join.end(), outputs.begin(), outputs.end());
  return std::make_unique<SimpleIREvaluator>(nest.root_stmt(), join);
}

TEST(Ops, Sum) {
  constexpr int M = 8;
  constexpr int N = 16;
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};
  std::vector<std::vector<ExprHandle>> outputShapes = {{N}, {M}, {}};
  for (unsigned idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    BufHandle a("a", {M, N}, kFloat);
    std::vector<ExprHandle> outStrides =
        c10::fmap<ExprHandle>(make_contiguous_strides(outShape));
    Tensor b = computeSum(
        {a, dims, false}, outShape, outStrides, c10::kFloat, at::kCPU);
    auto cg = compile({a}, {b});

    auto at = at::arange(M * N, at::kFloat).view({M, N});
    auto ref = at::sum(at, dims);
    auto bt = at::empty_like(ref);

    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    ASSERT_TRUE(at::allclose(bt, ref));
  }
}

TEST(Ops, ChannelsLastSum) {
  constexpr int A = 2;
  constexpr int B = 3;
  constexpr int C = 4;
  constexpr int D = 5;
  constexpr int E = 6;
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};

  std::vector<std::vector<ExprHandle>> outputShapes = {
      {B, C, D, E}, {A, C, D, E}, {C, D, E}};
  for (unsigned idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    BufHandle a("a", {A, B, C, D, E}, kFloat);
    std::vector<ExprHandle> outStrides =
        c10::fmap<ExprHandle>(make_channels_last_strides(outShape));
    Tensor b = computeSum(
        {a, dims, false}, outShape, outStrides, c10::kFloat, at::kCPU);
    auto cg = compile({a}, {b});

    auto at = at::arange(A * B * C * D * E, at::kFloat).view({A, B, C, D, E});
    auto ref = at::sum(at, dims);
    auto bt = at::empty_like(ref);

    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    ASSERT_TRUE(at::allclose(bt, ref));
  }
}

TEST(Ops, LinearWithBias) {
  const auto graph_string = R"IR(
    graph(%x : Float(1, 16, strides=[16, 1], device=cpu),
          %w : Float(8, 16, strides=[16, 1], device=cpu),
          %b : Float(8, strides=[1], device=cpu)):
      %1 : Float(1, 8, strides=[8, 1], device=cpu) = aten::linear(%x, %w, %b)
      return (%1))IR";
  auto graph = std::make_shared<torch::jit::Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 16}, at::TensorOptions(c10::kCPU).dtype(at::kFloat));
  auto w = at::rand({8, 16}, at::TensorOptions(c10::kCPU).dtype(at::kFloat));
  auto b = at::rand({8}, at::TensorOptions(c10::kCPU).dtype(at::kFloat));
  auto y_expected = at::linear(x, w, b);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x, w, b};

  std::vector<c10::IValue> stack = at::fmap<c10::IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();

  bool check = at::allclose(y_expected, y);
  if(!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST(Ops, LinearWithoutBias) {
  const auto graph_string = R"IR(
    graph(%x : Float(1, 16, strides=[16, 1], device=cpu),
          %w : Float(8, 16, strides=[16, 1], device=cpu)):
      %1 : NoneType = prim::Constant()
      %2 : Float(1, 8, strides=[8, 1], device=cpu) = aten::linear(%x, %w, %1)
      return (%2))IR";
  auto graph = std::make_shared<torch::jit::Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 16}, at::TensorOptions(c10::kCPU).dtype(at::kFloat));
  auto w = at::rand({8, 16}, at::TensorOptions(c10::kCPU).dtype(at::kFloat));
  auto y_expected = at::linear(x, w);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x, w};

  std::vector<c10::IValue> stack = at::fmap<c10::IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();

  bool check = at::allclose(y_expected, y);
  if(!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}