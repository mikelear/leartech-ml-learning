/**
 * Session 6b: Debugging with the REAL PyTorch C++ Library (libtorch)
 *
 * This is the same library that Python's PyTorch calls under the hood.
 * Same torch::Tensor, same autograd, same backward() — but in C++
 * where you can step into the actual implementation.
 *
 * Open in CLion. Set breakpoints. Debug.
 * You can step INTO torch::sigmoid, tensor.backward(), etc.
 */

#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=== libtorch Debug Session ===" << std::endl;
    std::cout << "PyTorch C++ version: " << TORCH_VERSION << std::endl;
    std::cout << std::endl;

    // ============================================================
    // PART 1: Same tensors as Python — but in C++
    // ============================================================

    // These are the SAME tensor objects that Python uses
    auto features = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    auto weights = torch::randn({8}, torch::requires_grad());
    auto bias = torch::zeros({1}, torch::requires_grad());

    std::cout << "Features: " << features << std::endl;
    std::cout << "Weights: " << weights << std::endl;
    std::cout << "Bias: " << bias << std::endl;

    // 🔴 BREAKPOINT — Line 32
    // Inspect 'weights' in the Variables pane.
    // Expand it — you'll see the internal TensorImpl structure:
    //   sizes_, strides_, storage_ (the actual memory)
    // This is what a tensor REALLY is inside PyTorch.

    // ============================================================
    // PART 2: Forward pass — same as Python but debuggable
    // ============================================================

    // torch::dot is the same as Python's torch.dot
    auto z = torch::dot(weights, features) + bias;

    // torch::sigmoid — you can STEP INTO this function
    // It goes: sigmoid.cpp → TensorIterator → vectorized kernel
    auto prediction = torch::sigmoid(z);

    std::cout << "\nForward pass:" << std::endl;
    std::cout << "  z (raw): " << z.item<float>() << std::endl;
    std::cout << "  prediction: " << prediction.item<float>() << std::endl;

    // 🔴 BREAKPOINT — Line 49
    // Inspect 'z' — look at z.grad_fn()
    // The grad_fn is the COMPUTATION GRAPH node.
    // It records: "z was created by AddBackward0(DotBackward0(...), ...)"
    // This is how backward() knows what operations to reverse.

    // ============================================================
    // PART 3: Loss
    // ============================================================

    auto truth = torch::tensor({1.0f});
    auto loss = -torch::log(prediction);

    std::cout << "\nLoss: " << loss.item<float>() << std::endl;

    // 🔴 BREAKPOINT — Line 62
    // Inspect 'loss' — its grad_fn chain shows the full computation:
    //   NegBackward0 → LogBackward0 → SigmoidBackward0 → AddBackward0 → DotBackward0
    // This IS the computation graph that backward() walks.

    // ============================================================
    // PART 4: Backward pass — THE REAL autograd
    // ============================================================

    // This is the call that's a black box in Python.
    // In CLion, you can STEP INTO this function.
    // It calls: torch::autograd::Engine::execute()
    //   → walks the grad_fn chain backwards
    //   → calls each node's apply() method (the chain rule)
    //   → accumulates gradients into .grad
    loss.backward();

    std::cout << "\nGradients (computed by autograd):" << std::endl;
    std::cout << "  weights.grad: " << weights.grad() << std::endl;
    std::cout << "  bias.grad: " << bias.grad() << std::endl;

    // 🔴 BREAKPOINT — Line 81
    // weights.grad() now has values — same as Python's weight.grad
    // These were computed by the autograd engine walking the graph backwards.
    //
    // If you stepped INTO loss.backward(), you would have seen:
    //   1. NegBackward0::apply()    → gradient of negation
    //   2. LogBackward0::apply()    → gradient of log = 1/x
    //   3. SigmoidBackward0::apply() → gradient of sigmoid = sig * (1-sig)
    //   4. AddBackward0::apply()    → gradient passes through
    //   5. DotBackward0::apply()    → gradient = input (features)
    //
    // Each apply() is one link in the chain rule.

    // ============================================================
    // PART 5: Weight update — gradient descent
    // ============================================================

    float lr = 0.1f;
    {
        torch::NoGradGuard no_grad;  // Same as Python's "with torch.no_grad():"
        weights -= lr * weights.grad();
        bias -= lr * bias.grad();
    }

    // New prediction
    auto new_z = torch::dot(weights, features) + bias;
    auto new_pred = torch::sigmoid(new_z);

    std::cout << "\nAfter one update:" << std::endl;
    std::cout << "  Old prediction: " << prediction.item<float>() << std::endl;
    std::cout << "  New prediction: " << new_pred.item<float>() << std::endl;
    std::cout << "  (Should be closer to 1.0)" << std::endl;

    // 🔴 BREAKPOINT — Line 107
    // Same result as Python — because it IS the same library.
    // The only difference: in C++ you can step into torch::sigmoid,
    // loss.backward(), and see the actual autograd engine running.

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "This is the SAME code that runs when you write Python:" << std::endl;
    std::cout << "  torch.tensor(...)  → calls at::tensor() in C++" << std::endl;
    std::cout << "  torch.sigmoid(x)   → calls at::sigmoid() in C++" << std::endl;
    std::cout << "  loss.backward()    → calls autograd::Engine::execute() in C++" << std::endl;
    std::cout << "  optimizer.step()   → calls SGD::step() in C++" << std::endl;
    std::cout << "\nPython is just the UI. This C++ is the engine." << std::endl;

    return 0;
}
