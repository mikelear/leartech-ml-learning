/**
 * Session 6 (C++ Companion): Manual Forward + Backward Pass
 *
 * In Python, loss.backward() is one line. Here you'll step through
 * the actual computation — the chain rule applied to each operation.
 *
 * Build:  g++ -std=c++17 -g -O0 -o loss loss.cpp
 * Debug:  open in CLion, set breakpoints, Debug
 *
 * No libraries needed — just raw maths with raw floats.
 */

#include <cmath>
#include <cstdio>

// ============================================================
// PART 1: A single neuron — all values visible in memory
// ============================================================

int main() {
    printf("=== C++ Manual Forward + Backward Pass ===\n\n");

    // --- The neuron's parameters (like PyTorch's requires_grad=True) ---
    // In Python these are tensors. Here they're just floats.
    float weight0 = 0.5f;   // Weight for feature 0 (eval count)
    float weight1 = -0.3f;  // Weight for feature 1 (import count)
    float bias = 0.1f;

    // --- Input features ---
    float feature0 = 1.0f;  // eval present
    float feature1 = 0.0f;  // no imports

    // --- The truth ---
    float truth = 1.0f;     // Should be FAIL

    // 🔴 BREAKPOINT — Line 30: inspect all variables
    // In CLion's Variables pane, you see raw floats in memory.
    // No tensors, no PyTorch, no abstraction — just numbers.
    // This is what the GPU actually works with.

    // ============================================================
    // PART 2: Forward pass — step by step
    // ============================================================

    printf("--- Forward Pass ---\n");

    // Step 1: weighted sum (the dot product)
    float z1 = weight0 * feature0;  // 0.5 * 1.0 = 0.5
    float z2 = weight1 * feature1;  // -0.3 * 0.0 = 0.0
    float z = z1 + z2 + bias;       // 0.5 + 0.0 + 0.1 = 0.6

    printf("  z1 = w0 * f0 = %.4f * %.4f = %.4f\n", weight0, feature0, z1);
    printf("  z2 = w1 * f1 = %.4f * %.4f = %.4f\n", weight1, feature1, z2);
    printf("  z  = z1 + z2 + bias = %.4f\n", z);

    // 🔴 BREAKPOINT — Line 48: inspect z1, z2, z
    // You can see each multiplication result separately.
    // In Python: torch.dot(weights, features) + bias — all in one call.
    // Here you see the individual operations.

    // Step 2: sigmoid activation
    float prediction = 1.0f / (1.0f + expf(-z));  // sigmoid(0.6) ≈ 0.646

    printf("  prediction = sigmoid(%.4f) = %.4f\n", z, prediction);

    // 🔴 BREAKPOINT — Line 57: inspect prediction
    // sigmoid is just: 1 / (1 + e^(-z))
    // Step into expf() if you want to see the exponential computation.

    // Step 3: loss (binary cross-entropy)
    float loss = -(truth * logf(prediction) + (1.0f - truth) * logf(1.0f - prediction));
    // Since truth=1: loss = -log(prediction) = -log(0.646) ≈ 0.437

    printf("  loss = -log(%.4f) = %.4f\n", prediction, loss);

    // 🔴 BREAKPOINT — Line 64: inspect loss
    // This single number is what we want to minimise.

    // ============================================================
    // PART 3: Backward pass — THE CHAIN RULE
    // ============================================================
    // This is what loss.backward() does inside PyTorch.
    // We compute d(loss)/d(weight) for each weight.
    //
    // Chain rule: d(loss)/d(weight) = d(loss)/d(pred) × d(pred)/d(z) × d(z)/d(weight)
    //
    // We work BACKWARDS from the loss:

    printf("\n--- Backward Pass (Chain Rule) ---\n");

    // Step 1: d(loss)/d(prediction)
    // loss = -log(pred), so d(loss)/d(pred) = -1/pred
    float dloss_dpred = -1.0f / prediction;
    printf("  d(loss)/d(pred) = -1/%.4f = %.4f\n", prediction, dloss_dpred);

    // Step 2: d(prediction)/d(z)
    // prediction = sigmoid(z), derivative of sigmoid = sigmoid(z) * (1 - sigmoid(z))
    float dpred_dz = prediction * (1.0f - prediction);
    printf("  d(pred)/d(z) = %.4f * (1 - %.4f) = %.4f\n", prediction, prediction, dpred_dz);

    // Step 3: d(z)/d(weight0)
    // z = weight0 * feature0 + weight1 * feature1 + bias
    // so d(z)/d(weight0) = feature0
    float dz_dw0 = feature0;  // = 1.0
    float dz_dw1 = feature1;  // = 0.0
    float dz_dbias = 1.0f;    // bias contributes 1.0

    // 🔴 BREAKPOINT — Line 82: inspect dloss_dpred, dpred_dz, dz_dw0, dz_dw1
    // These are the three links in the chain rule.
    // Each one is a simple derivative.
    // The CHAIN = multiply them together.

    // Chain rule: multiply them together
    float grad_w0 = dloss_dpred * dpred_dz * dz_dw0;
    float grad_w1 = dloss_dpred * dpred_dz * dz_dw1;
    float grad_bias = dloss_dpred * dpred_dz * dz_dbias;

    printf("\n  Chain rule (multiply the links):\n");
    printf("  grad_w0 = %.4f × %.4f × %.4f = %.4f\n",
           dloss_dpred, dpred_dz, dz_dw0, grad_w0);
    printf("  grad_w1 = %.4f × %.4f × %.4f = %.4f\n",
           dloss_dpred, dpred_dz, dz_dw1, grad_w1);
    printf("  grad_bias = %.4f × %.4f × %.4f = %.4f\n",
           dloss_dpred, dpred_dz, dz_dbias, grad_bias);

    // 🔴 BREAKPOINT — Line 97: inspect grad_w0, grad_w1, grad_bias
    // These are THE GRADIENTS — exactly what PyTorch computes with loss.backward()
    //
    // grad_w0 is negative → increase weight0 to reduce loss
    // grad_w1 is zero → feature1 was 0, so weight1 doesn't matter for this example
    // grad_bias is negative → increase bias
    //
    // In Python: loss.backward() then weight.grad gives you these same numbers.
    // Here you computed each step by hand.

    // ============================================================
    // PART 4: Update weights (gradient descent)
    // ============================================================

    float learning_rate = 0.1f;

    printf("\n--- Weight Update ---\n");
    printf("  weight0: %.4f - %.1f * %.4f = ", weight0, learning_rate, grad_w0);
    weight0 -= learning_rate * grad_w0;
    printf("%.4f\n", weight0);

    printf("  weight1: %.4f - %.1f * %.4f = ", weight1, learning_rate, grad_w1);
    weight1 -= learning_rate * grad_w1;
    printf("%.4f\n", weight1);

    printf("  bias:    %.4f - %.1f * %.4f = ", bias, learning_rate, grad_bias);
    bias -= learning_rate * grad_bias;
    printf("%.4f\n", bias);

    // New prediction with updated weights
    float z_new = weight0 * feature0 + weight1 * feature1 + bias;
    float pred_new = 1.0f / (1.0f + expf(-z_new));
    float loss_new = -logf(pred_new);

    printf("\n  Old prediction: %.4f  loss: %.4f\n", prediction, loss);
    printf("  New prediction: %.4f  loss: %.4f\n", pred_new, loss_new);
    printf("  Improvement: %.4f\n", loss - loss_new);

    // 🔴 BREAKPOINT — Line 130: inspect before vs after
    // The prediction moved closer to 1.0 (correct).
    // The loss decreased.
    // One step of gradient descent, fully transparent.

    printf("\n=== Summary ===\n");
    printf("What loss.backward() does in PyTorch:\n");
    printf("  1. d(loss)/d(pred) = -1/pred                (how loss depends on prediction)\n");
    printf("  2. d(pred)/d(z) = pred * (1 - pred)         (how sigmoid depends on input)\n");
    printf("  3. d(z)/d(weight) = feature_value            (how weighted sum depends on weight)\n");
    printf("  4. Multiply them: grad = step1 * step2 * step3  (chain rule)\n");
    printf("  5. Store in weight.grad                      (ready for optimizer.step())\n\n");
    printf("That's all backward() does — chain rule on each operation in reverse.\n");
    printf("PyTorch builds a graph of operations during forward, then walks it backwards.\n");

    return 0;
}
