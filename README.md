# Geometric-Regularization-Theory
Implementation of the Jacobian Oracle and Riemannian Gradient Correction for adversarial robustness. The computational price of geometric truth
# Riemannian Adversarial Dynamics

> "Computing the Price of Truth in Geometric Regularization."

[![Status](https://img.shields.io/badge/Status-Research_Preview-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Abstract

Modern adversarial training operates under a "Euclidean Illusion"—assuming that the steepest ascent direction is given by the Euclidean gradient $\nabla_{Euc} \mathcal{L}$. On curved data manifolds, this assumption fails, leading to a misalignment between the attack vector and the data geometry.

This repository implements the **Riemannian Adversary**, a theoretically rigorous algorithm that corrects this flaw. By explicitly computing the **Jacobian Pullback Metric** $G(x)$, we recover the true gradient flow on the manifold.

$$
\text{Update Rule: } \quad x_{t+1} \leftarrow \text{Exp}_{x_t} \left( \alpha \cdot \underbrace{G(x_t)^{-1}}_{\text{Correction}} \cdot \nabla_{Euc} \mathcal{L} \right)
$$

This implementation represents the "Hardcore" regime of the Computational-Statistical Trade-off: paying a high query cost (Oracle Complexity) to achieve a tighter theoretical bound on generalization.

## Theoretical Framework

This work is based on the derivation **"Upgrading a Geometric Regularization Framework"** (Shan YU, 2025).

* **The Gap:** The divergence between the tractable average-case regularizer ($\mathbb{E}_x$) and the required worst-case supremum ($\sup_z$).
* **The Bridge:** A Zero-Sum Game where the Maximizer (this adversary) queries a **Jacobian Oracle** to find the true local Lipschitz constant.
* **The Cost:** The algorithm solves a linear system involving the Metric Tensor at every step ($O(d^3)$), explicitly trading compute for geometric validity.

## Usage

```python
from riemannian_adversary import RiemannianAdversary
import torch

# 1. Define your map (Model)
model = MyNeuralNetwork()

# 2. Initialize the Riemannian Adversary
# We accept the computational cost to find the true worst-case perturbation
adversary = RiemannianAdversary(model, epsilon=0.1, alpha=0.01, num_steps=10)

# 3. Query the Oracle (Generate Attack)
x_clean = torch.randn(16, 10) # Batch of data
x_adv = adversary.perturb(x_clean)
