import torch
import torch.linalg as la

class RiemannianAdversary:
    """
    The 'Hardcore' Implementation of the Jacobian Oracle Query.
    Unlike standard PGD which assumes Euclidean geometry (G(x) = I),
    this adversary respects the intrinsic curvature of the data manifold.
    Theory Reference: Shan YU, "A Derivation on Upgrading a Geometric Regularization Framework"
    """
    def __init__(self, model, epsilon, alpha, num_steps):
        self.model = model
        self.epsilon = epsilon  # Constraint radius
        self.alpha = alpha      # Step size
        self.m = num_steps      # Query complexity (number of steps)

    def compute_pullback_metric(self, x):
        J = torch.autograd.functional.jacobian(self.model.forward_features, x)
        G = torch.bmm(J.transpose(1, 2), J)
        G += 1e-6 * torch.eye(G.shape[-1], device=x.device)
        return G

    def query_oracle(self, x_seed):
        x_adv = x_seed.clone().detach().requires_grad_(True)
        for t in range(self.m):
            current_J = torch.autograd.functional.jacobian(self.model.forward_features, x_adv)
            obj = torch.linalg.norm(current_J, ord=2, dim=(1,2))
            grad_euc = torch.autograd.grad(obj.sum(), x_adv)[0]
            G = self.compute_pullback_metric(x_adv)
            grad_riem = torch.linalg.solve(G, grad_euc.unsqueeze(-1)).squeeze(-1)
            x_adv.data = x_adv.data + self.alpha * grad_riem
            delta = x_adv.data - x_seed.data
            delta = delta.renorm(p=2, dim=0, maxnorm=self.epsilon)
            x_adv.data = x_seed.data + delta
        return x_adv.detach()

    def evaluate_adversary(self, x_seed, y_true):
        x_adv = self.query_oracle(x_seed)
        y_pred = self.model(x_adv)
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean().item()
        return {"loss": loss.item(), "accuracy": accuracy}

    def visualize_adversary(self, x_seed, x_adv):
        import matplotlib.pyplot as plt
        if x_seed.dim() == 4:
            x_seed = x_seed[0].permute(1, 2, 0).cpu().numpy()
            x_adv = x_adv[0].permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Input")
        plt.imshow(x_seed)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Adversarial Example")
        plt.imshow(x_adv)
        plt.axis("off")
        plt.show()
