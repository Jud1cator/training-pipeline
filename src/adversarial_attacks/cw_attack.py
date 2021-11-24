import torch


class CWAttack:
    def __init__(
            self,
            c: float = 1e-4,
            kappa: float = 0,
            max_iter: int = 50,
            learning_rate: float = 1e-2,
            patience: int = 5
    ):
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.patience = patience

    def generate_adversarial_samples(self, model, input, labels):
        def f(x):
            outputs = model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels]
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.byte())
            return torch.clamp(j - i, min=-self.kappa), outputs

        delta = torch.zeros_like(input, requires_grad=True)

        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)

        prev = 1e10
        best_delta = torch.clone(delta)

        n_iter_no_improve = 0

        for i in range(self.max_iter):
            loss1 = 5 * torch.max(delta)
            loss2, output = f(torch.clip(input + delta, 0, 1))
            loss2 = torch.sum(self.c * loss2)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss > prev:
                n_iter_no_improve += 1
                if n_iter_no_improve == self.patience:
                    return best_delta.detach(), output
            else:
                n_iter_no_improve = 0
                prev = loss
                best_delta = torch.clone(delta)
        return delta.detach(), output
