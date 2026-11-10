import torch
import torch.nn as nn

class SparseLoss(nn.Module):
    def __init__(self, base_loss_func, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.base_loss = base_loss_func # eg torch.nn.MSELoss()

    def forward(self, preds, targets, limitingks):
        prediction_loss = self.base_loss(preds, targets)
        sparsity_loss = torch.mean(limitingks)
        total_loss = prediction_loss + self.alpha * sparsity_loss
        return total_loss



# # Example Usage
# model = AGCRN()
# criterion = SparseLoss
# preds, limitingks = model(inputs)
# loss = criterion(preds, targets, limitingks)