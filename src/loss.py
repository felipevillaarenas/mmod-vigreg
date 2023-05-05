import torch
import torch.nn.functional as F
import torch.distributed as dist


class VICRegLoss():
    """
    VICReg Loss implementation.

    The loss function uses with three terms:_

    Invariance: the mean square distance between the embedding
    vectors.

    Variance: a hinge loss to maintain the standard deviation (over
    a batch) of each variable of the embedding above a given threshold.
    This term forces the embedding vectors of samples within a batch
    to be different.

    Covariance: a term that attracts the covariances (over a batch)
    between every pair of (centered) embedding variables towards zero.
    This term decorrelates the variables of each embedding and prevents
    an inf contains.
    """
    def __init__(
        self,
        invariance_coeff,
        variance_coeff,
        covariance_coeff,
        batch_size,
        num_nodes,
        devices,
        num_features

    ):
        """
        Args:
            invariance_coeff (float): weight for invariance metric.
            variance_coeff (float): weight for variance metric.
            covariance_coeff (float): weight for covariance metric.
            batch_size (int): mini batch size (batch size per GPU)
            num_nodes (int): Number os nodes
            devices (int): Number of GPUs per node.
            num_features (int): Output dimesion of the expander.
        """
        super().__init__()
        self.invariance_coeff = invariance_coeff
        self.variance_coeff = variance_coeff
        self.covariance_coeff = covariance_coeff
        self.effective_batch_size = batch_size * num_nodes * devices
        self.num_features = num_features

    def off_diagonal(self, x):
        """
        Return a flattened view of the off-diagonal elements of a
        square matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def __call__(self, z1, z2):
        """
        Variance, Invariance, Covariance losses.

        Returns:
           {
               'loss': <loss>,
               'invariance_loss': <invariance_loss>,
               'variance_loss': <variance_loss>,
               'covariance_loss: <covariance_loss>
           }
        """        
        # invariance Loss
        invariance_loss = F.mse_loss(z1, z2)

        # Share operation for Variance and Covariance
        if torch.distributed.is_available():
            z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
            z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)
            
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        #  Variance Loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_z1 = torch.mean(F.relu(1 - std_z1)) / 2
        std_z2 = torch.mean(F.relu(1 - std_z2)) / 2
        variance_loss = std_z1 + std_z2
        
        # Covariance Loss
        cov_z1 = (z1.T @ z1) / (self.effective_batch_size  - 1)
        cov_z2 = (z2.T @ z2) / (self.effective_batch_size  - 1)
        cov_z1 = self.off_diagonal(cov_z1).pow_(2).sum()
        cov_z2 = self.off_diagonal(cov_z2).pow_(2).sum()
        cov_z1 = cov_z1.div(self.num_features)
        cov_z2 = cov_z2.div(self.num_features)
        covariance_loss = cov_z1 + cov_z2

        # Loss function is a weighted average of the loss terms
        loss = (
            self.invariance_coeff * invariance_loss
            + self.variance_coeff * variance_loss
            + self.covariance_coeff * covariance_loss
        )

        return loss


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]