import torch
import torch.nn.functional as F


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
    def __init__(self):
        super().__init__()

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

        embedding_dims = z1.shape
        batch_size, num_features = embedding_dims[0], embedding_dims[1]

        # invariance Loss
        invariance_loss = F.mse_loss(z1, z2)

        # Share operation for Variance and Covariance
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # Variance Loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_z1 = torch.mean(F.relu(1 - std_z1)) / 2
        std_z2 = torch.mean(F.relu(1 - std_z2)) / 2
        variance_loss = std_z1 + std_z2

        # Covariance Loss
        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        cov_z1 = self.off_diagonal(cov_z1).pow_(2).sum()
        cov_z2 = self.off_diagonal(cov_z2).pow_(2).sum()
        cov_z1 = cov_z1.div(num_features)
        cov_z2 = cov_z2.div(num_features)
        covariance_loss = cov_z1 + cov_z2

        # Loss function is a weighted average of the loss terms
        loss = (
            self.args.invariance_coeff * invariance_loss
            + self.args.variance_coeff * variance_loss
            + self.args.covariance_coeff * covariance_loss
        )

        return {
            'loss': loss,
            'invariance_loss': invariance_loss,
            'variance_loss': variance_loss,
            'covariance_loss': covariance_loss
        }




