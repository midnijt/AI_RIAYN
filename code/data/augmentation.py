import torch


class CutOut1D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        """
        Applies the Cut-Out augmentation to the input batch of 1D vectors.

        :param data: Input batch of 1D vectors (PyTorch tensor)
        :return: Augmented batch of 1D vectors
        """
        mask = torch.ones_like(data)
        mask = torch.where(torch.rand_like(data) < self.p, torch.zeros_like(data), mask)
        return data * mask


class MixUp1D:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data):
        """
        Applies the Mix-Up augmentation to an input batch of 1D vectors.

        :param data: Input batch of 1D vectors (PyTorch tensor)
        :return: Augmented batch of 1D vectors
        """
        data2 = data[torch.randperm(data.size(0))]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        return lam * data + (1 - lam) * data2


class CutMix1D:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data):
        """
        Applies the Cut-Mix augmentation to a pair of input 1D vectors.

        :param data1: First input 1D vector (PyTorch tensor)
        :param data2: Second input 1D vector (PyTorch tensor)
        :return: Augmented 1D vector
        """

        data2 = data[torch.randperm(data.size(0))]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        mask = torch.bernoulli(torch.full_like(data, lam.item()))
        return data * mask + data2 * (1 - mask)
