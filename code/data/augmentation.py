import torch


class CutOut1D:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, data):
        """
        Applies the Cut-Out augmentation to the input 1D vector.

        :param data: Input 1D vector (PyTorch tensor)
        :return: Augmented 1D vector
        """
        data_length = len(data)
        mask = torch.ones(data_length, dtype=torch.float32)

        for _ in range(self.n_holes):
            indices = torch.randint(0, data_length, (self.length,))
            mask[indices] = 0.0

        return data * mask


class MixUp1D:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data1, data2):
        """
        Applies the Mix-Up augmentation to a pair of input 1D vectors.

        :param data1: First input 1D vector (PyTorch tensor)
        :param data2: Second input 1D vector (PyTorch tensor)
        :return: Augmented 1D vector
        """
        assert len(data1) == len(data2), "Input vectors must have the same length."

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        return lam * data1 + (1 - lam) * data2


class CutMix1D:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data1, data2):
        """
        Applies the Cut-Mix augmentation to a pair of input 1D vectors.

        :param data1: First input 1D vector (PyTorch tensor)
        :param data2: Second input 1D vector (PyTorch tensor)
        :return: Augmented 1D vector
        """
        assert len(data1) == len(data2), "Input vectors must have the same length."

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        mask = torch.bernoulli(torch.full_like(data1, lam))
        return data1 * mask + data2 * (1 - mask)
