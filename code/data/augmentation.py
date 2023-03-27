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
        data_length = data.size(0)
        mask = torch.ones(data_length, dtype=torch.float32)

        for _ in range(self.n_holes):
            indices = torch.randint(0, data_length - self.length + 1, (1,))
            mask[indices : indices + self.length] = 0.0

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
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (data.size(0), 1, 1)
        )
        return lam * data + (1 - lam) * data2


class CutMix1D:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data):
        """
        Applies the Cut-Mix augmentation to an input batch of 1D vectors.

        :param data: Input batch of 1D vectors (PyTorch tensor)
        :return: Augmented batch of 1D vectors
        """
        data2 = data[torch.randperm(data.size(0))]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (data.size(0), 1, 1)
        )
        mask = torch.bernoulli(torch.full((data.size(0), data.size(1), 1), lam))
        return data * mask + data2 * (1 - mask)
