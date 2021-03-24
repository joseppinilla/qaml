import torch
import itertools
import numpy as np

class BAS(torch.utils.data.Dataset):
    """ Bars And Stripes (BAS) Synthetic Dataset

    Args:
        rows (int):

        cols (int):

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

        Example:

            >>> # Using Class method
            >>> import matplotlib.pyplot as plt
            >>> img,target = BAS.generate_bars_and_stripes(4,5)
            >>> fig,axs = plt.subplots(2,len(img)//2)
            >>> for i,ax in enumerate(axs.flat):
            >>>     ms = ax.matshow(img[i],vmin=0, vmax=1)
            >>>     ax.axis('off')
    """

    def __init__(self, rows, cols, transform=None, target_transform=None):
        self.data, self.targets = self.generate_bars_and_stripes(rows,cols)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @classmethod
    def generate_bars_and_stripes(cls, rows, cols):
        """ Generate the full dataset of rows*cols Bars And Stripes (BAS).
                Args:

                Returns:

            Implementation based on DDQCL project for benchmarking generative models
            with shallow gate-level quantum circuits.

            [1] https://github.com/uchukwu/quantopo
            [2] https://www.nature.com/articles/s41534-019-0157-8
        """
        bars = []
        for h in itertools.product([0., 1.], repeat=cols):
            pattern = np.repeat([h], rows, 0)
            bars.append(pattern)

        stripes = []
        for h in itertools.product([0., 1.], repeat=rows):
            pattern = np.repeat([h], cols, 1)
            stripes.append(pattern.reshape(rows,cols))

        data = np.concatenate((bars[:-1], # ignore all ones
                               stripes[1:]), # ignore all zeros
                               axis=0)

        # Create labels synthetically
        labels  = [(1.,1.)] # All zeros
        labels += [(0.,1.)]*(2**cols-2) # Bars
        labels += [(1.,0.)]*(2**rows-2) # Stripes
        labels += [(1.,1.)] # All ones

        return np.asarray(data,dtype=float), np.asarray(labels,dtype=float)


img,target = BAS.generate_bars_and_stripes(7,7)
len(img)
