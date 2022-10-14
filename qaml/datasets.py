import os
import torch
import requests
import itertools
import torchvision

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as torch_transforms

from PIL import Image

class PhaseState(torch.utils.data.Dataset):
    """ Synthetic data points that represent the phase of a 1D state at N
        spatial points. The '0' phase is accumulated to the left, and the '1'
        phase is accumulated to the right with at most 1 boundary. Used in [1].
        [1] Srivastava, S., & Sundararaghavan, V. (2020). Machine learning in
            quantum computers via general Boltzmann Machines: Generative and
            Discriminative training through annealing.
            https://doi.org/10.48550/arxiv.2002.00792

        Example:
        >>> PhaseState.generate(3)
        >>> (array([[1., 1., 1.],
                    [0., 1., 1.],
                    [0., 0., 1.],
                    [0., 0., 0.]]),
             array([0, 1, 2, 3]))
        >>> PhaseState.generate(3,labeled=True)
        >>> (array([[1., 1., 1.],
                    [0., 1., 1.],
                    [0., 0., 1.],
                    [0., 0., 0.],
                    [0., 1., 0.],
                    [1., 1., 0.],
                    [1., 0., 1.],
                    [1., 0., 0.]]),
             array([0., 0., 0., 0., 1., 1., 1., 1.]))

    Args:
        N (int): Number of spatial points.

        labeled (bool): If labeled, 1 boundary phase states have label 1., and
            random states have 0. If false, labels simply index each data point
            [0,N].

    """

    def __init__(self, N, labeled=False, transform=None, target_transform=None):
        self.transform = transform
        self.labeled = labeled
        self.target_transform = target_transform
        self.data,self.targets = self.generate(N,labeled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def find(self, item):
        if isinstance(item,torch.Tensor):
            item = item.numpy()

        if isinstance(item,np.ndarray):
            iter = (i for i,d in enumerate(self.data) if np.array_equiv(d,item))
        else:
            raise RuntimeError("Item isn't `torch.Tensor` or `numpy.ndrray`")

        return next(iter,None)

    def __contains__(self, item):
        return self.find(item) is not None

    @classmethod
    def generate(cls, N, labeled=False, random_seed=42):
        rng = np.random.default_rng(random_seed)
        if not labeled:
            return np.triu(np.ones((N+1,N),dtype='float64')), np.arange(N+1)
        else:
            phase = np.triu(np.ones((N+1,N)))

            # Random data points
            int_set = set(phase.dot(2**np.arange(phase[0].size)[::-1]))
            R = min(N*3-1, ((2**N)//2))
            labels = np.asarray(np.concatenate((np.zeros(N+1),np.ones(R))),
                                dtype='float64')

            while i:=0 < R:
                rand_set = []
                while R>0:
                    rand = rng.integers(1,2**N-1)
                    if rand in int_set: continue
                    if rand in rand_set: continue
                    else:
                        rand_set.append(rand)
                        R-=1
            def int2binarray(x,width):
                return np.asarray(list(np.binary_repr(x).zfill(width)),int)

            random = [int2binarray(x,N) for x in rand_set]
            data = np.asarray(np.concatenate((phase,random)),dtype='float64')

            return data, labels

    def score(self, samples):
        total_samples = len(samples)
        total_patterns = len(self)

        sampled_patterns = [i for i in map(self.find,samples) if i is not None]
        if not sampled_patterns: return 0.0,0.0,0.0

        precision = len(sampled_patterns)/total_samples
        recall = len(set(sampled_patterns))/total_patterns
        score = 2.0*precision*recall/(precision+recall)

        return precision, recall, score


class BAS(torch.utils.data.Dataset):
    """ Bars And Stripes (BAS) Synthetic Dataset

    Args:
        rows (int): Number of rows in the BAS image.

        cols (int): Number of columns in the BAS image.

        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version.

        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

    Example:

        >>> # Using Class method
        >>> import matplotlib.pyplot as plt
        >>> img,target = BAS.generate_bars_and_stripes(4,5)
        >>> fig,axs = plt.subplots(2,len(img)//2)
        >>> for i,ax in enumerate(axs.flat):
        >>>     ms = ax.matshow(img[i],vmin=0, vmax=1)
        >>>     ax.axis('off')
    """
    classes = ['0 - All Zeros', '1 - Bars', '2 - Stripes', '3 - All Ones']

    def __init__(self, rows, cols, transform=None, target_transform=None):
        self.data, self.targets = self.generate(rows,cols)
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

    def find(self, item):
        if isinstance(item,torch.Tensor):
            item = item.numpy()

        if isinstance(item,np.ndarray):
            iter = (i for i,d in enumerate(self.data) if np.array_equiv(d,item))
        else:
            raise RuntimeError("Item isn't `torch.Tensor` or `numpy.ndrray`")

        return next(iter,None)

    def __contains__(self, item):
        return self.find(item) is not None

    def score(self, samples):
        """ Given a set of samples, compute the qBAS[1] sampling score:
                qBAS = 2pr/(p+r)
            p: precision or number of correct samples over total samples
            r: recall or number of sampled patterns over total patterns

        Args:
            samples (list or numpy.ndarray): An iterable of numpy.ndarray values
                to be compared to the original data in the dataset.

        Returns:
            precision (float): number of correct samples over total samples

            recall (float): number of sampled patterns over total patterns

            score (float): qBAS score as defined above

        [1] Benedetti, M., et al. A generative modeling approach for
        benchmarking and training shallow quantum circuits. (2019).
        https://doi.org/10.1038/s41534-019-0157-8
        """
        total_samples = len(samples)
        total_patterns = len(self)

        sampled_patterns = [i for i in map(self.find,samples) if i is not None]
        if not sampled_patterns: return 0.0,0.0,0.0

        precision = len(sampled_patterns)/total_samples
        recall = len(set(sampled_patterns))/total_patterns
        score = 2.0*precision*recall/(precision+recall)

        return precision, recall, score

    @classmethod
    def generate(cls, rows, cols):
        """ Generate the full dataset of rows*cols Bars And Stripes (BAS).
        Args:
            cols (int): number of columns in the generated images

            rows (int): number of rows in the generated images

        Returns:
            data (numpy.ndarray): Array of (rows,cols) images of BAS dataset

            targets (numpy.ndarray): Array of labels for data. Where the empty
                (all zeros), bars (vertical lines), stripes (horizontal lines),
                and full (all ones) images belong to different classes.

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

        data = torch.DoubleTensor(np.concatenate((bars[:-1], #ignore all one
                                                  stripes[1:]), #ignore all zero
                                                  axis=0))

        # Create labels synthetically
        labels  = [0] # All zero
        labels += [1]*(2**cols-2) # Bars
        labels += [2]*(2**rows-2) # Stripes
        labels += [3] # All one
        return data, torch.LongTensor(labels)

class OptDigits(torchvision.datasets.vision.VisionDataset):
    """ Based on the MNIST Dataset implementation, but enough differences to not
        make it a subclass.
    """
    mirrors = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
    ]

    training_file = ("optdigits.tra", "268ce7771f3f15afbc54402478b1d454")
    test_file = ("optdigits.tes", "a0339c30a8a5312a1b6f9e5c719dcce5")

    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True):
        super(OptDigits, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        if download:
            self.download()
        self.data, self.targets = self._load_data()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_data(self):
        filename, _ = self.training_file if self.train else self.test_file
        fpath = os.path.join(self.raw_folder, filename)
        dataset = torch.from_numpy(np.genfromtxt(fpath,delimiter=',',dtype='uint8'))
        data,targets = torch.split(dataset,[64,1],1)
        return data.view(-1,8,8)*15, targets.squeeze().to(torch.int64)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        filename, md5 = self.training_file if self.train else self.test_file
        fpath = os.path.join(self.raw_folder, filename)

        if torchvision.datasets.utils.check_integrity(fpath,md5):
            print("Using downloaded and verified file " + fpath)
            return

        for mirror in self.mirrors:
            try:
                print('Downloading ' + mirror+filename + ' to ' + fpath)
                with open(fpath, 'wb') as f:
                    response = requests.get(mirror+filename)
                    f.write(response.content)
            except:
                print("Failed download.")
                continue
            if not torchvision.datasets.utils.check_integrity(fpath,md5):
                raise RuntimeError("File not found or corrupted.")
            break

def _embed_labels(dataset, axis=0, reversed=False, append=True,
                  encoding='binary', scale=1):
    """ Modifies the images of a dataset to include a binary or one hot label.
    Args:
        dataset (torch.utils.data.Dataset): Torch dataset object

        axis (int): determines the direction of the embedded labels (0 or 1)

        append (bool): wether the label is at the first or last line of `axis'

        enconding {'binary','one_hot'}: representation of label

    """

    data,targets = dataset.data,dataset.targets

    labels = torch.LongTensor(targets.flatten())
    if encoding=='one_hot':
        N = len(dataset.classes)
        labelset = torch.unique(dataset.targets)
        labelidx = torch.searchsorted(labelset,labels)
        labels = F.one_hot(labelidx,N)*scale
    elif encoding=='binary':
        N = int(np.log2(len(dataset.classes)-1)+1)
        bin_array = np.arange(N,dtype=int)
        binary_repr = lambda x: x[:,None] & (1 << bin_array) > 0
        labels = binary_repr(labels)*scale
    else:
        raise ValueError(f"Invalid encoding: {encoding}")

    M = -1 if append else 0
    if axis==0:
        if reversed:
            data[:,M,N:] = labels
        else:
            data[:,M,:-N] = labels
    elif axis==1:
        if reversed:
            data[:,-N:,M] = labels
        else:
            data[:,:N,M] = labels
    else:
        raise ValueError(f"Invalid axis: {axis}")

def _subset_classes(dataset, subclasses):
    idx = [i for i,y in enumerate(dataset.targets) if y in subclasses]
    dataset.classes = [dataset.classes[i] for i in subclasses]
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

class ToBinaryTensor:
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, pic):
        """
        Args:
            pic (Image or numpy.ndarray): Image to be converted to bool Tensor

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic,Image.Image):
            pic = torch_transforms.functional.to_tensor(pic)

        return torch.round(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToSpinTensor:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    def __call__(self, pic):
        """
        Args:
            pic (Image or numpy.ndarray): Image to be converted to spin Tensor

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic,Image.Image):
            pic = torch_transforms.functional.to_tensor(pic)
        return (2.0*torch.round(pic)-1.0)

    def __repr__(self):
        return self.__class__.__name__ + '()'
