import itertools
import numpy as np


__all__ = ['generate_bars_and_stripes']

rows = 3
cols = 3
label = False


def bars_and_stripes(rows, cols, label=False):
    """ Generate the full dataset of rows*cols Bars And Stripes (BAS).
            Args:

            Returns:

        Implementation based on DDQCL project for benchmarking generative models
        with shallow gate-level quantum circuits.

        [1] https://github.com/uchukwu/quantopo
        [2] https://www.nature.com/articles/s41534-019-0157-8
    """
    bars = []
    for h in itertools.product([0, 1], repeat=cols):
        pattern = np.repeat([h], rows, 0)
        bars.append(pattern.ravel())

    stripes = []
    for h in itertools.product([0, 1], repeat=rows):
        pattern = np.repeat([h], cols, 1)
        stripes.append(pattern.ravel())

    data = np.concatenate((np.asarray(bars[:-1]), np.asarray(stripes[1:])), axis=0)

    if not label:
        return np.asarray(data)
    else:
        # All zeros, bars,
        labels = [(1,1)] + [(0,1)]*(2**cols-2) + [(1,0)]*(2**rows-2) + [(1,1)]
        return np.asarray(data), np.asarray(labels)


def sample_bars_and_stripes(samples, width,
                                height=None, label=False,
                                batch_size=None,
                                seed=None, noise=0,
                                verbose=False):
    """ Creates a dataset containing samples showing bars or stripes.
        Args:
            samples: int
                Number of samples.
            width: int
                Width of Bars/Stripes.
            **params: (optional)
                height: int (default=width)
                    Height of Bars/Stripes.
                label: bool (default=False)
                    Include label of data as binary values at the end.
                seed: int or None (default=None)
                    Seed to initial numpy random number generator.

        Returns:
            data: numpy array.
            if batch_size is not None, shape = [samples, width*size + 2*label]
            else shape = [batches, batch_size, ]
    """

    rng = np.random.RandomState(seed)

    if height is None: height = width

    size = width * height
    if not label:
        data = np.zeros((samples, size))
    else:
        data = np.zeros((samples, width, height))
        labels = np.zeros((samples, 2))

    for i in range(samples):
        if rng.rand() > 0.5: # Bars
            pattern = rng.randint(low=0, high=2, size=(1, width))
            values = np.repeat(pattern, height, axis=0)
            bar_stripe = [1.,0.]
        else: # Stripes
            pattern = rng.randint(low=0, high=2, size=(height, 1))
            values = np.repeat(pattern, width, axis=1)
            bar_stripe = [0.,1.]

        if noise:
            values = np.bitwise_xor(values,rng.binomial(1,noise,values.shape))

        if not label:
            # data[i, :] = values.reshape(size)
            data[i, :] = values
        else:
            # data[i, :] = values.reshape(size)
            data[i, :] = values
            labels[i, :] = np.array(bar_stripe)

    if not label:
        if batch_size is not None:
            return [data[i:i+batch_size] for i in range(0,len(data),batch_size)]
        return data
    else:
        if batch_size is not None:
            return [[data[i:i+batch_size],labels[i:i+batch_size]]
                    for i in range(0,len(data),batch_size)]
        return data, labels
