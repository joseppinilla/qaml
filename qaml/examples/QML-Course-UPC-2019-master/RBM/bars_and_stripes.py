import itertools
import numpy as np
import matplotlib.pyplot as plt

def bars_and_stripes(rows, cols):
    """ Implementation from the DQCL project for generative modeling
        with shallow quantum circuits.

        [1] https://github.com/uchukwu/quantopo
        [2] https://www.nature.com/articles/s41534-019-0157-8
    """
    data = []

    for h in itertools.product([0, 1], repeat=cols):
        pic = np.repeat([h], rows, 0)
        data.append(pic.ravel().tolist())

    for h in itertools.product([0, 1], repeat=rows):
        pic = np.repeat([h], cols, 1)
        data.append(pic.ravel().tolist())

    data = np.unique(np.asarray(data), axis=0)

    return data


def __main__():

    n, m = 3, 3

    bas = bars_and_stripes(n, m)

    n_points, n_qubits = bas.shape

    fig, ax = plt.subplots(1, bas.shape[0], figsize=(9, 1))
    for i in range(bas.shape[0]):
        ax[i].matshow(bas[i].reshape(n, m), vmin=-1, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])


    np.save('bars_and_stripes', bas)
