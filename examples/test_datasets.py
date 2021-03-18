import datasets
import matplotlib.pyplot as plt

n, m = 3, 3

# Without Labels
bas = datasets.bars_and_stripes(n, m, label=False)

n_points, n_qubits = bas.shape

fig, ax = plt.subplots(1, bas.shape[0], figsize=(9, 1))
for i in range(bas.shape[0]):
    ax[i].matshow(bas[i].reshape(n, m), vmin=-1, vmax=1)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

# With Labels
bas, labels = datasets.bars_and_stripes(n, m, label=True)

fig, ax = plt.subplots(1, bas.shape[0], figsize=(9, 1))
for i in range(bas.shape[0]):
    ax[i].matshow(bas[i].reshape(n, m), vmin=-1, vmax=1)
    ax[i].set_title(str(labels[i]))
    ax[i].set_xticks([])
    ax[i].set_yticks([])


# Sample
