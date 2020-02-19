import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def main():
    # test_idx = [1, 2, 3, 10, 22, 40, 59, 63, 87, 94, 78]

    data = fetch_olivetti_faces()
    data = data.images.reshape((len(data.images), -1))

    # Center the data
    centered_data = data - np.mean(data, axis=0)
    n_samples = centered_data.shape[0]
    centered_cov = np.matmul(centered_data.T, centered_data) / (n_samples - 1)
    pc_eigvals, principal_components = np.linalg.eigh(centered_cov)

    k = 40 # Number of principal components
    # sort the eigenvalues in descending order
    idx = np.argsort(pc_eigvals)[::-1][:k]
    evals = pc_eigvals[idx]
    fig, ax = plt.subplots(figsize=(10, 6))
    k_range = range(1, k + 1)
    ax.plot(k_range, evals, "-", color="red", label="variance with the largest {} principal components".format(k))
    ax.set_xlabel("principal component")
    ax.set_ylabel("Variance")
    ax.set_title(r"Explained Variance of the $k^{th}$ component")
    ax.legend()
    plt.show()
    fig.savefig("pb_4_c.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
