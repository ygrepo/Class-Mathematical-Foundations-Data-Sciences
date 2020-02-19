import numpy as np
import plot_tools
from sklearn.datasets import fetch_olivetti_faces


def compute_nearest_neighbors(train_matrix, testImage):
    distances = np.sqrt(np.sum((train_matrix - testImage) ** 2, axis=1))
    idx_of_closest_point_in_train_matrix = np.argsort(distances)
    return idx_of_closest_point_in_train_matrix[0]


def main():
    #test_idx = [1, 2, 3, 10, 22, 40, 59, 63, 87, 94, 78]
    test_idx = [1, 87, 94, 78]

    data = fetch_olivetti_faces()
    targets = data.target
    data = data.images.reshape((len(data.images), -1))

    train_idx = np.array(list(set(list(range(data.shape[0]))) - set(test_idx)))

    train_set = data[train_idx]
    y_train = targets[train_idx]
    test_set = data[np.array(test_idx)]
    y_test = targets[np.array(test_idx)]
    print(train_set.shape, test_set.shape)

    imgs = []
    estLabels = []
    for i in range(test_set.shape[0]):
        testImage = test_set[i, :]
        nnIdx = compute_nearest_neighbors(train_set, testImage)
        imgs.extend([testImage, train_set[nnIdx, :]])
        estLabels.append(y_train[nnIdx])

    row_titles = ['Test', 'Nearest']
    col_titles = ['%d vs. %d' % (i, j) for i, j in zip(y_test, estLabels)]
    plot_tools.plot_image_grid(imgs,
                               "Image-NearestNeighbor",
                               (64, 64), len(test_set), 2, True, row_titles=row_titles, col_titles=col_titles)


if __name__ == "__main__":
    main()
