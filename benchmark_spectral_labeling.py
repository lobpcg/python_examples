"""Benchmarks for labeling in spectral clustering methods."""

import timeit
import csv
import gc

import numpy as np
from scipy import sparse
from memory_profiler import memory_usage

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score


def to_numpy(x_file, y_file):
    i_list, j_list = [], []
    for row in x_file:
        # Convert from one-based to zero-based indexing.
        i = int(row[0]) - 1
        j = int(row[1]) - 1
        i_list.append(i)
        j_list.append(j)
        # symmetrize
        j_list.append(i)
        i_list.append(j)

    vals = [1 for _ in i_list]
    n = max(i_list) + 1

    s = sparse.coo_matrix((vals, (i_list, j_list)), shape=(n, n), dtype=np.float32)
    y = []
    for row in y_file:
        y.append(int(row[1]) - 1)

    return s, np.array(y)


def get_data():
    cases = [
        # (1000, "static_lowOverlap_lowBlockSizeVar"),
        # (5000, "static_lowOverlap_lowBlockSizeVar"),
        # (20000, "static_lowOverlap_lowBlockSizeVar"),
        # (50000, "static_lowOverlap_lowBlockSizeVar"),
        # (200000, "static_lowOverlap_lowBlockSizeVar"),
        # (1000, "static_lowOverlap_highBlockSizeVar"),
        # (5000, "static_lowOverlap_highBlockSizeVar"),
        # (20000, "static_lowOverlap_highBlockSizeVar"),
        # (50000, "static_lowOverlap_highBlockSizeVar"),
        # (200000, "static_lowOverlap_highBlockSizeVar"),
        #(1000, "static_highOverlap_lowBlockSizeVar"),
        #(5000, "static_highOverlap_lowBlockSizeVar"),
        (20000, "static_highOverlap_lowBlockSizeVar"),
        #(50000, "static_highOverlap_lowBlockSizeVar"),
        #(200000, "static_highOverlap_lowBlockSizeVar"),
        # (1000, "static_highOverlap_highBlockSizeVar"),
        # (5000, "static_highOverlap_highBlockSizeVar"),
        # (20000, "static_highOverlap_highBlockSizeVar"),
        # (50000, "static_highOverlap_highBlockSizeVar"),
        # (200000, "static_highOverlap_highBlockSizeVar"),
        # (1000000, "static_lowOverlap_lowBlockSizeVar"),
        # (1000000, "static_lowOverlap_highBlockSizeVar"),
        # (1000000, "static_highOverlap_lowBlockSizeVar"),
        # (1000000, "static_highOverlap_highBlockSizeVar"),
    ]

    for size, name in cases:
        x_file = csv.reader(open(f"{name}_{size}_nodes.tsv"), delimiter="\t")
        y_file = csv.reader(
            open(f"{name}_{size}_nodes_truePartition.tsv"), delimiter="\t"
        )
        yield to_numpy(x_file, y_file), name


def profile_and_score(s, y, assign_labels, n_clusters):
    def cluster():
        return SpectralClustering(
            random_state=0,
            n_clusters=n_clusters,
            affinity="precomputed",
            eigen_solver="lobpcg",
            assign_labels=assign_labels,
        ).fit(s)

    gc.collect()
    time = np.mean(timeit.repeat(cluster, repeat=3, number=1))
    memory = np.max(memory_usage(cluster))
    score = adjusted_rand_score(y, cluster().labels_)
    return time, memory, score


def run_benchmark():

    for (s, y), name in get_data():
        n_clusters = np.max(y) + 1
        print(f"Test {name} of size {s.shape} with {n_clusters} clusters")

        for assign_labels in ("kmeans", "discretize", "cluster_qr"):
            time, memory, score = profile_and_score(
                s, y, assign_labels=assign_labels, n_clusters=n_clusters
            )
            print(f"{assign_labels:10}: {score:.3f} ({time:.3f} sec., {memory:.3f} MB)")

        print("\n")


if __name__ == "__main__":
    run_benchmark()
