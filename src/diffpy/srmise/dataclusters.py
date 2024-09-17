#!/usr/bin/env python
##############################################################################
#
# SrMise            by Luke Granlund
#                   (c) 2014 trustees of the Michigan State University
#                   (c) 2024 trustees of Columbia University in the City of New York
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################
"""Defines class to partition sequences representing the x and y axis into peak-like clusters."""

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("diffpy.srmise")


class DataClusters:
    """Find clusters corresponding to peaks in the PDF (y-array)

    DataClusters determines which points in inter-atomic distane, r,
    correspond to peaks in the PDF.  The division between clusters
    is contiguous, with borders between clusters likely near relative
    minima in the data.

    Clusters are iteratively formed around points with the largest
    PDF values.  New clusters are added only when the unclustered data
    point under consideration is greater than a given distance (the
    'resolution') from the nearest existing cluster.

    Data members
    ------------
    x : array
      The array of r values.
    y : sequence of y values
      The array of PDF values, G(r)
    res : float
      The clustering resolution, i.e., the number of distance another point has to
      be away from the center of an existing cluster to before a new cluster is
      formed.  A value of zero allows every point to be a cluster.
    data_order : array
      The array of x, y indices ordered by decreasing y
    clusters : ndarray
      The array of cluster ranges
    current_idx : int
      The index of data_order currently considered
    """

    def __init__(self, x, y, res):
        """Constructor

        Parameters
        ----------
        x : array
          The array of r values.
        y : sequence of y values
          The array of PDF values, G(r)
        res : float
          The clustering resolution, i.e., the distance another point has to
          be away from the center of an existing cluster to before a new cluster is
          formed.  A value of zero allows every point to be cluster.
        """
        # Track internal state of clustering.
        self.INIT = 0
        self.READY = 1
        self.CLUSTERING = 2
        self.DONE = 3
        self._clear()
        self._setdata(x, y, res)

        return

    # This iterator operates not over found clusters, but over the process of
    # clustering.  This behavior could cause confusion and should perhaps be
    # altered.
    def __iter__(self):
        return self

    def __eq__(self, other):
        if not isinstance(other, DataClusters):
            return False
        return (
            np.array_equal(self.x, other.x)
            and np.array_equal(self.y, other.y)
            and np.array_equal(self.data_order, other.data_order)
            and np.array_equal(self.clusters, other.clusters)
            and self.res == other.res
            and self.current_idx == other.current_idx
            and self.lastcluster_idx == other.lastcluster_idx
            and self.lastpoint_idx == other.lastpoint_idx
            and self.status == other.status
            and self.INIT == other.INIT
            and self.READY == other.READY
            and self.CLUSTERING == other.CLUSTERING
            and self.DONE == other.DONE
        )

    def _clear(self):
        """
        Clear all data and reset the cluster object to a transient initial state.

        The purpose of this method is to provide a clean state before creating new clustering operations.
        The object is updated in-place and no new instance is returned.

        Returns
        -------
        None
        """
        self.x = np.array([])
        self.y = np.array([])
        self.data_order = np.array([])
        self.clusters = np.array([[]])
        self.res = 0
        self.current_idx = 0
        self.lastcluster_idx = None
        self.lastpoint_idx = None
        self.status = self.INIT
        return

    def reset_clusters(self):
        """Reset all progress on clustering."""
        self.clusters = np.array([[self.data_order[-1], self.data_order[-1]]])
        self.current_idx = self.data_order.size - 1
        self.lastcluster_idx = 0
        self.lastpoint_idx = self.data_order[-1]

        if self.status != self.INIT:
            self.status = self.READY
        return

    def _setdata(self, x, y, res):
        """Assign data members for x- and y-coordinates, and resolution.

        Parameters
        ----------
        x : array
          The array of r values.
        y : sequence of y values
          The array of PDF values, G(r)
        res : float
          The clustering resolution, i.e., the distance another point has to
          be away from the center of an existing cluster to before a new cluster is
          formed.  A value of zero allows every point to be cluster.
        """
        if len(x) != len(y):
            raise ValueError("Sequences x and y must have the same length.")
        if res < 0:
            raise ValueError(
                "Value of resolution parameter is less than zero.  Please rerun specifying a non-negative res"
            )
        self.x = x
        self.y = y
        self.res = res
        if x.size == 0:
            self.data_order = np.array([])
            self.clusters = np.array([[]])
            self.current_idx = 0
            self.lastpoint_idx = None
            self.status = self.INIT
        else:
            self.data_order = self.y.argsort()
            self.clusters = np.array([[self.data_order[-1], self.data_order[-1]]])
            self.current_idx = len(self.data_order) - 1
            self.lastpoint_idx = self.data_order[-1]
            self.status = self.READY
        self.lastcluster_idx = None
        return

    def __next__(self):
        """Cluster point with largest y-coordinate left, returning self.

        next() always adds at least one additional point to the existing
        cluster, or raises an exception if all points have been clustered.
        """
        if self.status == self.INIT:
            raise Exception("Cannot cluster next point while status is INIT.")

        if self.status == self.READY:
            self.status = self.CLUSTERING
        elif self.status == self.DONE:
            raise StopIteration

        # Find next unclustered point, if one exists.
        nearest_cluster = [0, 0.0]
        while nearest_cluster[1] == 0.0 and self.current_idx > 0:
            self.current_idx += -1
            test_idx = self.data_order[self.current_idx]
            nearest_cluster = self.find_nearest_cluster(test_idx)

        # Check status
        if self.current_idx <= 0:
            self.status = self.DONE
            if nearest_cluster[1] == 0.0:
                # Last point already clustered, so stop immediately.
                raise StopIteration

        self.lastpoint_idx = test_idx

        if np.abs(nearest_cluster[1]) <= self.res:
            # Add to an existing cluster
            self.lastcluster_idx = nearest_cluster[0]
            if test_idx < self.clusters[nearest_cluster[0], 0]:
                self.clusters[nearest_cluster[0], 0] = test_idx
            else:
                self.clusters[nearest_cluster[0], 1] = test_idx
        else:
            # Make a new cluster
            if nearest_cluster[1] < 0:
                # Insert left of nearest cluster
                self.lastcluster_idx = nearest_cluster[0]
            else:
                # insert right of nearest cluster
                self.lastcluster_idx = nearest_cluster[0] + 1
            self.clusters = np.insert(self.clusters, int(self.lastcluster_idx), [test_idx, test_idx], 0)
        return self

    def makeclusters(self):
        """Cluster all remaining data."""
        for i in self:
            pass
        return

    def find_nearest_cluster2(self, x):
        """Return [cluster index, distance] for cluster nearest to x.

        Parameters
        ----------
        x : ndarray
            Coordinate of point of interest

        The distance is positive/negative if the point is right/left of the
        nearest cluster.  If the point is within an existing cluster then
        distance = 0.

        Returns
        -------
        array-like
            The index of the nearest cluster, and the distance for cluster nearest to x. None if no cluster
        """
        if self.status == self.INIT:
            raise Exception("Cannot cluster next point while status is INIT.")

        idx = np.searchsorted(self.x, x)
        if idx > 0 or idx >= len(self.x):
            return self.find_nearest_cluster(idx)
        else:
            # Choose adjacent index nearest to x
            if (self.x[idx] - x) < (x - self.x[idx - 1]):
                return self.find_nearest_cluster(idx)
            else:
                return self.find_nearest_cluster(idx - 1)

    def find_nearest_cluster(self, idx):
        """Return [cluster index, distance] for cluster nearest to x[idx].

        The distance is positive/negative if the point is right/left of the
        nearest cluster.  If the point is within an existing cluster then
        distance = 0.

        Parameters
        ----------
        idx : array-like
            index of point in self.x of interest.

        Returns
        -------
        array-like
            The array of cluster index and the distacne to the nearest cluster. None if no clusters exist.
        """
        if self.status == self.INIT:
            raise Exception("Cannot cluster next point while status is INIT.")

        clusters_flat = self.clusters.flatten()
        if len(clusters_flat) == 0:
            return None

        flat_idx = clusters_flat.searchsorted(idx)
        near_idx = flat_idx / 2

        if flat_idx == len(clusters_flat):
            # test_idx is right of the last cluster
            return [near_idx - 1, self.x[idx] - self.x[self.clusters[-1, 1]]]
        if clusters_flat[flat_idx] == idx or flat_idx % 2 == 1:
            # idx is within some cluster
            return [near_idx, 0.0]
        if flat_idx == 0:
            # idx is left of the first cluster
            return [near_idx, self.x[idx] - self.x[self.clusters[0, 0]]]

        # Calculate which of the two nearest clusters is closer
        distances = np.array(
            [
                self.x[idx] - self.x[self.clusters[int(near_idx) - 1, 1]],
                self.x[idx] - self.x[self.clusters[int(near_idx), 0]],
            ]
        )
        if distances[0] < np.abs(distances[1]):
            return [near_idx - 1, distances[0]]
        else:
            return [near_idx, distances[1]]

    def cluster_is_full(self, cluster_idx):
        """Return whether the given cluster can grow.

        A cluster is full if no unclustered points remain adjacent to its
        boundaries.

        Parameters
        ----------
        cluster_idx : array-like
            The index of the cluster to test

        Returns
        -------
        bools
            True if the cluster is full, False otherwise
        """
        if cluster_idx > 0:
            low = self.clusters[cluster_idx - 1, 1] + 1
        else:
            low = 0
        if cluster_idx < len(self.clusters) - 1:
            high = self.clusters[cluster_idx + 1, 0] - 1
        else:
            high = len(self.data_order) - 1
        return self.clusters[cluster_idx, 0] == low and self.clusters[cluster_idx, 1] == high

    def combine_clusters(self, combine):
        """Combine clusters specified by each subarray of cluster indices.

        Clusters to combine must be contiguous, increasing, and have no
        unclustered points between them.

        Parameters
        ----------
        combine : ndarray
            [[leftmost_idx1, ..., rightmost_idx1], ...]

        Returns
        -------
        None
        """
        # Ensure that the same clusters aren't combined multiple times.
        combine_flat = np.array(combine).ravel()
        if len(combine_flat) != len(np.unique(combine_flat)):
            raise ValueError("Cannot combine a single cluster multiple times.")

        for c in combine:
            # Test that all clusters are contiguous and adjacent
            first = c[0]
            for i in range(c[0], c[-1]):
                if c[i + 1 - first] - 1 != c[i - first]:
                    raise ValueError(
                        "".join(
                            [
                                "Clusters  ",
                                str(c[i]),
                                " and ",
                                str(c[i + 1]),
                                " are not contiguous and/or increasing.",
                            ]
                        )
                    )
                if self.clusters[i + 1, 0] - self.clusters[i, 1] != 1:
                    raise ValueError(
                        "".join(
                            [
                                "Clusters  ",
                                str(c[i]),
                                " and ",
                                str(c[i + 1]),
                                " have unclustered points between them.",
                            ]
                        )
                    )

            # update cluster endpoints
            self.clusters[c[0], 1] = self.clusters[c[-1], 1]
        todelete = np.array([c[1:] for c in combine]).ravel()
        self.clusters = np.delete(self.clusters, todelete, 0)

    def find_adjacent_clusters(self):
        """Return all cluster indices with no unclustered points between them.

        Return array([[leftmost_idx1,...,rightmost_idx1],...]) such that there
        are no unclustered points between each element in subarray of clusters
        (inclusive).  If no such clusters exist, returns an empty array.
        """
        if self.status == self.INIT:
            raise Exception("Cannot cluster next point while status is INIT.")

        adj = []
        left_idx = 0

        while left_idx < len(self.clusters) - 1:
            while (
                left_idx < len(self.clusters) - 1
                and self.clusters[left_idx + 1, 0] - self.clusters[left_idx, 1] != 1
            ):
                left_idx += 1

            # Not left_idx+1 since left_idx=len(self.clusters)-2 even if no
            # clusters are actually adjacent.
            right_idx = left_idx

            while (
                right_idx < len(self.clusters) - 1
                and self.clusters[right_idx + 1, 0] - self.clusters[right_idx, 1] == 1
            ):
                right_idx += 1

            if right_idx > left_idx:
                adj.append(range(left_idx, right_idx + 1))
            left_idx = right_idx + 1  # set for next possible left_idx
        return np.array(adj)

    def cut(self, idx):
        """Return slice(s) for data given cluster index (or indices).

        Parameters
        idx - Cluster index (or sequence of indices).
        """
        data_ids = self.clusters[idx]
        if len(data_ids) == data_ids.size:
            # idx is a scalar, so give single slice object
            return slice(data_ids[0], data_ids[1] + 1)
        else:
            # idx is a list/slice, so give list of slice objects
            return [slice(c[0], c[1] + 1) for c in data_ids]

    def cluster_boundaries(self):
        """Return sequence with (x,y) of all cluster boundaries."""
        boundaries = []
        for cluster in self.clusters:
            xlo = np.mean(self.x[cluster[0] - 1 : cluster[0] + 1])
            ylo = np.mean(self.y[cluster[0] - 1 : cluster[0] + 1])
            xhi = np.mean(self.x[cluster[1] : cluster[1] + 2])
            yhi = np.mean(self.y[cluster[1] : cluster[1] + 2])
            boundaries.append((xlo, ylo))
            boundaries.append((xhi, yhi))
        return np.unique(boundaries)

    def plot(self, *args, **kwds):
        """Plot the data with vertical lines at the cluster divisions.

        args and kwds passed to matplotlib.plot()
        """
        plt.figure()
        ax = plt.subplot(111)
        plt.ion()
        plt.plot(self.x, self.y, *args, **kwds)
        plt.ioff()
        boundaries = self.cluster_boundaries()
        (ymin, ymax) = ax.get_ylim()
        for b in boundaries:
            plt.axvline(b[0], 0, (b[1] - ymin) / (ymax - ymin), color="k")
        plt.ion()
        ax.figure.canvas.draw()
        return

    def animate(self):
        """Animate clustering.  Restores state when complete."""
        clusters = self.clusters
        current_idx = self.current_idx
        lastcluster_idx = self.lastcluster_idx
        lastpoint_idx = self.lastpoint_idx
        status = self.status
        self.reset_clusters()

        fig, ax = plt.subplots()
        canvas = fig.canvas
        background = canvas.copy_from_bbox(ax.bbox)
        ymin, ymax = ax.get_ylim()
        all_lines = []
        for i in self:
            canvas.restore_region(background)
            boundaries = self.cluster_boundaries()
            for i, b in enumerate(boundaries):
                height = (b[1] - ymin) / (ymax - ymin)
                if i < len(all_lines):
                    all_lines[i].set_xdata([b[0], b[0]])
                    all_lines[i].set_ydata([0, height])
                    ax.draw_artist(all_lines[i])
                else:
                    line = plt.axvline(b[0], 0, height, color="k", animated=True)
                    ax.draw_artist(line)
                    all_lines.append(line)
            canvas.blit(ax.bbox)

        self.clusters = clusters
        self.current_idx = current_idx
        self.lastcluster_idx = lastcluster_idx
        self.lastpoint_idx = lastpoint_idx
        self.status = status
        return


# End of class DataClusters


# simple test code
if __name__ == "__main__":

    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    y = np.array(
        [
            0.0183156,
            0.105399,
            0.36788,
            0.778806,
            1.00012,
            0.780731,
            0.386195,
            0.210798,
            0.386195,
            0.780731,
            1.00012,
            0.778806,
            0.36788,
            0.105399,
            0.0183156,
        ]
    )

    testcluster = DataClusters(x, y, 0.1)
    testcluster.makeclusters()

    print(testcluster.clusters)
    adj = testcluster.find_adjacent_clusters()
    print(adj)
    if len(adj) > 0:
        testcluster.combine_clusters(adj)
    print(testcluster.clusters)
