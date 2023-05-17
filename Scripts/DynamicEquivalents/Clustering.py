import MergeRandomOutputs
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt

def getCentroids(curves, nb_clusters = 5):
    nb_names = curves.shape[0]
    nb_disturb = curves.shape[1]
    nb_runs = curves.shape[2]
    nb_time_steps = curves.shape[3]

    curves = np.moveaxis(curves, [0,1,2,3], [2, 3, 0, 1])
    curves = curves.reshape(nb_runs, nb_time_steps, nb_names * nb_disturb)

    model = TimeSeriesKMeans(n_clusters=nb_clusters, metric='euclidean', n_init=10)
    model.fit(curves)

    centroids = model.cluster_centers_
    centroids = centroids.reshape(nb_clusters, nb_time_steps, nb_names, nb_disturb)
    centroids = np.moveaxis(centroids, [2, 3, 0, 1], [0, 1, 2, 3])
    return centroids
    model2 = TimeSeriesKMeans(n_clusters=5, n_init=10, init=model.cluster_centers_)
    model.fit(curves)


if __name__ == '__main__':
    time_precision = 0.01
    # curves = -100 * MergeRandomOutputs.mergeCurves(['/home/fsabot/Desktop/dynawo-algorithms/examples/CIGRE_MV_Wind/SA/RandomRuns/It_%03d/fic_MULTIPLE.xml' % run_id for run_id in range(300)], ['InfBus_infiniteBus_PPu', 'InfBus_infiniteBus_QPu'], time_precision)
    curves = -100 * MergeRandomOutputs.mergeCurves(['/home/fsabot/Desktop/dynawo-algorithms/examples/CIGRE_MV_Wind/SA/RandomRuns/It_%03d/fic_MULTIPLE.xml' % run_id for run_id in range(300)], ['NETWORK_LINE-GEN_P2_value', 'NETWORK_LINE-GEN_Q2_value'], time_precision)
    nb_names = curves.shape[0]
    nb_disturb = curves.shape[1]
    nb_runs = curves.shape[2]
    nb_time_steps = curves.shape[3]

    curves = np.moveaxis(curves, [0,1,2,3], [2, 3, 0, 1])
    new_curves = curves[:,:,:,:]
    new_curves = new_curves.reshape(nb_runs, nb_time_steps, nb_names * nb_disturb)

    model = TimeSeriesKMeans(n_clusters=3, metric='euclidean', n_init=10)
    model.fit(new_curves) # array-like of shape=(n_ts, sz, d)
    # the fist axis is the sample axis, n_ts being the number of time series;
    # the second axis is the time axis, sz being the maximum number of time points
    # the third axis is the dimension axis, d being the number of dimensions.
    y = model.predict(new_curves)

    clustered_curves = {}
    for i in range(nb_runs):
        k = y[i]
        if k in clustered_curves:
            clustered_curves[k].append(new_curves[i, :, :])
        else:
            clustered_curves[k] = [new_curves[i, :, :]]


    # fig, axs = plt.subplots(3, 1)
    t_axis = np.array([i * time_precision for i in range(nb_time_steps)]) # * 2*4)

    index = 0
    for k, curves in clustered_curves.items():
        print('K = ', k, ', nb curves: ', len(curves))
        for curve in curves:
            plt.plot(t_axis, curve[:, 5])
        plt.savefig('Test%d.png' % k)
        plt.close()
        index += 1
    
    percentiles = np.percentile(new_curves[:,-1,:], axis=0, q=[10, 30, 50, 70, 90])
    clustered_curves_percentile = {}
    for i in range(nb_runs):
        curve = new_curves[i,:,:]
        distances = [sum((percentiles[j] - curve[-1,:])**2) for j in range(5)]
        min_dist_index = distances.index(min(distances))

        if min_dist_index in clustered_curves_percentile:
            clustered_curves_percentile[min_dist_index].append(new_curves[i, :, :])
        else:
            clustered_curves_percentile[min_dist_index] = [new_curves[i, :, :]]

    index = 0
    for k, curves in clustered_curves_percentile.items():
        print('K = ', k, ', nb curves: ', len(curves))
        for curve in curves:
            plt.plot(t_axis, curve[:, 5])
        plt.savefig('TestPercentile%d.png' % k)
        plt.close()
        index += 1

    for d in range(nb_disturb):
        for center in model.cluster_centers_:
            plt.plot(t_axis, center[:, d])
        plt.savefig('CentersP%d.png' % (d+1))
        plt.close()
        for center in model.cluster_centers_:
            plt.plot(t_axis, center[:, d + nb_disturb])
        plt.savefig('CentersQ%d.png' % (d+1))
        plt.close()

    # https://stats.stackexchange.com/questions/494078/kmeans-clustering-can-inertia-increase-with-number-of-clusters
    # Do k-means with different seeds and keep the best one (then should be monotonously decreasing)

    # How to keep the clusters for different system conditions?
    # Just build equivalent by checking all previous simulations (still issue of slightly different init P/Q -> rescale?)

    """ inertias = []
    for i in range(1,10):
        model = TimeSeriesKMeans(n_clusters=i, metric='euclidean')
        model.fit(new_curves)
        inertias.append(model.inertia_)

    inertias_better = []
    for i in range(1,10):
        model = TimeSeriesKMeans(n_clusters=i, metric='euclidean', n_init=10)
        model.fit(new_curves)
        inertias_better.append(model.inertia_)

    plt.plot(range(1,10), inertias, marker='o')
    plt.plot(range(1,10), inertias_better, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('Elbow.png')
    plt.close() """
