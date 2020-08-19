from sklearn.cluster import KMeans
from sklearn import mixture


class Cluster:

    @staticmethod
    def kmeans_cluster(n_class, data):
        results = KMeans(n_clusters=n_class, random_state=0).fit(data)
        labels = results.labels_
        return results, labels

    @staticmethod
    def gmm_cluster(n_class, data):
        g = mixture.GaussianMixture(n_components=n_class)
        g.fit(data)
        labels = g.predict(data)
        return g, labels