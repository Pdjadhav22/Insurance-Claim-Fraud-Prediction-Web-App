from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from fileOperations.fileMethods import fileMethods

class kMeansClustring:
    def __init__(self):
        pass

    def elbowPlot(self,data):
        self.wcss = []
        for i in range(1,11):
            kmeans = KMeans(n_clusters=i,init='k-means++',random_state=40)
            kmeans.fit(data)
            self.wcss.append(kmeans.inertia_)
        plt.plot(range(1,11),self.wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('Data_Preprocessor/KMeansElbow.png')

        # programaticaly finding optimum cluster value
    def getKnee(self,data):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=40)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        kn = KneeLocator(range(1,11),wcss, curve='convex',direction='decreasing')
        print(kn.knee)
        return kn.knee

    def createClusters(self,data,knee):
        cluster = KMeans(n_clusters=knee,init='k-means++', random_state=40)
        y_means = cluster.fit_predict(data)
        data['Cluster'] = y_means
        print(data['Cluster'].value_counts())

        # Save Model
        modelStore = fileMethods()
        modelStore.saveModel(cluster,'Kmeans','Kmeans')
        print('Model Stored')
        return data


