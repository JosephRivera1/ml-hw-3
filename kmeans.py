import numpy as np
import scipy
import argparse
from sklearn.metrics.pairwise import cosine_similarity

def jaccard(A, B):
    #Find intersection
    AinterB = A.intersection(B)

    #Find union
    AunionB = A.union(B)

    #Take the ratio of sizes
    sim = len(AinterB)/len(AunionB)
    
    return sim

def sse(trueClass, predClass):
    # compute sum of squared errors
    return np.sum((trueClass - predClass) ** 2)

class KMeans:
    def __init__(self, k=10, max_iter=500, metric='euclidean'):
        self.k = k
        self.max_iter = max_iter
        self.metric = metric

    def fit(self, X, earlyStopping):
        sse_prev = float('inf') # initialize at inf, so we can continue

        # randomly initialize centroid placement
        # print('data shape: ' + str(X.shape)) # 10k,784

        # randomly select k centers
        placement = np.random.choice(X.shape[0], self.k, replace=False)
        # print(placement) # 10 numbers

        self.centroids = X[placement, :] # extract those random points as our starting centroids

        # iterate
        for i in range(self.max_iter):

            # compute the distances using each metric
            distances = scipy.spatial.distance.cdist(X, self.centroids, metric=self.metric)

            # print(distances[0,:].shape) # 10k,10
            
            # alternate implementations of the metrics
            
            # if self.metric == 'euclidean':
            #     distances = np.sqrt(np.sum((X - self.centroids)**2, axis=-1))
            # elif self.metric == 'jaccard':
            #     distances = jaccard(X, self.centroids)
            # elif self.metric == 'cosine':
            #     distances = cosine_similarity(X,self.centroids)

            # define labels by minimum distance - argmin returns the indices of the minimum values along an axis. 
            labels = np.argmin(distances, axis=1)

            # print(labels[0].shape)
            
            # empty new_centroids variable
            new_centroids = np.empty(self.centroids.shape)
            
            # for all clusters (k)
            for j in range(self.k):
            
                # if any of our labelled datapoints equal a cluster number
                if np.any(labels == j):
                    
                    # we set the new centroid center as the average of our data points
                    new_centroids[j, :] = X[labels == j, :].mean(axis=0)
            
                else:
        
                    new_centroids[j, :] = self.centroids[j, :]

            # calculate sum of squared errors
            sse_curr = sse(X, new_centroids[labels])

            ''' Set up the same stop criteria: “when there is no change in centroid position OR when the
            SSE value increases in the next iteration OR when the maximum preset value (e.g., 500, you
            can set the preset value by yourself) of iteration is complete”, for Euclidean-K-means, Cosine-Kmeans, Jarcard-K-means. Which method requires more iterations and times to converge? (10
            points) '''

            if earlyStopping and sse_curr > sse_prev:
                print('stopping early on iteration: ' + str(i))
                break

            if earlyStopping and np.allclose(self.centroids, new_centroids):
                print('stopping early on same centriods')
                break

            if earlyStopping and i == self.max_iter - 1:
                print('finished at max iters')
                break

            self.centroids = new_centroids
            sse_prev = sse_curr

        self.labels = labels

    def pred(self, X):
        
        distances = scipy.spatial.distance.cdist(X, self.centroids, metric=self.metric)
        distances[distances == 0] = 0.0000001 # set small for 0 distance

        return np.argmin(distances, axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--maxIteration', type=int, default=500)
    parser.add_argument('-s', '--earlyStopping', type=bool, default=False)

    args = parser.parse_args()
    print(args)

    # Load data
    X = np.loadtxt('data.csv', delimiter=',').astype(int)
    trueClass = np.loadtxt('label.csv', delimiter=',').astype(int)

    ''' Q1: Run K-means clustering with Euclidean, Cosine and Jarcard similarity. Specify K= the
        number of categorical values of y (the number of classifications). Compare the SSEs of
        Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which method is better?  '''

    # find unique classes in truth labels
    k = np.unique(trueClass).size
    print("Number of Classes: " + str(k))

    # Cluster data with KMeans using each method
    metrics = ["euclidean","jaccard","cosine"]
    for metric in metrics:
        
        # initialize kmeans
        km = KMeans(k=k, max_iter=args.maxIteration, metric=metric)
        
        # train the model
        km.fit(X, earlyStopping=args.earlyStopping)

        # run predion
        predClass = km.pred(X)

        # score the model
        acc = sse(trueClass, predClass)

        # Print clustering accuracy
        print("SSE of " + str(metric) + " metric: " + str(round(acc,4)))

        ''' Q2: Compare the accuracies of Euclidean-K-means Cosine-K-means, Jarcard-K-means. First,
            label each cluster using the majority vote label of the data points in that cluster. Later, compute
            the predive accuracy of Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which metric
            is better? '''

        # label each cluster using the majority vote label of the data points in that cluster
        labels = np.zeros_like(predClass)
        for i in range(k):
            mask = predClass == i
            if np.sum(mask) == 0:
                continue
            label = np.bincount(trueClass[mask]).argmax()
            labels[mask] = label

        # compute accuracy of the model
        acc = np.sum(labels == trueClass) / len(trueClass)

        # Print clustering accuracy
        print("Accuracy of " + str(metric) + " metric: " + str(round(acc,4)))

