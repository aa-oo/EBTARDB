import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold, datasets
import pickle
import torch



def plot_embedding_2d(X, y, weight, title=None):

    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(X.shape[0]):
        if y[i] == 1:
            ax.text(X[i, 0], X[i, 1], '+', color='red', fontdict={'weight': 'bold', 'size': 7})
        # elif y[i] == 0:
        #     ax.text(X[i,0], X[i,1], '*', color='b', fontdict={'weight': 'bold', 'size':7})
        # elif y[i] == 3:
        #     ax.text(X[i, 0], X[i, 1], 'o', color='g', fontdict={'weight': 'bold', 'size': 7})
        elif y[i] == 0:
            if weight[i] > 0.8:
                ax.text(X[i, 0], X[i, 1], 'o', color='red', fontdict={'weight': 'bold', 'size': 7})
            elif weight[i] > 0.7:
                ax.text(X[i, 0], X[i, 1], 'o', color='goldenrod', fontdict={'weight': 'bold', 'size': 7})
            elif weight[i] > 0.58:
                ax.text(X[i, 0], X[i, 1], 'o', color='darkturquoise', fontdict={'weight': 'bold', 'size': 7})
            elif weight[i] > 0.52:
                ax.text(X[i, 0], X[i, 1], 'o', color='dodgerblue', fontdict={'weight': 'bold', 'size': 7})
    if title is not None:
        plt.title(title)

def plot_embedding(dataname,X, y, domain, title=None):

    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(X.shape[0]):
        if int(y[i]) == 0 and domain[i]==0:
            ax.text(X[i, 0], X[i, 1], '+', color='red', fontdict={'weight': 'bold', 'size': 7})
       
        elif int(y[i]) == 1 and domain[i]==0:
            ax.text(X[i, 0], X[i, 1], '+', color='green', fontdict={'weight': 'bold', 'size': 7})
        elif int(y[i]) == 0 and domain[i]==1:
            ax.text(X[i, 0], X[i, 1], 'o', color='yellow', fontdict={'weight': 'bold', 'size': 7})
        elif int(y[i]) == 1 and domain[i]==1:
            ax.text(X[i, 0], X[i, 1], 'o', color='blue', fontdict={'weight': 'bold', 'size': 7})
    
    ax.set_yticks([])
    ax.set_xticks([])
    if title is not None:
        plt.title(title)

if __name__ == "__main__":

    word_embedding = pickle.load(open('new_data_distribution_3_1.pickle', 'rb'))
    word_weight = pickle.load(open('feature_distributions_MetaDetector.pickle', 'rb'))
    data, label = word_embedding[0], word_embedding[1]
    _, weights, _ = word_weight[0], word_weight[1], word_weight[2]

    # print(data.shape)
    # print(label.shape)
    # print(len(weights))

    data_np = np.array(data)
    label_np = np.array(label)
    weight_np = np.array(weights)
    # print(data_np.shape)
    # print(label_np.shape)
    # print(weight_np)

    tsne = manifold.TSNE(n_components=2, init='pca')
    features_tsne = tsne.fit_transform(data_np)
    print(features_tsne)
    #
    # plot_embedding_2d(features_tsne, label_np, weight_np, "The t-SNE visualization of event distributions")

    plt.show()

    # digits = datasets.load_digits(n_class=5)
    # X = digits.data
    # y = digits.target
    #
    # print(X.shape)
    # print(type(X))
    # print(y.shape)
    # print(type(y))