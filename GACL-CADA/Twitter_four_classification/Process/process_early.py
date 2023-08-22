import os
from Process.dataset_early import GraphDataset, GraphDataset_100,test_GraphDataset


cwd=os.getcwd()


def loadData(dataname, fold_x_train,fold_x_100_train,fold_x_test,method,droprate):
    print("loading source train set", )
    traindata_list = GraphDataset(fold_x_train,dataname, droprate=droprate)
    print("source train no:", len(traindata_list))
    print("loading target train set", )

    traindata_100_list = GraphDataset_100(fold_x_100_train, dataname,method,droprate=droprate)
    print("target train no:", len(traindata_100_list))
    print("loading test set", )
    testdata_list = test_GraphDataset(fold_x_test, dataname,method,droprate=0)
    print("test no:", len(testdata_list))
    return traindata_list,traindata_100_list, testdata_list
