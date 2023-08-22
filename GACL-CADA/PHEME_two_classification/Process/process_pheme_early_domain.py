import os
from Process.dataset_pheme_early import UdGraphDataset,UdGraphDataset_100,test_UdGraphDataset

cwd=os.getcwd()


# def loadUdData(dataname, fold_x_train,fold_x_100_train,fold_x_test,method,droprate):
    
#     print("loading source train set", )
#     traindata_list = UdGraphDataset(fold_x_train, droprate=droprate)
#     print("source train no:", len(traindata_list))
#     print("loading target train set", )
#     traindata_100_list = UdGraphDataset_100(fold_x_100_train, method,droprate=droprate)
#     print("target train no:", len(traindata_100_list))
#     print("loading test set", )
#     testdata_list = test_UdGraphDataset(fold_x_test, method,droprate=0) # droprate*****
#     print("test no:", len(testdata_list))
#     return traindata_list,traindata_100_list, testdata_list
def loadUdData(dataname, fold_x_train,fold_x_val,fold_x_test,method,droprate):
    
    print("loading source train set1", )
    traindata_list = UdGraphDataset(fold_x_train, droprate=droprate)
    print("source train no:", len(traindata_list))
    print("loading target train set", )
    traindata_100_list = test_UdGraphDataset(fold_x_val, method,droprate=0)
    print("target train no:", len(traindata_100_list))
    print("loading test set", )
    testdata_list = test_UdGraphDataset(fold_x_test, method,droprate=0) # droprate*****
    print("test no:", len(testdata_list))
    return traindata_list,traindata_100_list, testdata_list