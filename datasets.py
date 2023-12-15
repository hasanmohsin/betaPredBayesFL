from pkgutil import get_data
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import utils

####################################################
# DATASET GETTERS
####################################################

DATAROOT = "../Dataset" #"../../data" 

# FOR CLASSIFICATION 
# do an noniid split of the dataset into num_clients parts
# each client gets equal sized dataset specified by the client_data_size
# noniidpercent is a number between 0 and 100 indicating the level of non iid i.e. 100 means completely different
# labels for each client, 0 means iid
# outdim is the number of possible classes corresponding to the dataset

# FOR REGRESSION
# There isn't a continuum for heterogeneity: it just sorts the data by an attribute, 
# and splits the data into chunks
def non_iid_split(dataset, num_clients, client_data_size, batch_size, shuffle, shuffle_digits=True,
                        non_iid_frac=1.0, outdim=10):
    
    if outdim != 1:
        noniidpercent = non_iid_frac*100
        digits = torch.arange(outdim) if shuffle_digits == False else torch.randperm(outdim,
                                                                                    generator=torch.Generator().manual_seed(
                                                                                        0))

        lens = num_clients * [client_data_size]

        # split the digits in a fair way
        digits_split = list()
        i = 0
        for n in range(min(outdim, num_clients), 0, -1):
            inc = int((outdim - i) / n)
            digits_split.append(digits[i:i + inc])
            i += inc

        # load and shuffle num_clients*client_data_size from the dataset
        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=num_clients * client_data_size,
                                            shuffle=shuffle)
        dataiter = iter(loader)
        images_train_mnist, labels_train_mnist = dataiter.next()
        data_splitted = list()
        for i in range(num_clients):
            idx = torch.stack([y_ == labels_train_mnist for y_ in digits_split[i % min(outdim, num_clients)]]).sum(0).bool()
            idx_out = torch.stack([y_ == labels_train_mnist for y_ in torch.arange(outdim)]).sum(0).bool()

            data_t = torch.cat((images_train_mnist[idx][:int(client_data_size * noniidpercent / 100)],
                                images_train_mnist[idx_out][:int(client_data_size * (1 - noniidpercent / 100))]))
            target_t = torch.cat((labels_train_mnist[idx][:int(client_data_size * noniidpercent / 100)],
                                labels_train_mnist[idx_out][:int(client_data_size * (1 - noniidpercent / 100))]))
            data_splitted.append(
                torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_t, target_t), batch_size=batch_size,
                                            shuffle=shuffle))
    #regression case
    else:
        data_size = len(dataset) #data.targets.shape[0]
        c_data_size = int(np.floor(data_size/num_clients))

        print("Size of dataset: ", data_size)

        #to use all data
        c_data_size_last = data_size - c_data_size*(num_clients - 1)

        lens = num_clients*[c_data_size]
        lens[-1] = c_data_size_last
        
        #first sort dataset by its value in an input column
        dataset = sort_by_col_value(dataset)

        print("Size of dataset after sorting: ", len(dataset))

        print("Using sequential split for regression NONIID!")
        c_data = utils.seq_split(dataset, lens)
        
        c_dataloaders = []

        #construct dataloaders
        for shard in c_data:
            c_dataloader = torch.utils.data.DataLoader(shard, 
                                                batch_size=batch_size, shuffle=True, 
                                                pin_memory=True)
            c_dataloaders.append(c_dataloader)
        return c_dataloaders, lens

    return data_splitted, lens


def sort_by_col_value(dataset):
    #the dataset is expected to be torch Subset object
    input_data = dataset.dataset.tensors[0][dataset.indices] #need .dataset if this is actually a torch Subset object
    y_data = dataset.dataset.tensors[1][dataset.indices] # should be 1 D

    assert(len(y_data.shape) == 1)

    #pick column with most variance  (sorted between 0 to 1)
    _, max_var_inds = torch.max(torch.var(input_data, dim=0, keepdims=True), dim=1)#min(0, input_data.shape[-1]-1)
    col_index_to_sort = int(max_var_inds[0].item()) 
    print("Sorting along column: ", col_index_to_sort)

    sorted_indices = input_data[:, col_index_to_sort].sort()[1]
    input_data = input_data[sorted_indices]
    y_data = y_data[sorted_indices]

    sorted_dataset = torch.utils.data.TensorDataset(input_data,
                                              y_data)

    return sorted_dataset

#do an iid split of the dataset into num_clients parts
# each client gets equal sized dataset
def iid_split(data, num_clients, batch_size):
    data_size = len(data) #data.targets.shape[0]
    c_data_size = int(np.floor(data_size/num_clients))

    #to use all data
    c_data_size_last = data_size - c_data_size*(num_clients - 1)

    lens = num_clients*[c_data_size]
    lens[-1] = c_data_size_last

    c_data = torch.utils.data.random_split(data, lens)

    c_dataloaders = []

    #construct dataloaders
    for shard in c_data:
        c_dataloader = torch.utils.data.DataLoader(shard, 
                                                batch_size=batch_size, shuffle=True, 
                                                pin_memory=True)
        c_dataloaders.append(c_dataloader)

    #array of datasets
    return c_dataloaders, lens


# Real Estate dataset
# 289 training points
# 125 validation points
def get_realestate(batch_size, normalize = True):
    col1=['No','X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores',\
      'X5 latitude','X6 longitude', 'Y house price of unit area']

    df1 = pd.read_excel('./Dataset/Real estate valuation data set.xlsx',header=None,skiprows=1, na_filter=True,names=col1)
    df1 = df1.dropna()
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 =(df1 - df1.mean())/(df1.std()) #(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    df1 = df1.drop(columns=['No'])
    col1=df1.columns.tolist()
    target = data.pop('Y house price of unit area')
    ds = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    train_size = int(len(ds) * 0.8)
    dataset = (
        ds
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=len(ds))
        .cache()
    )
    train_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[:train_size, :]).float(),
                                              torch.Tensor(target.values[:train_size]).float())
    test_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[train_size:, :]).float(),
                                             torch.Tensor(target.values[train_size:]).float())

    # We shuffle with a buffer the same size as the dataset.
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False
    )
    return trainloader, testloader, train_ds, df1

## Forest Fire dataset
# 361 training points
# 156 validation points
def get_forestfire(batch_size, normalize = True):
    col1=['X','Y','month','day','FFMC','DMC','DC',
     'ISI','temp','RH','wind','rain','area']

    df1 = pd.read_csv('./Dataset/forestfires.csv',header=None,skiprows=1, na_filter=True,names=col1)
    df1 = df1.dropna()

    df1["month"].replace({"jan":1, "feb":2,"mar":3,"apr":4,"may":5,"jun":6, "jul":7,\
                          "aug":8, "sep":9 ,"oct":10,"nov":11,"dec": 12},inplace=True)
    df1['day'].replace({'mon':1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6,\
                        "sun":7},inplace=True)
    df1['area'] = df1['area'].apply(lambda x: np.log(x+1))
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 = (df1 - df1.mean())/(df1.std()) #(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    target = data.pop('area')
    ds = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    train_size = int(len(ds) * 0.8)
    dataset = (
        ds
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=len(ds))
        .cache()
    )
    train_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[:train_size, :]).float(),
                                              torch.Tensor(target.values[:train_size]).float())
    test_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[train_size:, :]).float(),
                                             torch.Tensor(target.values[train_size:]).float())

    # We shuffle with a buffer the same size as the dataset.
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False
    )
    return trainloader, testloader, train_ds, df1


## Wine Quality dataset
# 1119 training points
# 480 validation points
def get_winequality(batch_size, normalize = True):
    col1=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',\
      'free sulfur dioxide','total sulfur dioxide', 'density','pH','sulphates', \
      'alcohol', 'quality']
    df1 = pd.read_csv('./Dataset/winequality-red.csv',header=None,skiprows=1, na_filter=True,names=col1,delimiter=';')
    df1 = df1.dropna()
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 = (df1 - df1.mean())/(df1.std())#(df1-df1.min())/(df1.max()-df1.min())
    
    data=df1[col1]
    target = data.pop('quality')
    ds = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    train_size = int(len(ds) * 0.8)
    dataset = (
        ds
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=len(ds))
        .cache()
    )
    train_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[:train_size, :]).float(),
                                              torch.Tensor(target.values[:train_size]).float())
    test_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[train_size:, :]).float(),
                                             torch.Tensor(target.values[train_size:]).float())

    # We shuffle with a buffer the same size as the dataset.
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False
    )
    return trainloader, testloader, train_ds, df1


## Air quality dataset
# 7485 training points
# 1872 validation points
def get_airquality(batch_size, normalize = True):
    col1=['DATE','TIME','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT','PT08_S2_NMHC',
     'NOX_GT','PT08_S3_NOX','NO2_GT','PT08_S4_NO2','PT08_S5_O3','T','RH','AH']

    df1 = pd.read_excel('./Dataset/AirQualityUCI.xlsx',header=None,skiprows=1, na_filter=True,names=col1)
    #print("Airquality dataframe: " len(df1))
    
    df1 = df1.dropna()
    df1['DATE']=pd.to_datetime(df1.DATE, format='%d-%m-%Y')
    df1['MONTH']= df1['DATE'].dt.month
    df1['HOUR']=df1['TIME'].apply(lambda x: int(str(x).split(':')[0]))
    df1 = df1.drop(columns=['NMHC_GT'])
    df1 = df1.drop(columns=['DATE'])
    df1 = df1.drop(columns=['TIME'])
    
    df1.to_pickle("./Dataset/AirQualityProcessedDF.pickle")
    col1 = df1.columns.tolist()
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 = (df1 - df1.mean())/(df1.std()) #(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    target = data.pop('CO_GT')

    train_size = int(len(data.values) * 0.8)

    train_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[:train_size, :]).float(), torch.Tensor(target.values[:train_size]).float())
    test_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[train_size:, :]).float(), torch.Tensor(target.values[train_size:]).float())
    
    torch.save(train_ds, "./Dataset/AirQualityTrain.pt")
    torch.save(test_ds, "./Dataset/AirQualityTest.pt")
    
    # We shuffle with a buffer the same size as the dataset.
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle = True
    )
    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle = False
    )
    return trainloader, testloader, train_ds, df1


## Bike Sharing dataset
# 584 training points
# 147 validation points
def get_bike(batch_size, normalize = True):
    col1=['instant','dteday','season','yr','mnth','holiday','weekday',
     'workingday','weathersit','temp','atemp','hum','windspeed','casual','registered', 'cnt']
    df1 = pd.read_csv('./Dataset/bike.csv',header=None,skiprows=1, na_filter=True,names=col1)
    df1.dropna()
    df1 = df1.drop(columns=['dteday'])
    df1 = df1.drop(columns=['instant'])
    col1=df1.columns.tolist()
    if normalize:
        #df1 = (df1-df1.min())/(df1.max()-df1.min())
        df1 = (df1 - df1.mean())/(df1.std()) #(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    target = data.pop('cnt')

    train_size = int(len(data.values) * 0.8)

    train_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[:train_size, :]).float(), torch.Tensor(target.values[:train_size]).float())
    test_ds = torch.utils.data.TensorDataset(torch.Tensor(data.values[train_size:, :]).float(), torch.Tensor(target.values[train_size:]).float())
    #print((train_ds))
    
    # We shuffle with a buffer the same size as the dataset.
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle = True
    )
    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle = False
    )
    return trainloader, testloader, train_ds, df1

###############################################
# CLASSIFICATION DATASETS
###############################################

## MNIST dataset
# 60,000 train points, 
# 10,000 validation points
def get_mnist(use_cuda, batch_size, get_datamat = False):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    #60,000 datapoints, 28x28
    train_data = datasets.MNIST(
        root = DATAROOT,
        train =True,
        transform = transform_train,
        download= True
    )

    #10,000 datapoints
    val_data = datasets.MNIST(
        root = DATAROOT,
        train=False,
        download = True,
        transform = transform_val
    )


    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True
                                                , num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True
                                                ,num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False
                                                ,num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False
                                             ,num_workers=3)
    if get_datamat:
        return trainloader, valloader, train_data
    else:
        return trainloader, valloader



## EMNIST by letters dataset
# 124,800 train points, 
# 10,000 validation points
def get_emnist(use_cuda, batch_size, get_datamat = False):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    #60,000 datapoints, 28x28
    train_data = datasets.EMNIST(
        root = DATAROOT,
        split = 'bymerge',
        train =True,
        transform = transform_train,
        download= True
    )

    #10,000 datapoints
    val_data = datasets.EMNIST(
        root = DATAROOT,
        split = 'bymerge',
        train=False,
        download = True,
        transform = transform_val
    )


    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True
                                                , num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True
                                                ,num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False
                                                ,num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False
                                             ,num_workers=3)
    if get_datamat:
        return trainloader, valloader, train_data
    else:
        return trainloader, valloader

## fMNIST by letters dataset
# _ train points, 
# _ validation points
def get_fashion_mnist(use_cuda, batch_size, get_datamat = False):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    #_ datapoints, 28x28
    train_data = datasets.FashionMNIST(
        root = DATAROOT,
        train =True,
        transform = transform_train,
        download= True
    )

    #10,000 datapoints
    val_data = datasets.FashionMNIST(
        root = DATAROOT,
        train=False,
        download = True,
        transform = transform_val
    )


    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True
                                                , num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True
                                                ,num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False
                                                ,num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False
                                             ,num_workers=3)
    if get_datamat:
        return trainloader, valloader, train_data
    else:
        return trainloader, valloader


## CIFAR10 dataset
# 50,000 train points, 
# 10,000 validation points
def get_cifar10(use_cuda, batch_size, get_datamat = False):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #60,000 datapoints, 28x28
    train_data = datasets.CIFAR10(
        root = DATAROOT,
        train =True,
        transform = transform_train,
        download= True
    )

    #10,000 datapoints
    val_data = datasets.CIFAR10(
        root = DATAROOT,
        train=False,
        download = True,
        transform = transform_val
    )


    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True
                                                , num_workers=2)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True
                                                ,num_workers=2)

    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False
                                                ,num_workers=2)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False
                                             ,num_workers=2)
    if get_datamat:
        return trainloader, valloader, train_data
    else:
        return trainloader, valloader

## CIFAR100 dataset
# 50,000 train points, 
# 10,000 validation points
def get_cifar100(use_cuda, batch_size, get_datamat = False):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    #50,000 datapoints,
    train_data = datasets.CIFAR100(
        root = DATAROOT,
        train =True,
        transform = transform_train,
        download= True
    )

    #10,000 datapoints
    val_data = datasets.CIFAR100(
        root = DATAROOT,
        train=False,
        download = True,
        transform = transform_val
    )


    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True
                                                , num_workers=2)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True
                                                ,num_workers=2)

    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False
                                                ,num_workers=2)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False
                                             ,num_workers=2)
    if get_datamat:
        return trainloader, valloader, train_data
    else:
        return trainloader, valloader
