import copy
import torch
import numpy as np
import datasets
import fed_algos
import models
import train_nets
import utils
import kd 

############################################################
# Alternate method for computing teacher acc for Oneshot FL and
# FedKT
#############################################################

# Provide MODEL_LOC as directory of models saved from 1 round FedAvg
# and this will load the  models and do the inference rules for OneshotFL or FedKT
# (change mode = "teacher_oneshot_fl" or "teacher_oneshot_fl_cs" respectively)
# also need to specify seeds the fed Avg model was run for

MODEL_LOC = "./results/models/"

# compute predictions based on the majority vote of the client models
# this is for FedKT, assuming a single model is trained per client
def majority_vote_class_pred(x, method, num_classes, batch_size):
    pred_classes = torch.zeros((batch_size, method.num_clients))

    for c in range(method.num_clients):
        method.client_nets[c] = method.client_nets[c].eval()
        out_c = method.client_nets[c](x)

        _, pred_class = torch.max(out_c, 1)

        pred_classes[:, c] = pred_class 

    #get most common number along all clients for a datapoint
    preds,_ = torch.mode(pred_classes, -1)
    return preds#.cpu()

def test_classify_majority(testloader, method, num_classes):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(device)
            y = y.to(device)
            pred_class = majority_vote_class_pred(x, method, num_classes, y.shape[0])

            pred_class = pred_class.cpu()
            y = y.cpu() 

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc

use_cuda = torch.cuda.is_available()

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

num_clients = 5
num_rounds = 1
mode = "teacher_oneshot_fl_cs" #"teacher_oneshot_fl"

# specify seeds the fedavg was run for (need for loading models)
seeds = ["11", "12", "13", "14","15","16","17","18","19","20"]

for non_iid in [0.0, 0.3, 0.6, 0.9]:

    for dataset in ["mnist", "f_mnist", "emnist", "cifar10", "cifar100"]:
       
       # setup the oneshot FL or FedKT methods
        ################################
        # DATASET
        ################################
        task = "classify"

        #MNIST default
        inp_dim = 28*28
        net_type=  "fc"
        batch_size = 100

        if dataset == "mnist":
            trainloader, valloader, train_data  = datasets.get_mnist(use_cuda, batch_size, get_datamat=True)
            inp_dim = 28*28
        elif dataset == "cifar10":
            trainloader, valloader, train_data  = datasets.get_cifar10(use_cuda, batch_size, get_datamat=True)
            inp_dim = 32*32*3
            net_type = "cnn"
        elif dataset == "cifar100":
            trainloader, valloader, train_data  = datasets.get_cifar100(use_cuda,batch_size, get_datamat=True)
            inp_dim = 32*32*3
            net_type = "cnn"
        elif dataset == "emnist":
            trainloader, valloader, train_data  = datasets.get_emnist(use_cuda, batch_size, get_datamat=True)
            inp_dim = 28*28
        elif dataset == "f_mnist":
            trainloader, valloader, train_data  = datasets.get_fashion_mnist(use_cuda, batch_size, get_datamat=True)
            inp_dim = 28*28

        
        if task == "classify":
            out_dim = len(train_data.classes)
        
        if net_type == "fc":
            #set up network
            base_net = models.LinearNet(inp_dim=inp_dim, num_hidden = 100, out_dim=out_dim)
            base_net = base_net.to(device)
        elif net_type == "cnn":
            base_net = models.CNN(num_classes=out_dim)
            base_net = base_net.to(device)

        for seed in seeds:
            # load the FedAvg saved models
            exp_id = "{}_{}_{}_clients_{}_rounds_log_{}_noniid".format(dataset, mode, num_clients, num_rounds, non_iid)
            save_dir = "./results/"

            # these hyperparameters are irrelevant since local models are trained already
            sgd_hyperparams = { 'epoch_per_client': 10,
                                'lr': 1e-2,
                                'g_lr': 1e-2,
                                'batch_size': 100,
                                'optim_type': "adam",
                                'datasize': 60000,
                                'outdim': out_dim,
                                'seed': seed,
                                'model_save_dir': "./models_more/",
                                'exp_id': exp_id,
                                'save_dir': save_dir
                            }

            #setup OneShot FL method 
            if mode == "teacher_oneshot_fl":
                sgd_hyperparams['kd_lr'] = 1e-4
                sgd_hyperparams['kd_optim_type'] = "adam"
                sgd_hyperparams['kd_epochs'] = 100
                
                args = None

                method = fed_algos.ONESHOT_FL(num_clients = num_clients,
                                    base_net = base_net,
                                    traindata=train_data, distill_data = train_data,
                                    num_rounds = 1,
                                    hyperparams = sgd_hyperparams, device=device, logger = exp_id,
                                    args=args, non_iid = non_iid,
                                    task = task)
            
            # setup FedKT method
            elif mode == "teacher_oneshot_fl_cs":
                sgd_hyperparams['kd_lr'] = 1e-4
                sgd_hyperparams['kd_optim_type'] = "adam"
                sgd_hyperparams['kd_epochs'] = 100
                
                args = None

                method = fed_algos.ONESHOT_FL_CS(num_clients = num_clients,
                                    base_net = base_net,
                                    traindata=train_data, distill_data = train_data,
                                    num_rounds = 1,
                                    hyperparams = sgd_hyperparams, device=device, logger = exp_id,
                                    args=args, non_iid = non_iid,
                                    task = task)

            #load client models
            for client_num in range(method.num_clients):
                model_dir =  MODEL_LOC #"./results/models/"
                PATH = model_dir + dataset + "_fed_sgd_5_clients_1_rounds_sgdm_optim_log_{}_noniid_seed_".format(non_iid) +str(seed) + "_client_"+str(client_num)
                #print(PATH)
                method.client_nets[client_num].load_state_dict(torch.load(PATH))

            if mode == "teacher_oneshot_fl_cs":
                acc = test_classify_majority(valloader, method, out_dim)
            #teacher inference
            else:
                acc = method.test_acc(valloader)

            # write to pickle file in save_dir
            utils.write_result_dict_to_file(result = acc, seed = seed, 
                                        file_name = save_dir + exp_id)
