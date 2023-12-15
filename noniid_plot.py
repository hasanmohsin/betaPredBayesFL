import matplotlib.pyplot as plt
import numpy as np
import torch 
import pickle
import copy

###################################
# Code for generating plots
###################################

# needs to have pickle files saved beforehand from experiments

# PICKLE_LOC needs to be set to location of pickle files

NUM_EXP = 10  #number of seeds experiments run for
type = "nllhd" # change to cal for ECE results
PICKLE_LOC = "./"


def get_res(mode, data_name, num_rounds, non_iid, type):

    if type == "cal":
        type_tag = "CAL"
    elif type == "nllhd":
        type_tag = "NLLHD"
    else:
        type_tag = "ACC"

    if mode == "fed_sgd":
        fname = PICKLE_LOC + "{}_{}_5_clients_{}_rounds_sgdm_optim_log_{}_noniid{}.pickle".format(data_name,mode, num_rounds, non_iid, type_tag)
    elif mode == "teacher_oneshot_fl" or mode == "teacher_oneshot_fl_cs":
        fname = PICKLE_LOC + "{}_{}_5_clients_{}_rounds_log_{}_noniid{}.pickle".format(data_name,mode, num_rounds, non_iid,type_tag)
    else:
        fname = PICKLE_LOC + "{}_{}_5_clients_{}_rounds_log_{}_noniid{}.pickle".format(data_name,mode, num_rounds, non_iid, type_tag)
       

    res = pickle.load(open(fname, "rb"))
    np_res = np.array([v for v in res.values()])
    return np_res


x_het = [0.0, 0.3, 0.6, 0.9]



for data_name in ["mnist", "f_mnist", "emnist", "cifar10", "cifar100"]:
    plt.figure()  

     
    for mode_base in ["tune_f_mcmc", "tune_distill_f_mcmc", "tune_product_f_mcmc", "tune_mixture_f_mcmc",  "tune_ep_mcmc", "fed_prox", "adapt_fl", "teacher_fed_be", "teacher_oneshot_fl", "tune_fed_pa", "fed_sgd_1_round"]:    
        mode_mean = []
        mode_stder = []

        for non_iid in [0.0, 0.3, 0.6, 0.9]:
            
            if mode_base == "fed_sgd" or mode_base == "fed_pa":
                if data_name in ["cifar10", "cifar100"]:
                    num_rounds = 10
                else:
                    num_rounds = 5

            else:
                num_rounds = 1

            if mode_base == "fed_pa_1_round":
                mode = "fed_pa"
            elif mode_base == "fed_sgd_1_round":
                mode = "fed_sgd"
            else:
                mode = mode_base


            np_res = get_res(mode, data_name, num_rounds, non_iid=non_iid, type=type)
            
            mean = np.mean(np_res[:NUM_EXP])
            n = len(np_res)
            stder = np.std(np_res)/np.sqrt(NUM_EXP)


            mode_mean.append(mean)
            mode_stder.append(stder)
            

        #plot line for this method
        if mode_base == "tune_f_mcmc":
            label = "Beta-PredBayes (SR)"
            marker = 'o'
            linestyle = ':'
        elif mode_base =="tune_distill_f_mcmc":
            label = "D Beta-PredBayes (SR)"
            marker = '*'
            linestyle = '-'
        elif mode_base == "tune_fed_pa":
            label = "FedPA (SR)"
            marker = '^'
            linestyle='--'
        elif mode_base == "tune_ep_mcmc":
            label = "EPMCMC (SR)"
            marker = 'v'
            linestyle = '-.'
        elif mode_base == "fed_sgd_1_round":
            label = "FedAvg (SR)"
            marker = 'p'
            linestyle =  (0, (5, 10)) #'loosely dashed'
        elif mode_base == "tune_mixture_f_mcmc":
            label = "Mixture PredBayes (SR)"
            marker = '^'
            linestyle= (0, (3, 5, 1, 5, 1, 5)) #dash dotdotted
        elif mode_base == "fed_sgd":
            label = "FedAvg (MR)"
            marker = 's'
            linestyle = (0, (3, 1, 1, 1)) #'densely dashdotted'
        elif mode_base == "tune_product_f_mcmc":
            label = "Product PredBayes (SR)"
            marker = '+'
            linestyle =  (0, (3, 10, 1, 10)) #'loosely dashdotted'
        elif mode_base == "teacher_oneshot_fl":
            label = "Oneshot FL (SR)"
            marker = 'x'
            linestyle =  (0, (5, 1)) #'densely dashed'
        elif mode_base == "fed_prox":
            label = "Fed Prox (SR)"
            marker = 's'
            linestyle =  (0, (5, 1)) #'densely dashed'
        elif mode_base == "adapt_fl":
            label = "Adapt FL (SR)"
            marker = '+'
            linestyle =  (0, (5, 1)) #'densely dashed'
        elif mode_base == "teacher_fed_be":
            label = "Fed BE (SR)"
            marker = 'v'
            linestyle =  (0, (5, 1)) #'densely dashed'

        print("X axis: ", x_het)
        print("Mode mean: ", mode_mean)
        plt.errorbar(x = x_het, y = mode_mean, yerr=mode_stder, label=label, marker= marker, linestyle=linestyle)
    
    if data_name == "mnist":
        graph_title = "MNIST"
    elif data_name == "f_mnist":
        graph_title = "FMNIST"
    elif data_name == "emnist":
        graph_title = "EMNIST"
    elif data_name == "cifar10":
        graph_title = "CIFAR10"
    elif data_name == "cifar100":
        graph_title = "CIFAR100"
    
    plt.title(graph_title)
    plt.xlabel("Data Heterogeneity, h", fontsize=14)
    
    if type == "acc":
        y_tag = "Test Accuracy"
    elif type == "cal":
        y_tag = "Expected Calibration Error"
    elif type == "nllhd":
        y_tag = "Negative Log Likelihood"
    plt.ylabel(y_tag, fontsize=14)
    #legend = plt.legend(loc='best', framealpha=1.0)

    plt.tight_layout()
    plt.savefig('../plots/{}_results/{}_noniid_{}.png'.format(type, data_name,type))


