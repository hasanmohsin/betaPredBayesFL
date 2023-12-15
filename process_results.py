import pickle
import numpy as np
import copy 
from scipy.stats import wilcoxon

#######################################################
# Code for getting run avg and stder + statistical test
#######################################################

# Must set PICKLE_LOC to location of pickle files

NUM_EXP = 10 
type = "cal" #change between cal or nllhd
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



for non_iid in [0.0, 0.3, 0.6, 0.9]:
    print("\n\n\n")
    print("NONIID param: ", non_iid)
    for data_name in ["mnist","f_mnist","emnist","cifar10", "cifar100"]:
        print("\n DATASET: ", data_name)
        
        #base to compare with (DBPredBayes)
        base_res = get_res("tune_distill_f_mcmc", data_name, 1, non_iid, type)
        m = len(base_res)

                          
        for mode_base in  ["tune_f_mcmc", "tune_distill_f_mcmc", "tune_product_f_mcmc", "tune_mixture_f_mcmc",  "tune_ep_mcmc", "fed_prox", "adapt_fl", "teacher_fed_be", "teacher_oneshot_fl", "tune_fed_pa", "fed_sgd_1_round"]:
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

            if mode_base == "tune_distill_f_mcmc":
                base_res = copy.copy(np_res)
                score_stat = "N/A"
                score_pval = "N/A"

                print("{} {} num rounds {} Mean: {:.3f}, std error: {:.3f}".format(data_name, mode_base, 
                    num_rounds, mean, stder))
            else:
                score = wilcoxon(x= base_res[:min(min(NUM_EXP, n),m)], y=np_res[:min(min(NUM_EXP, n),m)])
                
                score_stat = score.statistic
                score_pval = score.pvalue
        
                print("{} {} num rounds {} Mean: {:.2f}, std error: {:.2f}, stat_score: {:.2f}, p_value: {:.5f}".format(data_name, mode_base, 
                num_rounds, mean, stder, score_stat, score_pval))
        