import pickle
import torch
import numpy as np
import copy 
from scipy.stats import wilcoxon

data_name = "mnist"

#process results for distill_f-mcmc
mode = "distill_f_mcmc"
NUM_EXP = 10
IID = 1.0 #0.0 - choose which iid to process
type = "nllhd" # choose between cal or nllhd

def get_res(mode, data_name, num_rounds, type):
    if type == "cal":
        type_tag = "CAL"
    elif type == "nllhd":
        type_tag = "NLLHD"
    else:
        type_tag = "ACC"

    if mode != "fed_sgd":
        fname = "./{}_{}_5_clients_{}_rounds_log_{}_noniid{}.pickle".format(data_name,mode, num_rounds, IID, type_tag)
    else:
        fname = "./{}_{}_5_clients_{}_rounds_adam_optim_log_{}_noniid{}.pickle".format(data_name,mode, num_rounds, IID, type_tag)

    res = pickle.load(open(fname, "rb"))
    np_res = np.array([v for v in res.values()])
    return np_res


for data_name in ["airquality", "bike", "winequality","real_estate", "forest_fire"]:
    
    base_res = get_res("tune_distill_f_mcmc", data_name, 1, type)
    print("\n")

    for mode_base in ["tune_ep_mcmc", "teacher_fed_be", "teacher_oneshot_fl", "tune_mixture_f_mcmc",  "tune_product_f_mcmc", "tune_f_mcmc", "tune_distill_f_mcmc"]:
        
        if mode_base == "fed_sgd" or mode_base == "fed_pa":
            num_rounds = 5
        else:
            num_rounds = 1

        if mode_base == "fed_pa_1_round":
            mode = "fed_pa"
        elif mode_base == "fed_sgd_1_round":
            mode = "fed_sgd"
        else:
            mode = mode_base

        np_res = get_res(mode, data_name, num_rounds, type)

        mean = np.mean(np_res[:NUM_EXP])
        std = np.std(np_res)
        n = len(np_res)
        std_error = std/np.sqrt(NUM_EXP)


        if mode == "tune_distill_f_mcmc":
            base_res = copy.copy(np_res)
            score_stat = "N/A"
            score_pval = "N/A"

            print("{} {} num rounds {} Mean: {:.4f}, std error: {:.4f}".format(data_name, mode_base, 
        num_rounds, mean, std_error))
        else:
           
            score = wilcoxon(x= base_res[:NUM_EXP], y=np_res[:NUM_EXP])
            
            score_stat = score.statistic
            score_pval = score.pvalue
        
            print("{} {} num rounds {} Mean: {:.3f}, std error: {:.3f}, stat_score: {:.3f}, p_value: {:.3f}".format(data_name, mode_base, 
        num_rounds, mean, std_error, score_stat, score_pval))



   