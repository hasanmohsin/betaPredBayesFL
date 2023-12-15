#!/bin/bash

mode=distill_f_mcmc

#which hyperparameters we are using, since this script evaluates DPredBayes, EP MCMC and FedPA (SR)
# the only change made is that for mnist, 0.0 noniid, lr=1e-1 for EP MCMC and fed_pa, while for
# DPredBayes it is 5e-1 
tuned_for=distill_f_mcmc



for non_iid in 0.0 0.3 0.6 0.9  
do

	if [ "$tuned_for" ==  "distill_f_mcmc" ]
	then
		if [ $non_iid == 0.0 ]
		then 
			lr_mnist=5e-1
		else
			lr_mnist=1e-1
		fi 
	else 
		lr_mnist=1e-1
	fi 

	for seed in 11 12 13 14 15 16 17 18 19 20 
	do
		printf "\nRunning $mode MNIST Experiment $seed: \n"
		python run_exp.py --mode $mode --dataset mnist --epoch_per_client 25 --lr $lr_mnist --num_round 1 --optim_type "sgdm" --non_iid $non_iid --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results_distill_f_mcmc/"


		printf "\nRunning $mode FMNIST Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset f_mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --non_iid $non_iid --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results_distill_f_mcmc/"



		printf "\nRunning $mode EMNIST Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset emnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --non_iid $non_iid --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results_distill_f_mcmc/"

	
		printf "\nRunning $mode CIFAR10 Experiment, seed $seed: \n"
        python run_exp.py --mode $mode --dataset cifar10 --epoch_per_client 50 --lr 1e-1 --num_round 1 --optim_type "sgdm" --non_iid $non_iid --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results_distill_f_mcmc/" --net_type cnn

        printf "\nRunning $mode CIFAR100 Experiment, seed $seed: \n"
        python run_exp.py --mode $mode --dataset cifar100 --epoch_per_client 50 --lr 1e-1 --num_round 1 --optim_type "sgdm" --non_iid $non_iid --seed $seed --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results_distill_f_mcmc/" --net_type cnn
	done
done

