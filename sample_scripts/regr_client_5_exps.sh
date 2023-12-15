#!/bin/bash

data=$1
save_dir="./results_REGR_tuned_distill/"
#SET HYPERPARAMS
if [ "$data" == "airquality" ]
then
	lr_mcmc=1e-1
	lr_sgd=1e-2
	lr_pa=1e-1

	lr_afl=1e-2

	g_lr_afl=1e-1
	tau=10e-3
	afl_optim_type="sgdm"

	lr_prox=1e-2

	prox_reg=1e-2
	prox_optim_type="sgdm"


	kd_lr=1e-3
	temp=1.0
	num_cycles=5
	max_samples=6
	sample_per_cycle=1
			
	rho=0.4
	g_lr_pa=1e-1
	pa_optim_type="sgdm"

	epochs=100 #for 1 round
	epochs_mult_round=20
	
elif [ "$data" == "winequality" ]
then 
	lr_mcmc=2e-1
	lr_sgd=1e-3
	lr_pa=5e-1

	lr_afl=1e-1

	g_lr_afl=1
	tau=10e-1
	afl_optim_type="sgd"

	lr_prox=1e-2

	prox_reg=1e-3
	prox_optim_type="adam"

	kd_lr=5e-3
	temp=0.05
	num_cycles=4
	max_samples=4
	sample_per_cycle=2
			
	rho=0.9
	g_lr_pa=5e-1
	pa_optim_type="sgdm"

	epochs=20 #for 1 round
	epochs_mult_round=4

elif [ "$data" == "forest_fire" ]
then
	lr_mcmc=1e-2
	lr_sgd=1e-4
	lr_pa=1e-2

	lr_afl=1e-2

	g_lr_afl=1
	tau=10e-3
	afl_optim_type="sgd"

	lr_prox=1e-1

	prox_reg=1e-1
	prox_optim_type="sgdm"

	kd_lr=5e-3
	temp=0.5
	num_cycles=2 #5
	max_samples=4 #6
	sample_per_cycle=2
			
	rho=0.4 #1.0
	g_lr_pa=1.0
	pa_optim_type="sgdm"

	epochs=10 #20 #for 1 round
	epochs_mult_round=6 #4

elif [ "$data" == "real_estate" ]
then
	lr_mcmc=2e-1
	lr_sgd=1e-2
	lr_pa=1e-2
	lr_prox=1e-1
	
	lr_afl=1e-1

	g_lr_afl=1e-1
	tau=10e-3
	afl_optim_type="sgd"

	prox_reg=1e-2
	prox_optim_type="adam"

	kd_lr=5e-3
	temp=0.5
	num_cycles=5
	max_samples=6
	sample_per_cycle=2
			
	rho=0.4
	g_lr_pa=1.0
	pa_optim_type="sgdm"

	epochs=20 #for 1 round
	epochs_mult_round=4


elif [ "$data" == "bike" ]
then 
	lr_mcmc=2e-1
	lr_sgd=1e-2
	lr_pa=5e-1

	lr_afl=1e-1

	g_lr_afl=1e-1
	tau=10e-2
	afl_optim_type="sgdm"

	lr_prox=1e-2

	prox_reg=1e-3
	prox_optim_type="adam"

	kd_lr=1e-3
	temp=-1
	num_cycles=5
	max_samples=4
	sample_per_cycle=2
			
	rho=1.0
	g_lr_pa=1e-2
	pa_optim_type="adam"

	epochs=20 #for 1 round
	epochs_mult_round=4


else
	#defaults
	lr_mcmc=1e-1
	lr_sgd=1e-2
	lr_pa=1e-1

	lr_afl=1e-2

	g_lr_afl=1e-1
	tau=10e-3
	afl_optim_type="sgdm"

	lr_prox=1e-2

	prox_reg=1e-2
	prox_optim_type="sgdm"

	kd_lr=5e-3
	temp=-1
	num_cycles=5
	max_samples=4
	sample_per_cycle=2
			
	rho=0.4
	g_lr_pa=1e-1
	pa_optim_type="sgdm"

	epochs=20 #for 1 round
	epochs_mult_round=4


fi

for seed in 1 #2 4 5 6 7 8 9 10
do
	printf "\nRunning NONIID Tuned Distill F MCMC Experiment: \n"
	python run_exp.py --mode tune_distill_f_mcmc --dataset $data --epoch_per_client $epochs --lr $lr_mcmc --num_round 1 --save_dir $save_dir --kd_lr $kd_lr --kd_epochs 100 --kd_optim_type "adam" --temp $temp --num_cycles $num_cycles --max_samples $max_samples --sample_per_cycle $sample_per_cycle --seed $seed --non_iid 1.0

    printf "\nRunning IID Tuned Distill F MCMC Experiment: \n"
	python run_exp.py --mode tune_distill_f_mcmc --dataset $data --epoch_per_client $epochs --lr $lr_mcmc --num_round 1 --save_dir $save_dir --kd_lr $kd_lr --kd_epochs 100 --kd_optim_type "adam" --temp $temp --num_cycles $num_cycles --max_samples $max_samples --sample_per_cycle $sample_per_cycle --seed $seed --non_iid 0.0

	###############
	printf "\nRunning NONIID OneShot FL Experiment 1 round: \n"
	python run_exp.py --mode oneshot_fl --dataset $data --epoch_per_client $epochs --lr $lr_sgd --num_round 1 --optim_type "sgdm" --save_dir $save_dir --kd_lr $kd_lr --kd_epochs 100 --kd_optim_type "adam" --seed $seed --non_iid 1.0

	printf "\nRunning IID OneShot FL Experiment 1 round: \n"
	python run_exp.py --mode oneshot_fl --dataset $data --epoch_per_client $epochs --lr $lr_sgd --num_round 1 --optim_type "sgdm" --save_dir $save_dir --kd_lr $kd_lr --kd_epochs 100 --kd_optim_type "adam" --seed $seed --non_iid 0.0
	###############

	
	################
	printf "\nRunning NONIID FedBE Experiment 1 round: \n"
	python run_exp.py --mode fed_be --dataset $data --epoch_per_client $epochs --lr $lr_sgd --num_round 1 --optim_type "sgdm" --save_dir $save_dir --kd_lr $kd_lr --kd_epochs 100 --kd_optim_type "adam" --seed $seed --non_iid 1.0

	printf "\nRunning IID FedBE Experiment 1 round: \n"
	python run_exp.py --mode fed_be --dataset $data --epoch_per_client $epochs --lr $lr_sgd --num_round 1 --optim_type "sgdm" --save_dir $save_dir --kd_lr $kd_lr --kd_epochs 100 --kd_optim_type "adam" --seed $seed --non_iid 0.0
	###############

done
