Code for "One Round Federated Learning with Predictive Space Bayesian Inference"

To run experiments: do

python run_exp.py --mode MODE --data DATA --save_dir SAVE_DIR ...

Various arguments are specified in the run_exp.py file (They are hyperparameters, training length etc)
- MODE = 
    "fed_sgd": FedAvg
    "fed_pa": FedPA 
    "ep_mcmc": EP MCMC
    "fed_prox": FedProx
    "adapt_fl": Adaptive FL
    "fed_be": FedBE 
    "oneshot_fl": Oneshot FL 
    "oneshot_fl_cs": FedKT 
    "tune_distill_f_mcmc": betaPredBayes (ours) (distilled and undistilled + BCM + Mixture model + FedPA + EP MCMC on all same samples)
    

- DATA =
    For Classification: 
    "mnist" 
    "f_mnist"
    "emnist"
    "cifar10"
    "cifar100" 

    For Regression:
    "forest_fire"
    "bike"
    "real_estate"
    "airquality"
    "winequality"

Make sure to set "DATAROOT" in  datasets.py (at beginning of file)
Regression dataset xlsx, or csv are in the "Dataset" folder, downloaded from the UCI repository

Eg. to run FedAvg, for 1 round, on MNIST, can do:

python run_exp.py --mode "fed_sgd" --dataset mnist --epoch_per_client 25 --lr 1e-2 --num_round 1 --optim_type "sgdm" --non_iid 0.0 --seed 11 --save_dir "./results/"

To run D BetaPredBayes on MNIST, can do:
python run_exp.py --mode "tune_distill_f_mcmc" --dataset mnist --epoch_per_client 25 --lr 1e-1 --num_round 1 --optim_type "sgdm" --non_iid 0.0 --seed 11 --rho 0.4 --g_lr 1 --max_samples 6 --kd_lr 1e-4 --kd_optim_type "adam" --kd_epochs 100 --save_dir "./results/"


Example scripts for running the code with hyperparameters specified are given in the "./sample_scripts" folder 
- "mode_fedAvg_SR_client_5_exps.sh", for classification data - for FedAvg 1 round
- "mode_tuned_distill_f_mcmc_client_5_exps.sh", for classification data, requires mode as input - used to generate DBeta-PredBayes, Beta-PredBayes, BCM, Mixture, EP MCMC, FedPA (1 Round) results
    - can change "tuned_for" at start of script to use different hyperparameters
- "mode_fedPA_client_5_exps.sh", for classification data, for FedPA (note: reported results generated using tuned_distill_f_mcmc FedPA results, since they are on same samples)  
- "mode_oneshot_client_5_exps.sh" for classification data, for oneshot FL 
    - alternate way to get teacher oneshot FL (and teacher FedKT) results is to run fedavg_SR and then run python teacher_oneshot_fedkt_exp.py
- "mode_fedkt_client_5_exps.sh", for classification data - for FedKT 
- "mode_fedprox_client_5_exps.sh", for classification data - for FedProx 
- "mode_adaptfl_client_5_exps.sh", for classification data - for Adaptive FL 
- "mode_fed_be_client_5_exps.sh", for classification data - for FedBE  
- "regr_client_5_exps.sh": for regression data, requires dataset name as input, has regression hyperparameters set to tuned vals for each dataset

To use these scripts directly, must move to projects main directory first 

Results are:
1. Printed to standard output
2. Saved in a log file (in the save directory)
3. Saved in a pickle file (where runs over different seeds are saved separately, also in save directory)

Some methods (FedAvg, DPredBayes) also save their models for each client

Code for analysis includes:
- noniid_plot.py: for plotting the classification performance as h increases
    - needs pickle files, change PICKLE_LOC to location of pickle files generated from code
    - set type_tag to "cal" for ECE and "nllhd" for NLL plots
- process_results.py: for getting run averages + stder (and statistical test) for CLASSIFICATION
- process_results_regr.py: for gettting run averages + stder (and statistical test) for REGRESSION
    - needs pickle files, and change PICKLE_LOC to location of pickle files

- teacher_oneshot_fedkt_exp.py : used for getting teacher acc of OneshotFL and FedKT more quickly, if FedAvg 1 round has already been run and the models saved
                                - loads the models and performs the ensemble inference according to each technique, then saves results in pickle file

requirements.txt is included. Experiments were done on Python version 3.8.10, on a linux system, with cuda enabled (note: results weren't gathered with cpu ) 