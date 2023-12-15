import copy
import torch
import numpy as np
import datasets
import train_nets
import utils
import kd 
import itertools
import models

##############################################################
# IMPLEMENTATIONS FOR THE FL ALGORITHMS (BASELINES + OURS)
################################################################

#implements FedAvg (McMahan et al., 2017)
class FedAvg:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, logger, non_iid = 0.0, task = "classify"):

        # lr for SGD
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.datasize = hyperparams['datasize']
        self.optim_type = hyperparams['optim_type']
        self.outdim = hyperparams['outdim']
        self.device = hyperparams['device']
        self.seed = hyperparams['seed']
        self.model_save_dir = hyperparams['model_save_dir']
        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        self.logger = logger

        self.all_data = traindata

        self.num_clients = num_clients

        self.task = task

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                 self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr = self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))

        self.num_rounds = num_rounds

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net = self.client_nets[client_num],
                                 optimizer=self.optimizers[client_num],
                                 criterion = self.criterion,
                                 trainloader=c_dataloader, device = self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

        return

    #in aggregation step - average all models
    def alt_aggregate(self):
        global_state_dict = self.global_net.state_dict()

        for layer in global_state_dict:
            global_state_dict[layer] = 0*global_state_dict[layer]

            #average over clients    
            for c in range(self.num_clients):
                global_state_dict[layer] += self.client_nets[c].state_dict()[layer]/self.num_clients

        self.global_net.load_state_dict(global_state_dict)

        return

    def aggregate(self):
        #in aggregation step - average all models

        c_vectors = []

        #average over clients    
        for c in range(self.num_clients):
                c_vector = torch.nn.utils.parameters_to_vector(self.client_nets[c].parameters()).detach()
                c_vectors.append(torch.clone(c_vector))
        c_vectors = torch.stack(c_vectors, dim=0)
        global_v = torch.mean(c_vectors, dim=0)
        
        #load into global net
        torch.nn.utils.vector_to_parameters(global_v, self.global_net.parameters())

        return

    def global_update_step(self):
        local_infos = []
        
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_update_step(client_num)

        self.aggregate()
    
    
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_nets[c].load_state_dict(self.global_net.state_dict())

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            #last round, save models before sending to clients
            if i == self.num_rounds - 1:
                self.save_models()

            self.global_to_clients()
            acc = self.get_acc(self.global_net, valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i+1, acc), self.logger)
        
        
        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id, type="acc")
        
        return
    
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            #load client models
            path =  "./results/models/" + self.dataset + "_fed_sgd_5_clients_{}_rounds_sgdm_optim_log_{}_noniid_seed_".format(self.num_rounds, self.non_iid) +str(self.seed) + "_client_"+str(client_num) 

            weight_dict = torch.load(path)
            self.client_nets[client_num].load_state_dict(weight_dict)

        #aggregate as usual
        self.aggregate()

    # to train starting from saved models
    def train_saved_models(self, valloader):
        #load and aggregate trained models 
        self.global_update_step_trained_clients()

        acc = self.get_acc(self.global_net, valloader)
        
        nllhd, cal_error = utils.test_calibration(model = self.global_net, testloader=valloader, task=self.task, 
                                                      device = self.device, model_type= "single")

        utils.print_and_log("\nGlobal rounds completed: {}, FedAvg test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(self.num_rounds, acc, nllhd, cal_error), self.logger)

        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id, type="acc")
        utils.write_result_dict_to_file(result = nllhd, seed = self.seed, file_name = self.save_dir + self.exp_id, type="nllhd")
        utils.write_result_dict_to_file(result = cal_error, seed = self.seed, file_name = self.save_dir + self.exp_id, type="cal")



# implements FedProx from (Li et al., 2020a)
class FedProx:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, logger, non_iid = 0.0, task = "classify"):

        # lr for SGD
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.datasize = hyperparams['datasize']
        self.optim_type = hyperparams['optim_type']
        self.outdim = hyperparams['outdim']
        self.device = hyperparams['device']
        self.seed = hyperparams['seed']
        self.model_save_dir = hyperparams['model_save_dir']
        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        self.reg_global = hyperparams['reg_global']

        self.logger = logger

        self.all_data = traindata

        self.num_clients = num_clients

        self.task = task

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                 self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr = self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))

        self.num_rounds = num_rounds

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        #initial global net is same as client nets
        self.global_net = copy.deepcopy(base_net)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        #take optimization step regularized by global net
        self.client_nets[client_num], loss = train_nets.sgd_prox_train_step(net = self.client_nets[client_num], 
                                g_net = self.global_net,
                                reg = self.reg_global,
                                optimizer=self.optimizers[client_num],
                                criterion = self.criterion,
                                trainloader=c_dataloader, device = self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

        return

    #in aggregation step - average all models
    def alt_aggregate(self):
        global_state_dict = self.global_net.state_dict()

        for layer in global_state_dict:
            global_state_dict[layer] = 0*global_state_dict[layer]

            #average over clients    
            for c in range(self.num_clients):
                global_state_dict[layer] += self.client_nets[c].state_dict()[layer]/self.num_clients

        self.global_net.load_state_dict(global_state_dict)

        return

    #updates global net
    def aggregate(self):
        #in aggregation step - average all models
       
        c_vectors = []

        #average over clients    
        for c in range(self.num_clients):
                c_vector = torch.nn.utils.parameters_to_vector(self.client_nets[c].parameters()).detach()
                c_vectors.append(torch.clone(c_vector))
        c_vectors = torch.stack(c_vectors, dim=0)
        global_v = torch.mean(c_vectors, dim=0)
        
        #load into global net
        torch.nn.utils.vector_to_parameters(global_v, self.global_net.parameters())

        return

    def global_update_step(self):
        local_infos = []
        
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_update_step(client_num)

        self.aggregate()
        
      
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_nets[c].load_state_dict(self.global_net.state_dict())

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            #last round, save models before sending to clients
            if i == self.num_rounds - 1:
                self.save_models()

            self.global_to_clients()
            acc = self.get_acc(self.global_net, valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i+1, acc), self.logger)
        
        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)
        
        return
    
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            #load client models
            path =  "./results/models/" + self.dataset + "_fed_prox_5_clients_{}_rounds_log_{}_noniid_seed_".format(self.num_rounds, self.non_iid) +str(self.seed) + "_client_"+str(client_num) 

            weight_dict = torch.load(path)
            self.client_nets[client_num].load_state_dict(weight_dict)
            
        #aggregate as usual
        self.aggregate()

    def train_saved_models(self, valloader):
        #load and aggregate trained models 
        self.global_update_step_trained_clients()

        acc = self.get_acc(self.global_net, valloader)
        
        nllhd, cal_error = utils.test_calibration(model = self.global_net, testloader=valloader, task=self.task, 
                                                      device = self.device, model_type= "single")

        utils.print_and_log("\nGlobal rounds completed: {}, Fed Prox test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(self.num_rounds, acc, nllhd, cal_error), self.logger)

        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id, type="acc")
        utils.write_result_dict_to_file(result = nllhd, seed = self.seed, file_name = self.save_dir + self.exp_id, type="nllhd")
        utils.write_result_dict_to_file(result = cal_error, seed = self.seed, file_name = self.save_dir + self.exp_id, type="cal")

# implements algorithm from "Adaptive Federated Optimization" (Reddi et al., 2021)
# specifically, the FedYOGI variant
class AdaptiveFL:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, logger, non_iid = 0.0, task = "classify"):

        # lr for SGD
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.datasize = hyperparams['datasize']
        self.optim_type = hyperparams['optim_type']
        self.outdim = hyperparams['outdim']
        self.device = hyperparams['device']
        self.seed = hyperparams['seed']
        self.model_save_dir = hyperparams['model_save_dir']
        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        self.logger = logger

        self.all_data = traindata

        self.num_clients = num_clients

        self.task = task

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                 self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr = self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))

        self.num_rounds = num_rounds

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.g_vec = torch.nn.utils.parameters_to_vector(self.global_net.parameters()).detach()
        
        # parameters for server optimization
        
        self.g_v = torch.zeros_like(self.g_vec)
        self.g_momentum = torch.zeros_like(self.g_vec)

        self.tau = hyperparams['tau'] #10e-3
        self.g_beta1 = 0.9
        self.g_beta2 = 0.99

        self.global_optimizer = torch.optim.Adam(self.global_net.parameters(), lr = self.g_lr)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net = self.client_nets[client_num],
                                 optimizer=self.optimizers[client_num],
                                 criterion = self.criterion,
                                 trainloader=c_dataloader, device = self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

        return
    
    #set delta as the gradient in the global optimizer
    def global_opt(self, g_delta):
        self.global_optimizer.zero_grad()

        #set global optimizer grad
        torch.nn.utils.vector_to_parameters(g_delta, self.delta.parameters())

        #copy gradient data over to global net
        for p, g in zip(self.global_net.parameters(), self.delta.parameters()):
            p.grad = g.data

        #update
        self.global_optimizer.step()

        return 
    
    def fed_yogi_update(self, g_delta):
        self.g_momentum = self.g_beta1 * self.g_momentum + (1-self.g_beta1) * g_delta
        self.g_v = self.g_v - (1 - self.g_beta2) * (g_delta**2) * torch.sign( self.g_v - g_delta**2 )       
    
        self.g_vec = torch.nn.utils.parameters_to_vector(self.global_net.parameters()).detach()
        self.g_vec = self.g_vec + self.g_lr * self.g_momentum/( torch.sqrt(self.g_v) + self.tau )
        
        #save to global net
        torch.nn.utils.vector_to_parameters(self.g_vec, self.global_net.parameters())

        return

    def aggregate(self):
        #in aggregation step - average all models

        c_vectors = []

        #average over clients    
        for c in range(self.num_clients):
                c_vector = torch.nn.utils.parameters_to_vector(self.client_nets[c].parameters()).detach()
                c_vectors.append(torch.clone(c_vector))
        c_vectors = torch.stack(c_vectors, dim=0)
        global_vec = torch.nn.utils.parameters_to_vector(self.global_net.parameters()).detach()
        delta = torch.mean(c_vectors, dim=0) - global_vec
        
        self.fed_yogi_update(g_delta = delta)

        return

    def global_update_step(self):
        local_infos = []
        
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_update_step(client_num)

        self.aggregate()
        
    
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_nets[c].load_state_dict(self.global_net.state_dict())

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            #last round, save models before sending to clients
            if i == self.num_rounds - 1:
                self.save_models()

            self.global_to_clients()
            acc = self.get_acc(self.global_net, valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i+1, acc), self.logger)
        
        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)
        
        return

    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            #load client models
            path =  "./results/models/" + self.dataset + "_adapt_fl_5_clients_{}_rounds_log_{}_noniid_seed_".format(self.num_rounds, self.non_iid) +str(self.seed) + "_client_"+str(client_num) 

            weight_dict = torch.load(path)
            self.client_nets[client_num].load_state_dict(weight_dict)
            
        #aggregate as usual
        self.aggregate()

    def train_saved_models(self, valloader):
        #load and aggregate trained models 
        self.global_update_step_trained_clients()

        acc = self.get_acc(self.global_net, valloader)
        
        nllhd, cal_error = utils.test_calibration(model = self.global_net, testloader=valloader, task=self.task, 
                                                      device = self.device, model_type= "single")

        utils.print_and_log("\nGlobal rounds completed: {}, Adapt FL test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(self.num_rounds, acc, nllhd, cal_error), self.logger)

        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id, type="acc")
        utils.write_result_dict_to_file(result = nllhd, seed = self.seed, file_name = self.save_dir + self.exp_id, type="nllhd")
        utils.write_result_dict_to_file(result = cal_error, seed = self.seed, file_name = self.save_dir + self.exp_id, type="cal")


# EP MCMC 
class EP_MCMC:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, 
                hyperparams, device, logger, non_iid = 0.0, task = "classify"):
        self.logger = logger
        self.all_data = traindata

        self.seed = hyperparams['seed']
        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        self.device = device

        self.num_clients = num_clients

        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.seed = hyperparams['seed']
        self.exp_id = hyperparams['exp_id']
    
        self.model_save_dir = hyperparams['model_save_dir']
        self.save_dir = hyperparams['save_dir']

        #initialize nets and data for all clients
        self.client_train = []
        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim = self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        self.max_samples = hyperparams['max_samples']

        for c in range(num_clients):
            hyperparams_c = copy.deepcopy(hyperparams)
            
            self.client_train.append(train_nets.cSGHMC(copy.deepcopy(base_net), 
                                                        trainloader=self.client_dataloaders[c],
                                                        device = device, task = task, hyperparams=hyperparams_c))

        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        
        self.global_train = train_nets.cSGHMC(copy.deepcopy(base_net), 
                                            trainloader =None,
                                            device = device, task = task, hyperparams=hyperparams)#copy.deepcopy(base_net)
        self.base_net = base_net
        self.num_g_samples = 10


    def save_models(self):
        save_dir = self.model_save_dir
        save_name = utils.change_exp_id(self.exp_id, "distill_f_mcmc", "mcmc") + "_seed_" + str(self.seed)  
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            cpath = c_save + "_client_" + str(c)  

            for idx, weight_dict in enumerate(self.client_train[c].sampled_nets):
                path = cpath + "_sample_" + str(idx) 
                torch.save(weight_dict, path)


    def local_train(self, client_num):
        #trains for above specified epochs per client 
        self.client_train[client_num].train()

    def get_client_samples_as_vec(self, client_num):
        client_samples = self.client_train[client_num].sampled_nets
        c_vectors = []
        for sample in client_samples:
            sample_net = copy.deepcopy(self.base_net)
            sample_net.load_state_dict(sample)
            c_vec = torch.nn.utils.parameters_to_vector(sample_net.parameters())
            c_vectors.append(c_vec)
        c_vectors = torch.stack(c_vectors, dim=0)

        return c_vectors

    def get_client_sample_mean_cov(self, client_num):
        #a list of sampled nets from training client
        c_vectors = self.get_client_samples_as_vec(client_num)
        mean = torch.mean(c_vectors, dim=0)
        
        #too memory intensive - approximate with diagonal matrix
        #cov = torch.Tensor(np.cov((c_vectors).detach().numpy().T))

        cov = torch.var(c_vectors, dim = 0)#.diag()
        return mean, cov

    def aggregate(self):
        #in aggregation step - average all models
       
        global_prec = 0.0
        global_mean = 0.0

        #average over clients    
        for c in range(self.num_clients):
            mean, cov = self.get_client_sample_mean_cov(c)
            client_prec = 1/cov #torch.inv(cov)
            global_prec += client_prec
            global_mean += client_prec * mean #client_prec@mean
        
        global_mean = (1/global_prec) * global_mean #torch.inv(global_prec) @ global_mean
        global_var = (1/global_prec)

        dist = torch.distributions.Normal(global_mean, global_var.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)

        global_samples = dist.sample([self.num_g_samples])
        
      
        self.global_train.sampled_nets = []
        
        for s in range(self.num_g_samples):
            sample = global_samples[s,0,:]

            #load into global net
            torch.nn.utils.vector_to_parameters(sample, self.global_train.net.parameters())
            self.global_train.sampled_nets.append(copy.deepcopy(self.global_train.net.state_dict()))

        return

    def global_update_step(self):
        for client_num in range(self.num_clients):
            self.local_train(client_num)

        self.aggregate()
    
    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            acc = self.global_train.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

# BCM
#Implements PredBayes (ours) (no distillation step)
#similar to EP_MCMC, but inference step is different
# main functions are train(), predict_classify and predict_regr
class F_MCMC(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

    #do nothing in aggregate function
    def aggregate(self):
        return
    
    #prediction on input x
    def predict_classify(self, x):
        global_pred = 1.0
        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)

            #assuming a uniform posterior
            global_pred *= pred
        return global_pred/torch.sum(global_pred, dim=-1, keepdims=True)

    def predict_regr(self,x):
        global_pred = 0.0
        var_sum = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)
            pred_var = torch.var(pred_list, dim = 0, keepdims = False)

            #assuming a uniform posterior
            global_pred += pred_mean/pred_var
            var_sum += 1/pred_var
        
        return global_pred/var_sum


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)
            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
    
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)


    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            acc = self.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return



# Implements Distilled BCM
# same as F_MCMC but an additional distillation step is added
# Also runs evaluation for EP MCMC and FedPA (1 round) using the same samples
class F_MCMC_distill(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, distill_data,
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

        #make distillation dataset out of distill_data
        distill_loader = torch.utils.data.DataLoader(distill_data, 
                                                batch_size=self.batch_size, shuffle=True, 
                                                pin_memory=True)

        self.student = copy.copy(base_net)
        self.kd_optim_type = hyperparams['kd_optim_type']
        self.kd_lr = hyperparams['kd_lr']
        self.kd_epochs = hyperparams['kd_epochs']

        self.distill = kd.KD(teacher = self, 
                             student = self.student, lr = self.kd_lr,#5e-3
                             device = self.device,
                             train_loader = distill_loader,
                             kd_optim_type = self.kd_optim_type
                            )

        self.fed_pa_trainer = FedPA(num_clients = num_clients, base_net = base_net, 
                                    traindata = traindata, 
                                    num_rounds = 1, 
                                    hyperparams = hyperparams, logger=None, non_iid = non_iid, task = self.task,
                                    device= self.device)

    #do nothing in aggregate function
    def aggregate(self):
        #try better student init
        self.distill.set_student(self.client_train[0].sampled_nets[-1])

        #train the student via kd
        self.distill.train(num_epochs = self.kd_epochs) #kd_epochs = 50
        self.student = self.distill.student

        return
    
    #so we can compare with EP MCMC 
    def ep_mcmc_aggregate(self):
        #in aggregation step - average all models
        #global_v = 0.0 #torch.nn.utils.paramters_to_vector(self.global_net.parameters())

        global_prec = 0.0
        global_mean = 0.0

        #average over clients    
        for c in range(self.num_clients):
            mean, cov = self.get_client_sample_mean_cov(c)
            client_prec = 1 #/self.num_clients #1/cov #torch.inv(cov) ########################################
            global_prec += client_prec
            global_mean += client_prec * mean #client_prec@mean
            #print("Client {} mean {}".format(c, mean))

        global_mean = (1/global_prec) * global_mean #torch.inv(global_prec) @ global_mean
        global_var = (1/global_prec)*torch.ones_like(global_mean).to(self.device)

        dist = torch.distributions.Normal(global_mean, global_var.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)
        #dist = torch.distributions.MultivariateNormal(loc = global_mean, precision_matrix=global_prec)
        global_samples = dist.sample([self.num_g_samples])
        
        #print("global mean shape: ", global_mean.shape)
        #print("global var shape: ", global_var.shape)
        #print("global samples shape: ", global_samples.shape)
    
        self.global_train.sampled_nets = []
        
        for s in range(self.num_g_samples):
            sample = global_mean #global_samples[s,0,:] #######################################

            #load into global net
            torch.nn.utils.vector_to_parameters(sample, self.global_train.net.parameters())
            self.global_train.sampled_nets.append(copy.deepcopy(self.global_train.net.state_dict()))

        return

    #prediction on input x
    def predict_classify(self, x):
        global_pred = 1.0
        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred *= pred
        return global_pred/torch.sum(global_pred, dim=-1, keepdims=True)


    def predict_regr(self,x):
        global_pred = 0.0
        var_sum = 0.0

        #pred_list_prior = self.client_train[c].ensemble_inf(x,)

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
            
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)

            #the variance in the prediction is 1.0 (the homeoscedatic assumed variance) + bayesian variance
            pred_var = 1.0 + torch.var(pred_list, dim = 0, keepdims = False)

            #assuming a uniform posterior
            global_pred += pred_mean/pred_var
            var_sum += 1/pred_var

        return global_pred/var_sum


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)
            _, pred_class = torch.max(pred, 1)    

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
    
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)
        

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            acc = self.distill.test_acc(valloader)
          
            utils.print_and_log("Global rounds completed: {}, distilled_f_mcmc test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)

        #evaluate samples on other methods (FMCMC, and EP MCMC)
        f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, f_mcmc test_acc: {}".format(i, f_mcmc_acc), self.logger)
        
        #save to dict
        utils.write_result_dict_to_file(result=f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"))

        #compute ep mcmc result and store in global_train
        self.ep_mcmc_aggregate()
        ep_mcmc_acc = self.global_train.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, ep_mcmc test_acc: {}".format(i, ep_mcmc_acc), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=ep_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"))


        #for fed_pa 1 round results
        #copy over trained clients
        self.fed_pa_trainer.client_train = copy.deepcopy(self.client_train)

        #take step of FedPA 
        self.fed_pa_trainer.global_update_step_trained_clients()
        fed_pa_acc = self.fed_pa_trainer.get_acc(self.fed_pa_trainer.global_train.net, valloader)
        utils.print_and_log("Global rounds completed: {}, fed_pa test_acc: {}".format(i, fed_pa_acc), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=fed_pa_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"))


        #save client sample models 
        self.save_models()

        return



#Implements Federated posterior averaging from (Al-Shedivat et al., 2021)
# based specifically on covariance estimation from supplementary of that paper
#Federated posterior averaging
class FedPA(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, 
                hyperparams, device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams,
                 device, logger, non_iid, task)

        self.rho = hyperparams['rho'] # 1.0
        self.global_lr = hyperparams['global_lr'] # global learning rate should likely be higher
        self.global_train.net.requires_grad = True

        self.g_optim_type = hyperparams['optim_type']

        if self.g_optim_type == "sgdm":
            self.global_optimizer = torch.optim.SGD(self.global_train.net.parameters(), lr = self.global_lr, momentum=0.9)
        elif self.g_optim_type == "sgd":
            self.global_optimizer = torch.optim.SGD(self.global_train.net.parameters(), lr = self.global_lr)
        elif self.g_optim_type == "adam":
            self.global_optimizer = torch.optim.Adam(self.global_train.net.parameters(), lr = self.global_lr)
        
        #self.global_optimizer = torch.optim.SGD(params=self.global_train.net.parameters(), lr = self.global_lr, momentum=0.9)
        self.seed = hyperparams['seed']

    def get_global_vec(self):

        g_vec = torch.nn.utils.parameters_to_vector(self.global_train.net.parameters())
        return g_vec

    #compute local delta for client_num, based on its samples
    def local_delta(self, client_num):
        #client samples as tensor N x num_params
        c_vectors = self.get_client_samples_as_vec(client_num)
        global_vec = self.get_global_vec()

        #first compute sample means, 
        num_samples = c_vectors.shape[0]
        
        #initialize
        delta_sim = global_vec - c_vectors[0,:]
        rhot = 1
        u = c_vectors[0,:]
        u_vecs = torch.clone(u).reshape(1, -1)

        if num_samples == 1:
            return delta_sim/rhot

        v = c_vectors[1,:] - c_vectors[0,:] #assuming at least 2 samples
        v_vecs = torch.clone(v).reshape(1,-1)

        #u = c_vectors - sample_means
        for t in range(c_vectors.shape[0]):
            if t == 0:
                # from second sample onwards
                continue 
                #sample_mean = torch.zeros(c_vectors[0,:].shape)#c_vectors[0, :]
            else:
                sample_mean = torch.mean(c_vectors[:t, :], dim = 0)

            u = (c_vectors[t,:] - sample_mean)
            u_vecs = torch.cat([u_vecs, u.reshape(1,-1)], dim=0)

            v_1_t = u
            v = v_1_t
            #compute v_(t-1)_t
            for k in range(1, t):
                gamma_k = self.rho * k/(k+1)
                num =  gamma_k *(torch.dot(v_vecs[k, :], u)) * v_vecs[k,:]
                den = 1 + gamma_k * (torch.dot(v_vecs[k,:], u_vecs[k,:]))
                v -= num/den 
            v_vecs = torch.cat([v_vecs, v.reshape(1, -1)], dim=0)

            #update delta
            uv = torch.dot(u, v)
            gamma_t = self.rho * (num_samples-1)/num_samples

            diff_fact_num =  gamma_t*(num_samples*torch.dot(u, delta_sim) - uv)
            diff_fact_den = 1+gamma_t*(uv)

            delta_sim = delta_sim - (1+diff_fact_num/diff_fact_den)*v/num_samples
        
        rhot = 1/(1+(num_samples - 1)*self.rho)
        return delta_sim/rhot

    def global_opt(self, g_delta):
        self.global_optimizer.zero_grad()

        #set global optimizer grad
        torch.nn.utils.vector_to_parameters(g_delta, self.base_net.parameters())

        #copy gradient data over to global net
        for p, g in zip(self.global_train.net.parameters(), self.base_net.parameters()):
            p.grad = g.data

        #update
        self.global_optimizer.step()

        return 

    def global_update_step(self):
        deltas = []

        #train client models/sample
        for client_num in range(self.num_clients):
            self.local_train(client_num)
            delta = self.local_delta(client_num)
            deltas.append(delta)
        deltas = torch.stack(deltas, dim=0)
        
        #global gradient
        g_delta = torch.mean(deltas, dim= 0)
        
        #take optimizer step in direction of g_delta for global net
        self.global_opt(g_delta)
        return 
    
    #when clients are already trained
    def global_update_step_trained_clients(self):
        deltas = []

        #train client models/sample
        for client_num in range(self.num_clients):
            delta = self.local_delta(client_num)
            deltas.append(delta)
        deltas = torch.stack(deltas, dim=0)
        
        #global gradient
        g_delta = torch.mean(deltas, dim= 0)
        
        #take optimizer step in direction of g_delta for global net
        self.global_opt(g_delta)
        return 
        
    
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_train[c].net.load_state_dict(self.global_train.net.state_dict())
    
    def get_acc(self, net, valloader):
        print("Task ", self.task)
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)


    def train(self, valloader):
        acc = self.get_acc(self.global_train.net, valloader)
        
        for i in range(self.num_rounds):
            self.global_update_step()
            self.global_to_clients()
            acc = self.get_acc(self.global_train.net, valloader) #self.global_train.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

        #for reference, check client accuracies
        for c in range(self.num_clients):
            acc_c = self.client_train[c].test_acc(valloader)
            utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return
    
    def load_clients(self):
        for client_num in range(self.num_clients):
            model_PATH =  "./results_distill_f_mcmc/models/" + self.dataset + "_mcmc_5_clients_1_rounds_log_{}_noniid_seed_".format(self.non_iid) +str(self.seed) + "_client_"+str(client_num) 

            # there should be 6 samples for each dataset
            for idx in range(self.max_samples): 
                path = model_PATH + "_sample_" + str(idx) 
                weight_dict = torch.load(path)
                self.client_train[client_num].sampled_nets.append(weight_dict)

    def train_saved_models(self, valloader):
        #load and aggregate trained models 
        self.load_clients()
        self.global_update_step_trained_clients()
        self.global_to_clients()
       
        acc = self.get_acc(self.global_train.net, valloader)
        
        nllhd, cal_error = utils.test_calibration(model = self.global_train.net, testloader=valloader, task=self.task, 
                                                      device = self.device, model_type= "single")

        utils.print_and_log("\nGlobal rounds completed: {}, Fed Prox test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(self.num_rounds, acc, nllhd, cal_error), self.logger)

        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id, type="acc")
        utils.write_result_dict_to_file(result = nllhd, seed = self.seed, file_name = self.save_dir + self.exp_id, type="nllhd")
        utils.write_result_dict_to_file(result = cal_error, seed = self.seed, file_name = self.save_dir + self.exp_id, type="cal")

#Implements FedKT from (Li et al., 2021)
# Only applicable to classification setting
# can train 2 or 1 models per client for the ensemble
class ONESHOT_FL_CS:
    def __init__(self, num_clients, base_net,
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.args = args
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']

        self.model_save_dir = hyperparams['model_save_dir']

        # initialize nets and data for all clients
        self.client_nets = []
        self.client_nets2 = []
        self.optimizers = []
        self.optimizers2 = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            self.client_nets2.append(copy.deepcopy(base_net))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
                self.optimizers2.append(torch.optim.SGD(self.client_nets2[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
                self.optimizers2.append(torch.optim.SGD(self.client_nets2[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
                self.optimizers2.append(torch.optim.Adam(self.client_nets2[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
                self.optimizers2.append(torch.optim.SGD(self.client_nets2[c].parameters(), lr=self.lr))

        self.num_rounds = num_rounds

        self.task = task

        #select between training:
        #  2 models per client (allowing for consistent voting)
        # or 1 model per client
        self.onemodel = True #False
        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )

    def local_train(self, client_num, first_loaded = False):
        c_dataloader = self.client_dataloaders[client_num]

        if self.onemodel == False:
            self.client_nets2[client_num], loss = train_nets.sgd_train_step(net=self.client_nets2[client_num],
                                                                       optimizer=self.optimizers2[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)

            print("Client {}, Loss: {}".format(client_num, loss))

        #in case the first model is not loaded
        if not first_loaded:
            self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                            optimizer=self.optimizers[client_num],
                                                                            criterion=self.criterion,
                                                                        trainloader=c_dataloader, device=self.device)


    # prediction on input x
    def predict_classify(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred_logit = self.client_nets[c](x)
            client_pred.append(pred_logit.view(pred_logit.size(0), -1).max(dim=-1).indices)
            if self.onemodel == False:
                self.client_nets2[c] = self.client_nets2[c].eval()
                pred_logit2 = self.client_nets2[c](x)
                client_pred.append(pred_logit2.view(pred_logit2.size(0), -1).max(dim=-1).indices)
        pred_class = torch.mode(torch.transpose(torch.stack(client_pred), 0, 1)).values
        pred_dist = torch.zeros(pred_class.size(0), pred_logit.size(1))
        for i in range(x.size(0)):
            pred_dist[i][pred_class[i]] = 1
        return pred_dist.to(self.device)

    def predict_regr(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred = self.client_nets[c](x)
            self.client_nets2[c] = self.client_nets2[c].eval()
            pred2 = self.client_nets2[c](x)
            client_pred.append(pred)
            client_pred.append(pred2)

        return torch.mean(torch.stack(client_pred), dim=0, keepdims=False).to(self.device)


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)
    def aggregate(self):
        self.distill.train(num_epochs = self.kdepoch)
        self.student = self.distill.student
        return

    def global_update_step(self):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)
        self.aggregate()

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            PATH =  "./results/models/" + self.args.dataset + "_fed_be_5_clients_1_rounds_log_{}_noniid_seed_".format(self.args.non_iid) +str(self.args.seed) + "_client_"+str(client_num)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))
            if self.onemodel == False:
                for i in range(self.epoch_per_client):
                    self.local_train(client_num)
        self.aggregate()

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                if self.onemodel == False:
                    acc_c2 = self.get_acc(self.client_nets2[c],valloader)
                    utils.print_and_log("Client {}, test accuracy: {} , {}".format(c, acc_c, acc_c2), self.logger)
                else:
                    utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)

        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl_cs", target_mode="teacher_oneshot_fl_cs"))

        return

#Implements Oneshot FL (adapted to neural nets) from (Guha et al., 2019)
# Note: original technique was described for SVMs
# We adapt to neural nets by averaging the client logits for the ensemble (in the case of classification)
# The other alternative is to average post-softmax layer
# Variance estimate is derived from variance of ensemble predictions + constant estimate of aleatoric variance
class ONESHOT_FL:
    def __init__(self, num_clients, base_net,
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']
        self.args = args

        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        # initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']
        self.model_save_dir = hyperparams['model_save_dir']

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))

        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )

    def local_train(self, client_num):
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                       optimizer=self.optimizers[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

    # prediction on input x
    def predict_classify(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred_logit = self.client_nets[c](x)

            client_pred.append(pred_logit)

        #make sure to convert from logits to prob predictions
        preds = torch.nn.functional.softmax(torch.mean(torch.stack(client_pred), axis = 0), dim=-1)
        
        return preds 

    def predict_regr(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred = self.client_nets[c](x)
            client_pred.append(pred)

        mean_pred = torch.mean(torch.stack(client_pred), dim=0, keepdims=False)
        var_pred  = 2.0 + torch.var(torch.stack(client_pred), dim=0, keepdims=False)

        return mean_pred, var_pred 


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)
    
    def aggregate(self):
        self.distill.set_student(self.client_nets[0].state_dict())

        #train the student via kd
        self.distill.train(num_epochs = self.kdepoch)
        self.student = self.distill.student
        
        return
    def global_update_step_trained_clients(self, distill = True):
        for client_num in range(self.num_clients):
            PATH = "./results/models/" + self.args.dataset + "_fed_be_5_clients_1_rounds_log_{}_noniid_seed_".format(self.args.non_iid) +str(self.args.seed) + "_client_"+str(client_num)
            print(PATH)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))
        
        
        if distill:
            self.aggregate()

    def global_update_step(self, distill = True):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)

        if distill:
            self.aggregate()

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc

    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            # predict expected to return a mean and variance for an ensemble
            pred, _ = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            #self.global_update_step()
            self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"))

        return
    
    def train_no_distill(self, valloader):
        for i in range(self.num_rounds):
           
            self.global_update_step(distill = False)
            
            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        teacher_nllhd, teacher_cal = utils.test_calibration(model = self, testloader= valloader,
                                                            task = self.task, device = self.device,
                                                            model_type = "ensemble")

        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}, Teacher test NLLHD: {}, Cal Error: {}\n".format(self.num_rounds, teacher_acc, teacher_nllhd, teacher_cal), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"), type="acc")
        utils.write_result_dict_to_file(result = teacher_nllhd, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"), type="nllhd")
        utils.write_result_dict_to_file(result = teacher_cal, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"), type="cal")
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

    def train_saved_models_no_distill(self, valloader):
        for i in range(self.num_rounds):
            if self.task == "regression":
                self.global_update_step(distill = False)
            else:
                # just loads the models
                self.global_update_step_trained_clients(distill=False)
            
            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        teacher_nllhd, teacher_cal = utils.test_calibration(model = self, testloader= valloader,
                                                            task = self.task, device = self.device,
                                                            model_type = "ensemble")

        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}, Teacher test NLLHD: {}, Cal Error: {}\n".format(self.num_rounds, teacher_acc, teacher_nllhd, teacher_cal), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"), type="acc")
        utils.write_result_dict_to_file(result = teacher_nllhd, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"), type="nllhd")
        utils.write_result_dict_to_file(result = teacher_cal, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"), type="cal")
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

# implements Federated Bayesian Ensemble (FedBE) from (Chen and Chao, 2021)
class FedBE:
    def __init__(self, num_clients, base_net,
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']
        self.args = args

        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        # initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        self.model_save_dir = hyperparams['model_save_dir']

        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))

        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )
        
        #posterior samples
        self.post_samples = []
        self.num_samples = 10 #set to number of clients (so we sample an equivalent number of clients)?

    def local_train(self, client_num):
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                       optimizer=self.optimizers[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

    #####################################
    #  FOR ensemble inference
    ####################################

    #returns list of client nets as vectors
    def get_client_nets_as_vec(self):
        c_vectors = []
        for c in range(self.num_clients):
            c_vec = torch.nn.utils.parameters_to_vector(self.client_nets[c].parameters())
            c_vectors.append(c_vec)
        c_vectors = torch.stack(c_vectors, dim=0)

        return c_vectors

    def get_client_mean_cov(self):
        c_vectors = self.get_client_nets_as_vec()
        mean = torch.mean(c_vectors, dim=0)
        
        #too memory intensive - approximate with diagonal matrix
        #cov = torch.Tensor(np.cov((c_vectors).detach().numpy().T))
        
        #cov matrix has this vector as diagonal
        cov = torch.mean((c_vectors - mean)**2, dim = 0)
        cov = torch.clamp(cov, min=1e-6) # to make sure we dont get 0 variance

        return mean, cov

    def vec_to_net(self,vec):
        net = copy.deepcopy(self.base_net)
        torch.nn.utils.vector_to_parameters(vec, net.parameters())
        return net

    #calculates mean and variance of posterior, based
    #on client models
    def get_post_samples(self):
        #posterior samples = {client nets, mean net, sampled gaussian nets}
        
        #reset list of posterior samples
        self.post_samples = []

        #1 - client nets
        for c in range(self.num_clients):
            self.post_samples.append(copy.deepcopy(self.client_nets[c]))

        #2 - mean net
        mean, cov = self.get_client_mean_cov()

        self.post_samples.append(self.vec_to_net(mean))

        #3 - sampled gaussian nets
        
        dist = torch.distributions.Normal(mean, cov.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)

        global_samples = dist.sample([self.num_samples])
 
        
        for s in range(self.num_samples):
            sample = global_samples[s,0,:]
            self.post_samples.append(self.vec_to_net(sample))



    # prediction on input x
    def predict_classify(self, x):
        preds = []
        for m in range(len(self.post_samples)):
            self.post_samples[m] = self.post_samples[m].eval()
            pred_logit = self.post_samples[m](x)

            #append p(y|x, m_i) - for that model m_i
            preds.append(torch.nn.functional.softmax(pred_logit, dim=1))
        return torch.mean(torch.stack(preds), axis = 0)

    def predict_regr(self, x):
        preds = []
        for m in range(len(self.post_samples)):
            self.post_samples[m] = self.post_samples[m].eval()
            pred = self.post_samples[m](x)
            preds.append(pred)

        pred_mean = torch.mean(torch.stack(preds), dim=0, keepdims=False)
        pred_var = 2.0 + torch.var(torch.stack(preds), dim=0, keepdims=False) # aleatoric variance estimate of 2.0
        
        #mean of p(y|x) is avg of means
        return pred_mean, pred_var 


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def aggregate(self, distill):
        #get new set of posterior samples
        self.get_post_samples()

        if distill:
            self.distill.set_student(self.client_nets[0].state_dict())

            #train the student via kd
            self.distill.train(num_epochs = self.kdepoch)
            self.student = self.distill.student
        return
    ######################################################################################

    def global_update_step_trained_clients(self, distill = True):
        for client_num in range(self.num_clients):
            PATH = "./results/models/" + self.args.dataset + "_fed_be_5_clients_1_rounds_log_{}_noniid_seed_".format(self.args.non_iid) +str(self.args.seed) + "_client_"+str(client_num)
            print(PATH)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))

        self.aggregate(distill = distill)

    def global_update_step(self, distill = True):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)

        self.aggregate(distill = distill)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc

    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred, _ = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            #self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            #save client models (the global model is not sent to clients in this version, it is for 1 round only)
            self.save_models()

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"))
       
        return
    
    def train_no_distill(self, valloader):
        for i in range(self.num_rounds):
        
            self.global_update_step(distill = False)
            
            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        teacher_nllhd, teacher_cal = utils.test_calibration(model = self, testloader= valloader,
                                                            task = self.task, device = self.device,
                                                            model_type = "ensemble")

        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}, Teacher test NLLHD: {}, Cal Error: {}\n".format(self.num_rounds, teacher_acc, teacher_nllhd, teacher_cal), self.logger)
        
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"), type="acc")
        utils.write_result_dict_to_file(result = teacher_nllhd, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"), type = "nllhd")
        utils.write_result_dict_to_file(result = teacher_cal, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"), type = "cal")
        return

    def train_saved_models_no_distill(self, valloader):
        for i in range(self.num_rounds):
            if self.task == "regression":
                self.global_update_step(distill = False)
            else:
                # just loads the models and get posterior samples
                self.global_update_step_trained_clients(distill=False)
            
            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        teacher_nllhd, teacher_cal = utils.test_calibration(model = self, testloader= valloader,
                                                            task = self.task, device = self.device,
                                                            model_type = "ensemble")

        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}, Teacher test NLLHD: {}, Cal Error: {}\n".format(self.num_rounds, teacher_acc, teacher_nllhd, teacher_cal), self.logger)
        
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"), type="acc")
        utils.write_result_dict_to_file(result = teacher_nllhd, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"), type = "nllhd")
        utils.write_result_dict_to_file(result = teacher_cal, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"), type = "cal")
        return



# Implements D BetaPredBayes (ours)
# same as F_MCMC but an additional distillation step is added
# Also runs evaluation for BCM (product), mixture, EP MCMC, and FedPA (1 round) using the same samples
class Calibrated_PredBayes_distill(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, distill_data,
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

        #make distillation dataset out of distill_data
        distill_loader = torch.utils.data.DataLoader(distill_data, 
                                                batch_size=self.batch_size, shuffle=True, 
                                                pin_memory=True)

        if task == "classify":
            self.student = copy.copy(base_net)
        else:
            #student model for regression needs to output variance as well
            self.student = models.LinearNetVar(inp_dim = base_net.input_dim, 
                                               num_hidden = base_net.num_hidden, 
                                               out_dim = base_net.out_dim)
            self.student = self.student.to(self.device)

        self.kd_optim_type = hyperparams['kd_optim_type']
        self.kd_lr = hyperparams['kd_lr']
        self.kd_epochs = hyperparams['kd_epochs']

        self.distill = kd.KD(teacher = self, 
                             student = self.student, lr = self.kd_lr,#5e-3
                             device = self.device,
                             train_loader = distill_loader,
                             kd_optim_type = self.kd_optim_type
                            )

        #in this experiment run, we also want to compare to EP MCMC and FedPA with same MCMC samples
        # so we setup this here
        self.fed_pa_trainer = FedPA(num_clients = num_clients, base_net = base_net, 
                                    traindata = traindata, 
                                    num_rounds = 1, 
                                    hyperparams = hyperparams, logger=None, non_iid = non_iid, task = self.task,
                                    device= self.device)

        #for training the interpolation param on distillation set
        self.interp_param = torch.tensor([hyperparams['init_interp_param']], device=self.device)
        self.interp_param.requires_grad = True 
        
        
        self.interp_param_optim = torch.optim.Adam([self.interp_param], lr = hyperparams['interp_param_lr'])
    
    # Step 5 and 6 (implicitly 3 and 4 as well)
    # distill step - on unlabeled dataset, train student model to match 
    # the teacher predictions
    def aggregate(self):
        
        # Step 5 - tune beta
        #train interpolation parameter
        self.train_interp(num_epochs = 10)

        #initialize student to a client network
        if self.task == "classify":
            self.distill.set_student(self.client_train[0].sampled_nets[-1])
        #else: the student is randomly initialized 
        
        # Step 6 distill model
        #train the student via kd
        self.distill.train(num_epochs = self.kd_epochs) #kd_epochs = 50
        self.student = self.distill.student

        return
    
    # code for step 5
    # Tune Beta step
    def train_interp(self, num_epochs):
        
        for i in range(num_epochs):
            epoch_loss = 0.0
            count = 0
            for x, y in self.distill.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.interp_param_optim.zero_grad()

                #need to do log softmax to ensure the exp of this is normalized
                if self.task == "classify":
                    pred_logits = self.predict_classify(x) #should be predicted log prob over classes
                   
                else:
                    pred_mean, pred_var = self.predict_regr(x) # should get mean and var of regression prediction 


                #the loss is the negative log likelihood
                if self.task == "classify":
                    loss = torch.nn.CrossEntropyLoss()(torch.log(pred_logits), y)
                else:
                    loss = utils.GaussianNLLLoss(pred_mean, y, pred_var) #should be input, target, var
                  
                
                loss.backward()

                
                self.interp_param_optim.step()

                epoch_loss += loss.item()
                
                count+=1
            print("Epoch: ", i+1, "Loss: ", epoch_loss, "Interp Param: ", self.interp_param)

            if (i+1)%20 == 0:
                self.test_acc(self.distill_train_loader)

    
        print("Training Interp Param Done! Interp Param: {}".format(self.interp_param))
        return 
    

    # alternative training for beta 
    def train_interp_lfbgs(self, num_epochs):
        # do a pass thru dataset and collect product and mixture probabilities
        prod_list = []
        mix_list = []
        labels_list = []
            
        for x, y in self.distill.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
                
            prod_probs, mix_probs = self.get_prod_mix_predict_classify(x)

            prod_list.append(prod_probs)
            mix_list.append(mix_probs)
            labels_list.append(y)
        prods = torch.cat(prod_list).to(self.device)
        mixs = torch.cat(mix_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)
        

                
        def closure_eval():
            self.interp_param_optim.zero_grad()

            #need to do log softmax to ensure the exp of this is normalized
            if self.task == "classify":
                log_global_pred = self.interp_param*torch.log(prods) + (1- self.interp_param)*torch.log(mixs)
        
            #the loss is the negative log likelihood
            if self.task == "classify":
                loss = torch.nn.CrossEntropyLoss()(log_global_pred, labels)
          
            loss.backward()

            return loss 
                
        self.interp_param_optim.step(closure_eval)
        print("Training Interp Param Done! Interp Param: {}".format(self.interp_param))
        return 
    

    #so we can compare with EP MCMC 
    def ep_mcmc_aggregate(self):
        #in aggregation step - average all models

        global_prec = 0.0
        global_mean = 0.0

        #average over clients    
        for c in range(self.num_clients):
            mean, cov = self.get_client_sample_mean_cov(c)
            
            # in line below, either do this (for identity approx to covar), or 1/cov (for diagonal approx), former gave better results
            client_prec = 1 ############# 1/cov 
            global_prec += client_prec
            global_mean += client_prec * mean 

        global_mean = (1/global_prec) * global_mean 
        global_var = (1/global_prec)*torch.ones_like(global_mean).to(self.device)

        dist = torch.distributions.Normal(global_mean, global_var.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)
        
        global_samples = dist.sample([self.num_g_samples])
      
    
        self.global_train.sampled_nets = []
        
        for s in range(self.num_g_samples):
            # in line below, can either use only mean as sample, or sample from gaussian centered at mean
            # from experiments, just the mean works better (possibly due to high correlation between parameters that a diagonal approx can't capture)
            
            if self.task == "classify":
                sample = global_mean #global_samples[s,0,:] ####################################### 
            else:
                sample = global_samples[s,0,:] #for regression, it works better to have multiple samples (for classification, mean works better)

            #load into global net
            torch.nn.utils.vector_to_parameters(sample, self.global_train.net.parameters())
            self.global_train.sampled_nets.append(copy.deepcopy(self.global_train.net.state_dict()))

        return

    # substep of step 4 (aggregation)
    # output both the product and mixture predictions for x (we will mix between the two)
    def get_prod_mix_predict_classify(self, x):
        global_pred_product = 1.0
        global_pred_mixture = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred_product *= pred
            global_pred_mixture += pred/(self.num_clients)
        
        
        global_pred_product = global_pred_product/torch.sum(global_pred_product, dim=-1, keepdims=True)
        global_pred_mixture = global_pred_mixture/torch.sum(global_pred_mixture, dim=-1, keepdims=True)

        global_pred_product_no_grad = global_pred_product.detach().clamp(min = 1e-41)
        global_pred_mixture_no_grad = global_pred_mixture.detach().clamp(min= 1e-41)

        #renormalize
        global_pred_product_no_grad = global_pred_product_no_grad/global_pred_product_no_grad.sum(dim=-1, keepdims=True)
        global_pred_mixture_no_grad = global_pred_mixture_no_grad/global_pred_mixture_no_grad.sum(dim=-1, keepdims=True)

        return global_pred_product_no_grad, global_pred_mixture_no_grad


    #step 3,4 for aggregation (on classification)
    #prediction on input x
    def predict_classify(self, x):
        global_pred_product = 1.0
        global_pred_mixture = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
            
            # step 3, average over samples
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred_product *= pred
            global_pred_mixture += pred/(self.num_clients)
        
        
        global_pred_product = global_pred_product/torch.sum(global_pred_product, dim=-1, keepdims=True)
        global_pred_mixture = global_pred_mixture/torch.sum(global_pred_mixture, dim=-1, keepdims=True)

        global_pred_product_no_grad = global_pred_product.detach().clamp(min = 1e-41)
        global_pred_mixture_no_grad = global_pred_mixture.detach().clamp(min= 1e-41)

        #renormalize
        global_pred_product_no_grad = global_pred_product_no_grad/global_pred_product_no_grad.sum(dim=-1, keepdims=True)
        global_pred_mixture_no_grad = global_pred_mixture_no_grad/global_pred_mixture_no_grad.sum(dim=-1, keepdims=True)
    

        #interpolate between the two distributions with self.interp_param
        log_global_pred = torch.clamp(self.interp_param, min=0.0)*torch.log(global_pred_product_no_grad) + torch.clamp((1- self.interp_param), min=0.0)*torch.log(global_pred_mixture_no_grad)
        
       
        global_pred = torch.functional.F.softmax(log_global_pred, dim = -1)
        global_pred = global_pred/torch.sum(global_pred, dim=-1, keepdims=True) 
        
        return global_pred 
    
    # step 3,4 for aggregation (for regression)
    def predict_regr(self,x):
        global_pred_product = 0.0
        prec_sum_product = 0.0

        global_pred_mixture = 0.0 
        second_moment_mixture = 0.0 

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
            
            # step 3, average over samples
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)

            # the regression model is assumed to make predictions as N(f(x), 1) where f(x) = neural net output
            #Therefore, the variance in the prediction is 1.0 (the homeoscedatic assumed variance) + bayesian variance
            if pred_list.shape[0] == 1:
                pred_var = torch.ones_like(pred_mean)
            else: 
                if self.task == "classify":
                    pred_var = 1.0 + torch.var(pred_list, dim = 0, keepdims = False)
                else:
                    # set a higher predicted variance for regression (is a hyperparam basically) 
                    pred_var = 2.0 + torch.var(pred_list, dim = 0, keepdims = False)

            #product aggregation
            #assuming a uniform posterior
            global_pred_product += pred_mean/pred_var
            prec_sum_product += 1/pred_var
            
            # mixture aggregation
            global_pred_mixture += pred_mean/self.num_clients
            second_moment_mixture +=  (pred_var + pred_mean**2)/self.num_clients

        global_pred_product = global_pred_product/prec_sum_product
        var_product = 1/prec_sum_product 

        var_mixture = second_moment_mixture - (global_pred_mixture)**2


        # detach gradients (wrt interp_param these are all constants)
        global_pred_product = global_pred_product.detach()
        var_product = var_product.detach()
        global_pred_mixture = global_pred_mixture.detach()
        var_mixture = var_mixture.detach()


        #interpolate the mean and variance between the product and mixture 
        global_prec =  (self.interp_param)/var_product + (1 - self.interp_param)/var_mixture
        global_var = 1/global_prec 

        global_pred = (self.interp_param)*global_pred_product/var_product + (1-self.interp_param)*global_pred_mixture/var_mixture 
        global_pred = global_pred * global_var
        return global_pred, global_var #should output mean and variance !!!!!

    #Step 3 and 4 (local posterior estimate  + aggregation for global posterior)
    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)
            _, pred_class = torch.max(pred, 1)    

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
 
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred, pred_var = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    #load the local models trained previously, then tune beta and distill (call aggregate)
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            model_PATH =  "./results_distill_f_mcmc/models/" + self.dataset + "_mcmc_5_clients_1_rounds_log_{}_noniid_seed_".format(self.non_iid) +str(self.seed) + "_client_"+str(client_num) 

            # there should be 6 samples for each dataset
            for idx in range(self.max_samples): 
                path = model_PATH + "_sample_" + str(idx) 
                weight_dict = torch.load(path)
                self.client_train[client_num].sampled_nets.append(weight_dict)
            
        self.aggregate()

    def train(self, valloader):
        for i in range(self.num_rounds):
            # Step 1 and 2: local MCMC sampling + communicating to server
            
            ################################
            #train from scratch

            # will call aggregate() function, which implements step 5 and 6 (implicitly 3 and 4 as well)
            # saves distilled model after this step
            self.global_update_step()

            #OR load saved models
            #if self.task == "classify":
            #    self.global_update_step_trained_clients()
            #else:
            #    self.global_update_step() #regression is cheap enough to retrain
            ######################################

            utils.print_and_log("\nTuned value of interp param: {}\n".format(self.interp_param), logger=self.logger)

            acc = self.distill.test_acc(valloader)
            nllhd, cal_error = utils.test_calibration(model = self.distill.student, testloader=valloader, task=self.task, 
                                                      device = self.device, model_type= "single")

            utils.print_and_log("\nGlobal rounds completed: {}, distilled_f_mcmc test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(i, acc, nllhd, cal_error), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        ############################
        # PROCESS AND RECORD DATA
        ############################
        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger, type="acc")
        utils.write_result_dict(result=nllhd, seed=self.seed, logger_file=self.logger, type="nllhd")
        utils.write_result_dict(result=cal_error, seed=self.seed, logger_file=self.logger, type="cal")

        #evaluate samples on other methods (FMCMC, and EP MCMC)
        f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("\n\nGlobal rounds completed: {}, f_mcmc test_acc: {}".format(i, f_mcmc_acc), self.logger)
        f_mcmc_nllhd, f_mcmc_cal_error = utils.test_calibration(model = self, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("Global rounds completed: {}, f_mcmc test_ NLLHD: {}, Cal Error: {}\n".format(i, f_mcmc_nllhd, f_mcmc_cal_error), self.logger)

        #save to dict
        utils.write_result_dict_to_file(result=f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=f_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=f_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"),
                                 type = "cal")


        #evaluate f_mcmc_model without interpolation param = 1 (product) and 0 (mixture)
        self.tuned_interp_param =  torch.clone(self.interp_param)

        # PRODUCT model/ previous FMCMC
        self.interp_param = torch.tensor([1.0], device=self.device) 
        prod_f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("\nGlobal rounds completed: {}, PRODUCT f_mcmc test_acc: {}".format(i, prod_f_mcmc_acc), self.logger)
        prod_f_mcmc_nllhd, prod_f_mcmc_cal_error = utils.test_calibration(model = self, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("Global rounds completed: {}, PRODUCT f_mcmc test_ NLLHD: {}, Cal Error: {}\n".format(i, prod_f_mcmc_nllhd, prod_f_mcmc_cal_error), self.logger)

        #save to dict
        utils.write_result_dict_to_file(result=prod_f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="product_f_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=prod_f_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="product_f_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=prod_f_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="product_f_mcmc"),
                                 type = "cal")


        # MIXTURE model/ previous FMCMC
        self.interp_param = torch.tensor([0.0], device=self.device)
        mix_f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("\nGlobal rounds completed: {}, MIXTURE f_mcmc test_acc: {}".format(i, mix_f_mcmc_acc), self.logger)
        mix_f_mcmc_nllhd, mix_f_mcmc_cal_error = utils.test_calibration(model = self, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("Global rounds completed: {}, MIXTURE f_mcmc test_ NLLHD: {}, Cal Error: {}\n".format(i, mix_f_mcmc_nllhd, mix_f_mcmc_cal_error), self.logger)

        #save to dict
        utils.write_result_dict_to_file(result=mix_f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="mixture_f_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=mix_f_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="mixture_f_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=mix_f_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="mixture_f_mcmc"),
                                 type = "cal")

        #revert self interp param
        self.interp_param = self.tuned_interp_param

        #compute ep mcmc result and store in global_train
        self.ep_mcmc_aggregate()
        ep_mcmc_acc = self.global_train.test_acc(valloader)
        ep_mcmc_nllhd, ep_mcmc_cal_error = utils.test_calibration(model = self.global_train, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("\nGlobal rounds completed: {}, ep_mcmc test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(i, ep_mcmc_acc, ep_mcmc_nllhd, ep_mcmc_cal_error), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=ep_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=ep_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=ep_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"),
                                 type = "cal")


        #for fed_pa 1 round results
        #copy over trained clients
        self.fed_pa_trainer.client_train = copy.deepcopy(self.client_train)

        # comment out for cases where FedPA diverges (eg. in classification)
        #take step of FedPA 
        if self.task == "classify":
            self.fed_pa_trainer.global_update_step_trained_clients()
            fed_pa_acc = self.fed_pa_trainer.get_acc(self.fed_pa_trainer.global_train.net, valloader)
            fed_pa_nllhd, fed_pa_cal_error = utils.test_calibration(model = self.fed_pa_trainer.global_train.net, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "single")
            utils.print_and_log("Global rounds completed: {}, fed_pa test_acc: {},  NLLHD: {}, Calibration Error: {}".format(i, fed_pa_acc, fed_pa_nllhd, fed_pa_cal_error), self.logger)
        
            #save to dict 
            utils.write_result_dict_to_file(result=fed_pa_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"),
                                 type = "acc")
            utils.write_result_dict_to_file(result=fed_pa_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"),
                                 type = "nllhd")
            utils.write_result_dict_to_file(result=fed_pa_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"),
                                 type = "cal")

        return
