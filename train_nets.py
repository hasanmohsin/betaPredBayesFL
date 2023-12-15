import torch
import numpy as np
import copy

####################################################
# Local training algorithms (optimizer, or sampler)
#####################################################

def sgd_train_step(net, optimizer, criterion, trainloader, device):
    epoch_loss = 0.0
    count = 0
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred_logits = net(x)
        loss = criterion(pred_logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        count+=1
    return net, epoch_loss
 
# training step using proximal term in loss between 'net' parameters and 'g_net' parameters
# 'net' parameters are being updated
def sgd_prox_train_step(net, g_net, reg, optimizer, criterion, trainloader, device):
    epoch_loss = 0.0
    count = 0
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        prox_term = 0.0
        for w, w_t in zip(net.parameters(), g_net.parameters()):
            prox_term += (w - w_t).norm(2)

        pred_logits = net(x)
        loss = criterion(pred_logits, y) + reg*prox_term
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        count+=1
    return net, epoch_loss

# training with regular SGD
def sgd_train(net, lr, num_epochs, trainloader):
   
    optimizer = torch.optim.Adam(net.parameters(), lr =lr)
    net = net.train()

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_epochs):
        net, epoch_loss = sgd_train_step(net, optimizer, criterion, trainloader)
        print("Epoch: ", i+1, "Loss: ", epoch_loss)
    
    print("Training Done!")
    return net


#cyclic Stochastic Gradient Hamiltonian Monte-Carlo sampler (cSGHMC)
# code adapted from the code that was publically released with the paper
# "Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning" (Zhang et al., 2020)
# link to code is within that paper

class cSGHMC:
    def __init__(self,
                  base_net, 
                  trainloader, device, task, hyperparams):
        self.net = base_net
        self.trainloader = trainloader
        self.device = device

        #classify or regression
        self.task = task

        #hard-coded params
        self.num_epochs = hyperparams['epoch_per_client'] #24
        self.weight_decay = hyperparams['weight_decay'] #5e-4
        self.datasize = hyperparams['datasize']#/5 #60000
        self.batch_size = hyperparams['batch_size'] #100
        self.num_batch = int(self.datasize/self.batch_size) + 1
        self.init_lr = hyperparams['init_lr'] #0.1 #0.5
        self.M = hyperparams['M'] #4 #num_cycles
        self.cycle_len = (self.num_epochs/self.M) #6
        self.T = self.num_epochs*self.num_batch

        self.sample_per_cycle = hyperparams['sample_per_cycle']
        self.burn_in_iter = self.cycle_len - self.sample_per_cycle #4

        if self.task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        
        if hyperparams['temp'] == -1:
            self.temperature = 1/self.datasize
        else:
            self.temperature = hyperparams['temp'] #1.0 #1/self.datasize #0.5#0.05#1/self.datasize#/60000#self.datasize
        
        self.alpha = hyperparams['alpha'] #0.9

        self.max_samples = hyperparams['max_samples'] #15 #4#15
        self.sampled_nets = []

    #gradient rule for SG Hamiltonian Monte Carlo
    def update_params(self, lr,epoch):
        for p in self.net.parameters():
            if not hasattr(p,'buf'):
                p.buf = torch.zeros(p.size()).to(self.device)
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1-self.alpha)*p.buf - lr*d_p
            if (epoch%self.cycle_len)+1>self.burn_in_iter:
                eps = torch.randn(p.size()).to(self.device)
                buf_new += (2.0*lr*self.alpha*self.temperature/self.datasize)**.5*eps
            p.data.add_(buf_new)
            p.buf = buf_new

    #learning rate schedule according to cyclic SGMCMC
    def adjust_learning_rate(self, epoch, batch_idx):
        rcounter = epoch*self.batch_size+batch_idx
        cos_inner = np.pi * (rcounter % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5*cos_out*self.init_lr
        return lr

    def train_epoch(self, epoch):
        #print('\nEpoch: %d' % epoch)
        self.net = self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.net.zero_grad()
            lr = self.adjust_learning_rate(epoch,batch_idx)
            outputs = self.net(inputs)

            if self.task == "regression":
                outputs = outputs.reshape(targets.shape)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.update_params(lr,epoch)

            train_loss += loss.data.item()
          
        print("Epoch: {}, Loss: {}".format(epoch+1, train_loss))
      
    
    
    #sample a net by saving its state dict
    def sample_net(self):

        if len(self.sampled_nets) >= self.max_samples:
            self.sampled_nets.pop(0)

        self.sampled_nets.append(copy.deepcopy(self.net.state_dict()))

        print("Sampled net, total num sampled: {}".format(len(self.sampled_nets)))

        return None

    #for each net in ensemble, compute prediction (could be logits depending on net)
    def ensemble_inf(self, x, Nsamples=0, out_probs = True):
        if Nsamples == 0:
            Nsamples = len(self.sampled_nets)

        x = x.to(self.device)

        out = torch.zeros(Nsamples, x.shape[0], self.net.out_dim, device = self.device)
       

        if self.task != "classify":
            out_probs = False

        # iterate over all saved weight configuration samples
        for idx, weight_dict in enumerate(self.sampled_nets):
            if idx == Nsamples:
                break
            self.net.load_state_dict(weight_dict)
            self.net.eval()

            out_x = self.net(x)
            
            #reshape to [B, 1]
            if self.net.out_dim == 1:
                out_x = out_x.clone().unsqueeze(-1)

            #out[idx] = out_x

            if out_probs:
                out_x_val = torch.nn.functional.softmax(out_x, dim = 1)
            else:
                out_x_val = out_x 

            #out[idx] should be initially 0
            out[idx] = out_x_val.clone()

           
        return out
    
    def predict(self, x, out_probs=True):
        
        outs =  self.ensemble_inf(x, out_probs = out_probs)
        if self.task == "classify":
            return torch.mean(outs, dim=0)
        else:
            # when we do predict on EP MCMC, add an aleatoric variance estimate of 1.0
            return torch.mean(outs, dim=0), torch.var(outs, dim=0)

    def test_acc(self, testloader):
        #for classification
        total = 0
        correct = 0

        #FOR REGRESSION tasks
        criterion = torch.nn.MSELoss()
        total_loss  = 0.0

        for batch_idx, (x, y) in enumerate(testloader):
            x  = x.to(self.device)
            y = y.to(self.device)

            pred_list = self.ensemble_inf(x, out_probs = True)

            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            if self.task == "classify":
                _, pred_class = torch.max(pred, 1)    

                total += y.size(0)
                correct += (pred_class == y).sum().item()
            else:
                pred = pred.reshape(y.shape)
                loss = criterion(pred, y)
                total_loss += loss.item()

        if self.task == "classify":
            acc = 100*correct/total
            print("Accuracy on test set: ", acc)
            return acc
        else:
            print("MSE on test set: ", total_loss)
            return total_loss

    def train(self):

        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

            # 3 models sampled every 8 epochs
            if (epoch%self.cycle_len)+1 > self.burn_in_iter:
                self.sample_net()
        
        return 
#####################################################
