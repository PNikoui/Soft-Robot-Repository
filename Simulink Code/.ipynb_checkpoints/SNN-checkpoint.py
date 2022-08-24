from snntorch import surrogate
import torch, torch.nn as nn
import snntorch as snn
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

beta = 0.9  # neuron decay rate 
spike_grad = surrogate.fast_sigmoid()
num_steps = 100   

# class Net(torch.nn.Module):
    
#     #### Build the neural network

#     # Define neural network
#     def __init__(self, n_feature, n_hidden, n_output):
        
#         self.hidden = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
#         self.hidden2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#         self.hidden3 = nn.Linear(16*4*4, 10),
#         self.predict = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
#         self.Flatten = nn.Flatten()
# #         self.BN = torch.nn.BatchNorm1d(n_hidden,affine=False)
# #         self.p = 0.35
# #         self.DO = torch.nn.Dropout2d(self.p)

        
# Define Network
class Net(nn.Module):
    
    def __init__(self, num_inputs=2, num_hidden=64, num_outputs=1):
#     def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

      # initialize layers
        self.num_steps = num_steps
        self.num_hidden = num_hidden
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs
#         self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
#         self.lif1 = snn.Leaky(beta=beta, output=True)
# #         self.fc2 = nn.Linear(num_hidden, num_outputs)
#         self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
#         self.lif2 = snn.Leaky(beta=beta, output=True)
#         self.fc3 = nn.Linear(self.num_hidden, self.num_outputs)

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, output=True)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, output=True)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        
        
    def forward(self, x): 
#         print(type(x))
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk1_rec = torch.empty((self.num_steps,1,self.num_hidden))
        spk2_rec = torch.empty((self.num_steps,1,self.num_hidden))

#         spk2_rec = torch.empty((num_steps,20,3)) #[]  # Record the output trace of spikes
#         mem2_rec = torch.empty((num_steps,20,3)) #[]  # Record the output trace of membrane potential

        for step in range(num_steps):
            
#             print(np.shape(x))
#             print(self.fc1)
            cur1 = self.fc1(x.flatten(1))
#             print(np.shape(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            
#             spk1_rec.append(spk1)
#             spk2_rec.append(spk2)
            spk1_rec[step] = spk1
            spk2_rec[step] = spk2
            
#         spk1_rec = torch.stack(spk1_rec)
#         spk2_rec = torch.stack(spk2_rec)
#         print(np.shape(spk1_rec))
        self.Spike1 = spk1_rec
#         Spikes = torch.cat((spk1_rec,spk2_rec),1)
        
#         print(np.shape(cur3))
#         print(np.shape(Spikes[:,0,:]))

        return cur3 #, spk1_rec, spk2_rec #cur2
#         return torch.stack(list(spk2_rec), dim=0), torch.stack(list(mem2_rec), dim=0)  #torch.stack(spk2_rec), torch.stack(mem2_rec)

net = Net().to(device)



