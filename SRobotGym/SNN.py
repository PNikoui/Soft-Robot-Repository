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
    
    def __init__(self, num_inputs=3, num_hidden=32, num_outputs=3):
        super().__init__()

      # initialize layers
        self.num_steps = num_steps
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, output=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
#         self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x): 
#         print(type(x))
        mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()

#         spk2_rec = torch.empty((num_steps,20,3)) #[]  # Record the output trace of spikes
#         mem2_rec = torch.empty((num_steps,20,3)) #[]  # Record the output trace of membrane potential

        for step in range(num_steps):
            
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
#             spk2, mem2 = self.lif2(cur2, mem2)

############ TEST ##############
#             cur1 = self.fc1(x.flatten(1))
#             spk1, mem1 = self.lif1(cur1, mem1)
#             spk2, mem2 = self.lif2(spk1, mem2)
#             cur2 = self.fc2(spk2)
#             print([spk2,mem2])

#             spk2_rec[step] = spk2
#             mem2_rec[step] = mem2
#         print([type(spk2_rec),type(mem2_rec)])
        return cur2
#         return torch.stack(list(spk2_rec), dim=0), torch.stack(list(mem2_rec), dim=0)  #torch.stack(spk2_rec), torch.stack(mem2_rec)

net = Net().to(device)

# output, mem_rec = net(data)



#     def forward(self, x):
#         # print(x.shape)
#         # print(x)
#         x = self.hidden(x)
#         # print(np.shape(x))
#         # print(x.shape)
#         x = self.hidden2(x)
#         x = self.Flatten(x)
#         x = self.hidden3(x)
#         x = self.predict(x)
#         return x




