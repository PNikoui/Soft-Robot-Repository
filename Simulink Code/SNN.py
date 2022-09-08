from snntorch import surrogate
import torch, torch.nn as nn
import snntorch as snn
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

beta = 0.9  # neuron decay rate 
spike_grad = surrogate.fast_sigmoid()
num_steps = 100
# Spike1 = 0
  
# class Net(torch.nn.Module):
    
#     #### Build the neural network
class Net(nn.Module):
    
    def __init__(self, num_inputs=2, num_hidden=64, num_outputs=1):
#     def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

      # initialize layers
        self.num_inputs = num_inputs
        self.num_steps = num_steps
        self.num_hidden = num_hidden
        self.Spike1 = torch.empty((self.num_steps,1,self.num_hidden))
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs
#         self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
#         self.lif1 = snn.Leaky(beta=beta, output=True)
# #         self.fc2 = nn.Linear(num_hidden, num_outputs)
#         self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
#         self.lif2 = snn.Leaky(beta=beta, output=True)
#         self.fc3 = nn.Linear(self.num_hidden, self.num_outputs)

## For rate coding:
#         self.fc1 = nn.Linear(num_steps, num_hidden)


## Without Spike Encoding
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, output=True)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, output=True)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        
        
#         self.fc2 = nn.Linear(num_hidden, num_outputs)
#         self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        
#         DIST = torch.randint(0,2, size=(self.num_inputs,self.num_hidden))
# #         x = torch.round(DIST*x)*x
#         x = DIST@x.type(torch.LongTensor)
        
#         print(type(x))
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk1_rec = torch.empty((self.num_steps,1,self.num_hidden))
        spk2_rec = torch.empty((self.num_steps,1,self.num_hidden))
#         mem2 = self.lif2.init_leaky()

#         spk2_rec = torch.empty((num_steps,20,3)) #[]  # Record the output trace of spikes
#         mem2_rec = torch.empty((num_steps,20,3)) #[]  # Record the output trace of membrane potential

        for step in range(num_steps):
            
#             print(np.shape(x))
#             print(self.fc1)
            cur1 = self.fc1(x.flatten(1))
#             print(np.shape(x))
            # cur1 = self.fc1(x.flatten(1))
            cur1 = self.fc1(x)
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
#         self.Spike1 = spk1_rec
        Spike1 = spk1_rec
#         self.Spike2 = spk2_rec
        Spike2 = spk2_rec
#         Spikes = torch.cat((spk1_rec,spk2_rec),1)
        
#         print(np.shape(cur3))
#         print(np.shape(Spikes[:,0,:]))
#             spk2, mem2 = self.lif2(cur2, mem2)

############ TEST ##############
#             cur1 = self.fc1(x.flatten(1))
#             spk1, mem1 = self.lif1(cur1, mem1)
#             spk2, mem2 = self.lif2(spk1, mem2)
#             cur2 = self.fc2(spk2)
#             print([spk2,mem2])

        return cur3, Spike1, Spike2#, spk1_rec, spk2_rec #cur2
#             spk2_rec[step] = spk2
#             mem2_rec[step] = mem2
#         print([type(spk2_rec),type(mem2_rec)])
#         return cur2
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



