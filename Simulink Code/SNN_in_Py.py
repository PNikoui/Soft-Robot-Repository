import numpy as np
# import snntorch.spikeplot as splt
from decimal import Decimal
import torch, torch.nn as nn
import SNN
# import LearningWindow

class Network(nn.Module):
# class Network():

#     def __init__(self, num_inputs=2, num_hidden=64, num_outputs=1):
    def __init__(self,num_inputs, num_hidden, num_outputs):
#     def __init__(self):
        super(Network,self).__init__()

#         self.model = SNN.Net(num_inputs=2, num_hidden=64, num_outputs=1)
#         self.num_inputs = num_inputs
        self.num_hidden = num_hidden
#         self.num_outputs = num_outputs
#         self.model = SNN.Net(num_inputs, num_hidden, num_outputs)
        self.model = SNN.Net()
    #         self.model = SNN.Net(self.num_inputs, self.num_hidden, self.num_outputs)
#         self.W = torch.empty((SNN.num_steps,1,self.num_hidden))
        self.A_plus = 0.040  ## 40 mV
        self.A_minus = -0.04
        self.tau_plus = 0.01  ## 10 ms for NeoCortex basket cells
        self.tau_minus = 0.01

    def Load_model(self):

        ## Load previously learned weights for Curriculum learning
        
        self.model = self.model.load_state_dict(torch.load('turn_1.pth',map_location=torch.device('cpu')))
#         self.model = self.model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
    
#         return self.model

    def Mat2Py(self,Mat_array):

        Py_array = np.array(Mat_array)
#         [s1,s2,s3] = self.model(PyTorchTarget)

        return Py_array
    
    def Run_model(self,Target):
#         print('Hi')
    #     model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3)
    
#         model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
#         print(Target)
        PyTorchTarget = torch.tensor([Target])
#         print(PyTorchTarget)
#         print(self.model)
        OUT, S1,S2 = self.model(PyTorchTarget)
        s1 = OUT[0].detach().numpy()
#         s2 = OUT[:,1].detach().numpy()
#         s3 = OUT[:,2].detach().numpy()
#         print(list([s1,s2,s3]))
        
#         print(Spikes)
#         print(np.shape(Spikes))
        self.PreSynapticSpikeTimes = S1
#         self.PreSynapticSpikeTimes = SNN.Spike1 #Spikes[:,0,:]
#         print(self.PreSynapticSpikeTimes)
        self.PostSynapticSpikeTimes =  S2
#        
        return s1 #np.array([s1,s2,s3])  
    
    def Update(self,Error,Threshold):


        Dw_N = 0
        Dw_F = 0

        if abs(Error.any()) > Threshold:  ## For now, define a threshold for when to activate STDP. 5 cm 

            # Initialize pre- and post synaptic updates
#             Dw_N = 0
            Dw_N = torch.empty((1,int(self.num_hidden)))
#             Dw_F = 0
            Dw_F = torch.empty((1,int(self.num_hidden)))
             
            for tj_f in self.PreSynapticSpikeTimes:  ## Evaluate difference in spikes SNN.num_steps times
    
                Dw_F += Dw_N
#                 self.model = self.model.fc1.weight[tj_f] + Dw_N  ## Update PreSynaptic weights
                self.model.fc1.weight + Dw_N
                Dw_N = torch.empty((1,int(self.num_hidden)))  #0 # reset summation
    
                for ti_n in self.PostSynapticSpikeTimes:
                    Dw_ij = self.LearningWindow(ti_n-tj_f)  ## tensor of updates size: 1xnum_hidden
                    print(Dw_ij.shape)
#                     print(np.shape(Dw_N))
                    Dw_N[0] += Dw_ij
    
                    
    ### Plot Spike Times Raster Plot in terminal
    
    #         fig = plt.figure(facecolor="w", figsize=(15, 15))
    # #         ax = fig.add_subplot(411)
    #         
    #         #  s: size of scatter points; c: color of scatter points
    #         splt.raster(self.PreSynapticSpikeTimes[:, 0, :], ax, s=1.5, c="black")
    #         plt.title("Leaky Hidden Layer 1")
    #         plt.xlabel("Time step")
    #         plt.ylabel("Neuron Number")
    
    
            #         print("The model is learning")
            #         Error.backward()
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
                    
                #     model.
                #     torch.save(model.state_dict(), 'SNN_Updated_Weights.pth')
                
        return print("Total Weight Update: {}".format(Dw_F)) #model

    def Py2Mat(self,Py_array):

        Mat_array = Decimal(Py_array)

        return Mat_array

    def LearningWindow(self,x):

        W = torch.empty((1,int(self.num_hidden)))
        
#         if x.any() > 0: # loop through all neuron connections
                    
#         x_idx_plus = [n for n,i in enumerate(x[0]) if i>0]
#         x_idx_minus = [n for n,i in enumerate(x[0]) if i<0]
#         print(x_idx_plus)
#         print(x_idx_minus)
        for n,i in enumerate(x[0]):
            if i>0:
#                 print(i)
                x_idx_plus = n
#                 print(n)
#                 print(self.A_plus*torch.exp(-x[0,x_idx_plus]/self.tau_plus))
#                 print((self.A_plus*torch.exp(-x[0,x_idx_plus]/self.tau_plus)).shape)
#                 print(W.shape)
                W[0,x_idx_plus] = self.A_plus*torch.exp(-x[0,x_idx_plus]/self.tau_plus)
#                 print(x_idx_plus)
#                 print(W[0,x_idx_plus].shape)
    
            if i<0:
                x_idx_minus = n
#                 print(( self.A_minus*torch.exp(x[0,x_idx_minus]/self.tau_minus)).shape)
                W[0,x_idx_minus] = self.A_minus*torch.exp(x[0,x_idx_minus]/self.tau_minus)
#                 print(x_idx_minus)

#         if x_idx_plus:
#             print(W)
#             print(W.shape)
#             W[1,x_idx_plus] = self.A_plus*torch.exp(-x[x_idx_plus]/self.tau_plus)
#             print(W[x_idx_plus])
#         
#         if x_idx_minus:
#             W[:,x_idx_minus] = self.A_minus*torch.exp(x[x_idx_minus]/self.tau_minus)

#         if x>0:
#             W = self.A_plus*exp(-x/self.tau_plus)

#         else:
#             W = self.A_minus*exp(x/self.tau_minus)

        return W 
    