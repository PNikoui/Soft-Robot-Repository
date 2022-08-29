import numpy as np
# import snntorch.spikeplot as splt
from decimal import Decimal
import torch, torch.nn as nn
import SNN

class Network(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):

        self.model = SNN.Net(num_inputs=2, num_hidden=32, num_outputs=1)
#         self.model = SNN.Net(num_inputs, num_hidden, num_outputs)
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
        print('Hi')
    #     model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3)
    
#         model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
#         print(Target)
        PyTorchTarget = torch.tensor([Target])
#         print(PyTorchTarget)
#         print(self.model)
        OUT = self.model(PyTorchTarget)
        print(self.num_steps)
        s1 = OUT[0].detach().numpy()
#         s2 = OUT[:,1].detach().numpy()
#         s3 = OUT[:,2].detach().numpy()
#         print(list([s1,s2,s3]))
        
#         print(Spikes)
#         print(np.shape(Spikes))

        self.PreSynapticSpikeTimes = self.Spike1 #Spikes[:,0,:]
# 
#         self.PostSynapticSpikeTimes = Spikes[:,1,:]
#        
        return s1 #np.array([s1,s2,s3])
    
    
    def Update(self,Error,Threshold):

        if Error > Threshold:  ## For now, define a threshold for when to activate STDP. 5 cm 

            # Initialize pre- and post synaptic updates 
            Dw_N = 0
            Dw_F = 0
             
            for tj_f in self.PreSynapticSpikeTimes:
    
                Dw_F += Dw_N
                self.model = self.model.fc1.weight[tj_f] + Dw_N  ## Update PreSynaptic weights
                Dw_N = 0 # reset summation
    
                for ti_n in self.PostSynapticSpikeTimes:
    
                    Dw_ij = LearningWindow(ti_n-tj_f)
                    Dw_N += Dw_ij
    
                    
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
        

        if x>0:
            W = self.A_plus*exp(-x/self.tau_plus)

        else:
            W = self.A_minus*exp(x/self.tau_minus)

        return W
    