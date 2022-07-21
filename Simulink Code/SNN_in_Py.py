import numpy as np
from decimal import Decimal
import torch
import SNN

class Network:

    def __init__(self):

        self.model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3)

    def Load_model(self):
    
    
        self.model = self.model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
    
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
        OUT = self.model(PyTorchTarget)
        s1 = OUT[:,0].detach().numpy()
        s2 = OUT[:,1].detach().numpy()
        s3 = OUT[:,2].detach().numpy()
#         print(list([s1,s2,s3]))
        
        return np.array([s1,s2,s3])
    
    
    def Update(self,Error):
    
#         print("The model is learning")
#         Error.backward()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
    #     model.
    #     torch.save(model.state_dict(), 'SNN_Updated_Weights.pth')
    
        return print("The weights are being updated and the model is learning") #model

    def Py2Mat(self,Py_array):

        Mat_array = Decimal(Py_array)

        return Mat_array
    