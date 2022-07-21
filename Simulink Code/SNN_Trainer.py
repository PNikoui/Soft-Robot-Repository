import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
class Training:
    def __init__(self, XY_Train_data, XY_Target_data):  #, Y_Train_data, Y_Target_data):
            super(Training, self).__init__()
            #### Define training data 

            # TRAINGING DATA FOR X MOTOR CHANGE
            Train_XY_variation = []
            Target_XY_variation = []
            with open(XY_Train_data, 'r') as fd:
                reader_xy = csv.reader(fd)
                for row in reader_xy:
                    Train_XY_variation.append([int(float(i)) for i in row])

            with open(XY_Target_data, 'r') as ft:
                reader_xy1 = csv.reader(ft)
                for row in reader_xy1:
                    Target_XY_variation.append([int(float(i)) for i in row])

            self.XY = Train_XY_variation
            self.XYT = Target_XY_variation

    def Train(self,model, loss_func, optimizer, n_Train_epochs,num_steps):
        super().__init__()
            
        self.num_epochs = n_Train_epochs
        self.num_steps = num_steps
        # x = torch.FloatTensor(np.array(Train_X_variation)[:,1])
        # print(list(zip([xx[0] for xx in Train_X_variation],[yy[1] for yy in Train_Y_variation])))
        # print([xx[1] for xx in Train_X_variation])
        x = torch.FloatTensor(list(zip([xx[0] for xx in self.XY],[yy[1] for yy in self.XY],[zz[2] for zz in self.XY])))
#         print(type(x))
        # # x = torch.index_select(x, 1, torch.LongTensor([1,0]))
#         print(x.shape) 
        t = torch.zeros(self.num_steps,x.shape[0],x.shape[1])
        tt = torch.FloatTensor(list(zip([xxt[0] for xxt in self.XYT],[yyt[1] for yyt in self.XYT],[zzt[2] for zzt in self.XYT])))
        for i in range(self.num_steps):
            t[i] = tt
        # print(t)
        # # t = torch.index_select(t, 1, torch.LongTensor([1,0]))
#         print(t.shape)
        # t = torch.FloatTensor((np.array(Target_Y_variation)[:,1]))
        ERROR = []

        for _ in range(self.num_epochs):
            # print(x.shape)
#             prediction = torch.stack(model(x))
            prediction = model(x)
#             print(prediction)
#             print(type(prediction))
            loss = loss_func(prediction[0], t)
            # loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            ERROR.append(loss.detach().numpy())

        print("Loss: ", loss.detach().numpy())
        torch.save(model.state_dict(), 'SNN_Learned_Weights.pth')
        return ERROR


