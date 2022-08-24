import torch
import SNN

def Run_model(Target,num_hidden,num_steps):

#     num_steps = 10
#     model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3,num_steps = num_steps)
    print('Hi')
    model = SNN.Net(num_inputs=1, num_hidden=num_hidden, num_outputs=1,num_steps = num_steps)

#     num_inputs=3;
#     num_hidden=32;
#     num_outputs=3;
#     num_steps = 100;
#     model = SNN.Net(num_inputs, num_hidden, num_outputs,num_steps)
    model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
    
    Target = torch.tensor([list(Target)])
    [s1,s2,s3] = model(Target)
    
    return model,s1,s2,s3


def Update(model,Error):

    print("The model is learning")
    
#     model.

