#!/usr/bin/env python    
# -*- coding: utf-8 -*

import socket, struct, os
import numpy as np
# import pickle

import torch
import SNN

def Load_model():

    model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3)

    return model

def Run_model(model,Target):
    print('Hi')
#     model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3)

    model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
    
    Target = torch.tensor([Target])
    [s1,s2,s3] = model(Target)
    
    return s1,s2,s3


def Update(model,Error):

    print("The model is learning")
    
#     model.
#     torch.save(model.state_dict(), 'SNN_Updated_Weights.pth')

    return model

def main():
    # -------------------------------- Initializing --------------------------------------------
    # Create a socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind the IP address and port.  
    localaddr = ("127.0.0.1", 54320)
    udp_socket.bind(localaddr)
    # Create an increment for while loop
    count = 0
    # Create a list to restor the data from simulink.
    data_collect = []
    # Create a path to save figure:
    path = 'Your Path Here'

    Adaptive_switch = 0

    model = Load_model()

    print("Please open the Simulink file under the current working directory")
    print("The program is waiting until you run the Simulink file.")

    #----------------------------------- Data Receiving ----------------------------------------
    # Using a loop to receive data from Simulink
    while count < 24: # Can be modified by (simulationTime/sampleTime).
        # Start to receive data from Simulink.
        recv_data = udp_socket.recvfrom(1024)
        print(recv_data)
        # recv_data will return tuple, the first element is DATA, and the second is address information
#         recv_msg = recv_data[0]
        recv_msg = np.frombuffer(recv_data[0])
#         print(recv_msg)
        send_addr = recv_data[1]
        # Decode the data from Simulink whose type is double and return a tuple
#         print(recv_msg)
        Struct_size = struct.calcsize(recv_msg)
        print(Struct_size)
        recv_msg_decode = struct.unpack("d", recv_msg)[0]
        # Restore the data to a list:
        data_collect.append(recv_msg_decode)
#         data_collect.append(recv_msg)
        # Set the condition to jump out of this loop ???
        # Print the address information and the received data
#         print repr(recv_msg_decode)
#         print("Number from MATLAB %s is : %s" % (str(send_addr), recv_msg_decode))
        
        # Define Target
#         print(np.shape(data_collect))
#         print(data_collect)
#         PyData = struct.unpack("d", recv_msg)
        Target = np.array(PyData)[:,:3]
#         print(np.shape(Target))

        # Measure Error
        Error = np.array(PyData)[:,3:6]
#         print(np.shape(Error))
        
        # Update model weights
        if Adaptive_switch*count > 0:
            
            model = Update(model,Error)
        
        # Send to Controller
        [s1,s2,s3] = Run_model(model,Target)
        
        # Send input to plant
        send_msg = str([s1,s2,s3])

        print >>sys.stderr, 'sending "%s"' % send_msg
        sent = udp_socket.sendto(send_msg, localaddr)
#         udp_socket.sendto(localaddr)



        count += 1
#     writecsv('UDPOutout.csv',recv_msg_decode)
    np.savetxt('UDPOutout.csv', data_collect, delimiter=',')
    # Close the udp socket.
    udp_socket.close()


if __name__ == "__main__":
    main()