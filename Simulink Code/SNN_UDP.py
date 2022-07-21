#!/usr/bin/env python    
# -*- coding: utf-8 -*

import socket, struct, os
import numpy as np
import matplotlib.pyplot as plt


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

	print("Please open the Simulink file under the current working directory")
	print("The program is waiting until you run the Simulink file.")

	#----------------------------------- Data Receiving ----------------------------------------
	# Using a loop to receive data from Simulink
	while count < 101: # Can be modified by (simulationTime/sampleTime).
		# Start to receive data from Simulink.
		recv_data = udp_socket.recvfrom(1024)
		#print(recv_data)
		# recv_data will return tuple, the first element is DATA, and the second is address information
		recv_msg = recv_data[0]
		send_addr = recv_data[1]
		# Decode the data from Simulink whose type is double and return a tuple
		recv_msg_decode = struct.unpack("d", recv_msg)[0]
		# Restore the data to a list:
		data_collect.append(recv_msg_decode)
		# Set the condition to jump out of this loop ???
		# Print the address information and the received data
		print("Number from MATLAB %s is : %s" % (str(send_addr), recv_msg_decode))
		count += 1
	# Close the udp socket.
	udp_socket.close()

	# ------------------------------------ Visualization ----------------------------------------------- 
	# Set the time axis, 10 is the simulation end time that can be modified by user.
	index = list(np.linspace(0, 10, (len(data_collect))))
	plt.plot(index, data_collect)
	plt.title("Signal Received from Simulink")
	plt.xlabel("Time")
	plt.ylabel("Received Data")
	plt.savefig(os.path.join(path, 'data_figure.png'), dpi=600)
	print("Close the figure to restart.")
	plt.show()

if __name__ == "__main__":
	main()import torch
import SNN

def SNN_model(Target):

    model = SNN.Net(num_inputs=3, num_hidden=32, num_outputs=3,num_steps = num_steps)
    model.load_state_dict(torch.load('SNN_Learned_Weights.pth'))
    
    Target = torch.tensor([Target])
    [s1,s2,s3] = model(Target)
    
    return s1,s2,s3

