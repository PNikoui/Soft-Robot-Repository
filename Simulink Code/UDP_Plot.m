function UDP_Plot(data_collect)
import numpy as np
py.importlib()
%  ------------------------------------ Visualization ----------------------------------------------- 
% Set the time axis, 10 is the simulation end time that can be modified by user.
index = py.list(np.linspace(0, 10, (py.len(data_collect))))
py.plt.plot(index, data_collect)
py.plt.title("Signal Received from Simulink")
py.plt.xlabel("Time")
py.plt.ylabel("Received Data")
% py.plt.savefig(os.path.join(path, 'data_figure.png'), dpi=600)
py.print("Close the figure to restart.")
py.plt.show()
end