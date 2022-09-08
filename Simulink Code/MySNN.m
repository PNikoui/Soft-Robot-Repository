function MatOut = MySNN(Target,error, Threshold)
s1 = 0; s2 = 0; s3 = 0;
persistent PyModel

if isempty(PyModel)
    py.importlib.import_module('SNN_in_Py');
    PyModel = py.SNN_in_Py.Network(2,64,1);
%     PyModel.Load_model();
%     PyTarget = PyModel.Mat2Py(Target);

else
%     PyTarget = PyModel.Mat2Py(Target);
%     PyModel = py.SNN_in_Py.Network();
    Pyerror = PyModel.Mat2Py(error);
    PyThreshold = PyModel.Mat2Py(Threshold);
    PyModel.Update(Pyerror,PyThreshold);
end

% OUT = py.SNN_in_Py.Network().Run_model(Target)
OUT = PyModel.Run_model(Target);
% details(OUT)
MatOut = double(OUT);
% s1 = MatOUT(1)
% s2 = MatOUT(2)
% s3 = MatOUT(3)
end