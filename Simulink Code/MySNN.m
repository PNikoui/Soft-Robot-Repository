function MatOut = MySNN(Target,error)
s1 = 0; s2 = 0; s3 = 0;
persistent PyModel

if isempty(PyModel)
    py.importlib.import_module('SNN_in_Py');
    PyModel = py.SNN_in_Py.Network();
    PyModel.Load_model();
    PyTarget = PyModel.Mat2Py(Target);

else
    PyTarget = PyModel.Mat2Py(Target);
    Pyerror = PyModel.Mat2Py(error);
    model = PyModel.Update(Pyerror);
end

OUT = py.SNN_in_Py.Network().Run_model(Target);
MatOut = double(OUT);
% s1 = MatOUT(1)
% s2 = MatOUT(2)
% s3 = MatOUT(3)
end