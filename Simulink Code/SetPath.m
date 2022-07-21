function PathtoPy = SetPath(Module)
PathtoPy = fileparts(which(Module));
if count(py.sys.path,PathtoPy) == 0
    insert(py.sys.path,int32(0),PathtoPy);
    disp(PathtoPy)
end
end