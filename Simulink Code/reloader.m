function reloader(module)
% Reloads modules imported in matlab environment so that one doesn't have
% to restart anything to get the most up to date version
% clear classes
mod = py.importlib.import_module(module);
py.importlib.reload(mod)
end