from importlib import import_module as _imp
from graphembed_rs import *      

# add the helper into the same namespace
from .load_utils import *                   

del _imp    
