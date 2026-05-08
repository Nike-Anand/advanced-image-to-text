"""
Comprehensive timm compatibility fix
"""
import sys
import types

# Create timm.layers package structure
def create_timm_layers():
    # Main layers module
    layers_module = types.ModuleType('timm.layers')
    
    # Sub-modules that might be needed
    submodules = [
        'patch_embed', 'pos_embed', 'norm', 'mlp', 'drop', 
        'attention', 'transformer', 'conv_bn_act', 'create_act'
    ]
    
    for submodule_name in submodules:
        submodule = types.ModuleType(f'timm.layers.{submodule_name}')
        setattr(layers_module, submodule_name, submodule)
        sys.modules[f'timm.layers.{submodule_name}'] = submodule
    
    # Try to import from current timm structure
    try:
        import timm.models.layers as old_layers
        for attr_name in dir(old_layers):
            if not attr_name.startswith('_'):
                attr_value = getattr(old_layers, attr_name)
                setattr(layers_module, attr_name, attr_value)
    except ImportError:
        pass
    
    # Add to sys.modules
    sys.modules['timm.layers'] = layers_module
    
    # Also try to add to timm module
    try:
        import timm
        timm.layers = layers_module
    except:
        pass

create_timm_layers()
print("Comprehensive timm.layers fix applied")