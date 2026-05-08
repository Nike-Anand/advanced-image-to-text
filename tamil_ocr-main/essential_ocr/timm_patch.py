"""
Compatibility patch for timm.layers module
"""
import timm
import sys

# Create timm.layers module if it doesn't exist
if not hasattr(timm, 'layers'):
    import types
    layers_module = types.ModuleType('timm.layers')
    
    # Import common layers from timm.models.layers (old location)
    try:
        from timm.models.layers import *
        # Add all imported items to layers module
        import timm.models.layers as old_layers
        for attr_name in dir(old_layers):
            if not attr_name.startswith('_'):
                setattr(layers_module, attr_name, getattr(old_layers, attr_name))
    except ImportError:
        pass
    
    # Add to sys.modules so it can be imported
    sys.modules['timm.layers'] = layers_module
    timm.layers = layers_module

print("timm.layers compatibility patch applied")