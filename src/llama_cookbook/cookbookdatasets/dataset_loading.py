import importlib.util

def get_hf_datasets():
    """Get the HuggingFace datasets module using absolute import"""
    spec = importlib.util.find_spec('datasets')
    if spec is None:
        raise ImportError("HuggingFace datasets package not found. Please install it with: pip install datasets")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Export the load_dataset function directly
load_dataset = get_hf_datasets().load_dataset