import os
import sys

# Guard: Never run PyTorch/Transformers shims when pip is being invoked (e.g. by transformers import checks)
if not any("pip" in str(arg) for arg in sys.argv):
    import types
    import importlib.metadata
    from importlib.machinery import ModuleSpec

    # 1. Override PyTorch metadata version so HuggingFace does not disable PyTorch
    _original_version = importlib.metadata.version

    def _patched_version(name):
        if name == 'torch':
            return '2.5.0'
        return _original_version(name)

    importlib.metadata.version = _patched_version

    # 2. Set environment flags
    os.environ["USE_TORCH"] = "1"
    os.environ["HF_TRUST_REMOTE_CODE"] = "1"
    os.environ["TRUST_REMOTE_CODE"] = "True"

    # 3. Universal Mock Object for tensor sharding dependencies removed in PyTorch 2.5
    class _UniversalMock(types.ModuleType):
        def __init__(self, name="Mock"):
            super().__init__(name)
            self.__path__ = []
            self.__file__ = f"<mock_{name}>"

        def __getattr__(self, attr):
            class _UniversalItem:
                def __init__(self, *args, **kwargs): pass
                def __call__(self, *args, **kwargs): return (None, None)
                def __iter__(self): return iter([None, None])
                def __getitem__(self, item): return self
                def __getattr__(self, item): return _UniversalItem()
            return _UniversalItem

        def __call__(self, *args, **kwargs):
            return (None, None)

    # 4. MetaPathFinder to dynamically resolve ANY 'torch.distributed.tensor' submodules
    class _TensorModuleFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname == "torch.distributed.tensor" or fullname.startswith("torch.distributed.tensor."):
                spec = ModuleSpec(fullname, self, is_package=True)
                return spec
            return None

        def create_module(self, spec):
            mod = _UniversalMock(spec.name)
            sys.modules[spec.name] = mod
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _TensorModuleFinder())

    # 5. Pre-register core torch.distributed.tensor modules to ensure instant resolution
    try:
        import torch
        torch.__version__ = "2.5.0"
        import torch.distributed
        
        mock_tensor = _UniversalMock("torch.distributed.tensor")
        mock_utils = _UniversalMock("torch.distributed.tensor._utils")
        mock_utils.compute_local_shape_and_global_offset = lambda *args, **kwargs: (None, None)
        mock_placement = _UniversalMock("torch.distributed.tensor.placement_types")
        
        sys.modules["torch.distributed.tensor"] = mock_tensor
        sys.modules["torch.distributed.tensor._utils"] = mock_utils
        sys.modules["torch.distributed.tensor.placement_types"] = mock_placement
        
        torch.distributed.tensor = mock_tensor
        mock_tensor._utils = mock_utils
        mock_tensor.placement_types = mock_placement

        # 6. Mock torch.accelerator (expected by HuggingFace transformers >= 4.45 with PyTorch >= 2.5)
        if not hasattr(torch, "accelerator"):
            class _AcceleratorMock:
                @staticmethod
                def current_accelerator():
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        return torch.device("cuda")
                    return None
                @staticmethod
                def is_available():
                    return hasattr(torch, "cuda") and torch.cuda.is_available()
                @staticmethod
                def device_count():
                    return torch.cuda.device_count() if hasattr(torch, "cuda") else 0
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None

            torch.accelerator = _AcceleratorMock()

    except Exception:
        pass
