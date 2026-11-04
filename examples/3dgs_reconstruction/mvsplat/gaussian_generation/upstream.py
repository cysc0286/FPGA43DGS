"""Import the unchanged upstream encoder without training/visualizer registries.

The three namespace packages below point at real upstream files. They omit only
registry __init__.py execution, which eagerly imports Lightning, dataset loaders
and logging visualizers. No tensor operation, layer or weight is replaced.
"""
import importlib
import importlib.machinery
import sys
import types


def load_encoder(vendor):
    prefix = "_heterogs_mvsplat"
    source = vendor.resolve() / "src"
    for suffix in ("", ".dataset", ".model.encoder"):
        name = prefix + suffix
        path = source.joinpath(*suffix.strip(".").split(".")) if suffix else source
        if name in sys.modules:
            if list(sys.modules[name].__path__) != [str(path)]:
                raise RuntimeError("Different MVSplat source already imported")
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    encoder = importlib.import_module(prefix + ".model.encoder.encoder_costvolume")
    global_cfg = importlib.import_module(prefix + ".global_cfg")
    return encoder.EncoderCostVolume, global_cfg.set_cfg
