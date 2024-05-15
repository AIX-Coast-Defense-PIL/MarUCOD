import os
import torch
import pytorch_lightning as pl
from collections.abc import Iterable, Mapping

def load_weights(path):
    state_dict = torch.load(path, map_location='cpu')
    if 'model' in state_dict:
        # Loading weights from checkpoint
        state_dict = state_dict['model']

    return state_dict

class ModelExporter(pl.Callback):
    """Exports model weights at the end of the training."""
    def on_fit_end(self, trainer, pl_module):
        export_path = os.path.join(trainer.log_dir, 'weights.pt')
        torch.save(pl_module.model.state_dict(), export_path)

def tensor_map(obj, fn):
    """Map a function to a nested tensor structure.
    Example:
    >>> fn = lambda t: t.to('cpu')
    >>> a = tensor_map(a, fn)
    """

    if torch.is_tensor(obj):
        return fn(obj)

    elif isinstance(obj, Mapping):
        dtype = type(obj)
        res = ((k, tensor_map(v, fn)) for k, v in obj.items())
        res = dtype(res)
        return res
    elif isinstance(obj, Iterable):
        dtype = type(obj)
        res = (tensor_map(v, fn) for v in obj)
        res = dtype(res)
        return res
    else:
        raise TypeError("Invalid type for tensor_map")