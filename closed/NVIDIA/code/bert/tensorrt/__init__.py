from code.ops.harness import ExecutableHarness
from .constants import BERTComponent as Component

def _get_builder():
    from .builder import BERTBuilder
    return BERTBuilder

class _LazyBuilderMeta(type):
    _real = None
    def __instancecheck__(cls, instance):
        if cls._real is None: cls._real = _get_builder()
        return isinstance(instance, cls._real)
    def __subclasscheck__(cls, subclass):
        if cls._real is None: cls._real = _get_builder()
        return issubclass(subclass, cls._real) or subclass is cls._real
    def __call__(cls, *args, **kwargs):
        if cls._real is None: cls._real = _get_builder()
        return cls._real(*args, **kwargs)

class LazyBERTBuilder(metaclass=_LazyBuilderMeta):
    pass

COMPONENT_MAP = {Component.BERT: LazyBERTBuilder}

VALID_COMPONENT_SETS = {"gpu": [{Component.BERT}]}

class BenchmarkHarnessOp(ExecutableHarness):
    def __init__(self):
        super().__init__(executable_fpath="./build/bin/harness_bert")

    def build_flags(self, user_conf, engine_index):
        flags = super().build_flags(user_conf, engine_index)
        # harness_bert does not support gpu_engine_batch_size
        flags.pop("gpu_engine_batch_size", None)
        # Add required flags for harness_bert
        flags["scenario"] = engine_index.wl.scenario.valstr
        flags["model"] = "bert"
        flags["tensor_path"] = "build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy"
        flags["map_path"] = "data_maps/squad/val_map.txt"
        return flags
