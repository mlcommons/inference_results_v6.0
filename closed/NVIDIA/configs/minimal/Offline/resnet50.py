import code.common.constants as C
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields

_base = {
    model_fields.gpu_batch_size: {'resnet50': 2048},
    model_fields.precision: 'int8',
    model_fields.input_dtype: 'int8',
    model_fields.input_format: 'linear',
    loadgen_fields.offline_expected_qps: 40000,
    harness_fields.tensor_path: 'build/preprocessed_data/imagenet/ResNet50/int8_linear',
    harness_fields.map_path: 'data_maps/imagenet/val_map.txt',
    gen_engines_fields.workspace_size: 4294967296,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): _base,
}
