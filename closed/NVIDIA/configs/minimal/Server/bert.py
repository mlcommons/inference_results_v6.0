import code.common.constants as C
import code.bert.tensorrt.fields as bert_fields
import code.fields.harness as harness_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields

_base_99 = {
    bert_fields.bert_opt_seqlen: 384,
    harness_fields.coalesced_tensor: True,
    model_fields.gpu_batch_size: {'bert': 64},
    harness_fields.gpu_copy_streams: 2,
    harness_fields.gpu_inference_streams: 2,
    model_fields.input_dtype: 'int32',
    model_fields.input_format: 'linear',
    loadgen_fields.server_target_qps: 500,
    model_fields.precision: 'int8',
    harness_fields.tensor_path: 'build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy',
    harness_fields.use_graphs: False,
    bert_fields.use_small_tile_gemm_plugin: False,
    gen_engines_fields.workspace_size: 5368709120,
}

_base_999 = dict(_base_99)
_base_999[model_fields.precision] = 'fp16'

ATOMIC_EXPORTS = {
    C.WorkloadSetting(accuracy_target=C.AccuracyTarget.k_99).short: _base_99,
    C.WorkloadSetting(accuracy_target=C.AccuracyTarget.k_99_9).short: _base_999,
}
