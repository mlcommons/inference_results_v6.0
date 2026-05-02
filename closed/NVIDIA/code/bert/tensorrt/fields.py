from nvmitten.configurator import Field

bert_opt_seqlen = Field(
    "bert_opt_seqlen",
    description="Opt sequence length for BERT TRT optimization profile",
    from_string=int)

graphs_max_seqlen = Field(
    "graphs_max_seqlen",
    description="Maximum sequence length for CUDA graphs in BERT",
    from_string=int)

use_small_tile_gemm_plugin = Field(
    "use_small_tile_gemm_plugin",
    description="Enable Small Tile GEMM plugin for BERT",
    from_string=bool)
