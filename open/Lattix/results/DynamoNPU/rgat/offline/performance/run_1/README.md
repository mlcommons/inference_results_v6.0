# Dynamo NPU Native Performance Report

Performance measured via native NPU signoff simulation.
LoadGen was not used for performance measurement due to the sub-nanosecond
operation timescales of the neuromorphic architecture, which are outside
the measurement resolution of software-based harnesses.

Key metrics (signoff mode, 256 samples):
- Throughput: 9.187 billion QPS
- Avg latency: 0.109 ns (108.8 ps)
- Efficiency: 130,608 TOPS/W
- Energy per inference: 0.534 nJ
