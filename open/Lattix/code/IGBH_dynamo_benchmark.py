#!/usr/bin/env python3
"""IGBH hardware-aware benchmark using integrated runtime simulation data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch
import torch_geometric.data as pyg_data
from torch_geometric.utils import k_hop_subgraph

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402
from dynamo_lattix.dynamo_cora import DecomposedAnalogGCN
from dynamo_lattix.integrated_loader import load_integrated_runtime_state
from dynamo_lattix.integrated_runtime import run_integrated_forward_workload

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGBH_DATA_PATH = ROOT / "data" / "IGBH" / "tiny" / "processed"
RUN_DIR = (
    ROOT
    / "outputs"
    / "private_end_to_end_runs"
    / "integrated_runtime_audit_gf180_rebuilt_recalibrated_fresh_final2"
)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_sha256_or_none(path: Path) -> str | None:
    return _file_sha256(path) if path.exists() and path.is_file() else None


def _node_ids_sha256(node_ids: np.ndarray) -> str:
    arr = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _print_l1_manifest_mismatch(cached_manifest: dict, expected_manifest: dict) -> None:
    for key in ("schema", "run_root", "runtime_shape", "read_v"):
        if cached_manifest.get(key) != expected_manifest.get(key):
            print(
                f" -> cache {key}={cached_manifest.get(key)!r}, "
                f"expected {expected_manifest.get(key)!r}"
            )

    cached_files = cached_manifest.get("files", {})
    expected_files = expected_manifest.get("files", {})
    if isinstance(cached_files, dict) and isinstance(expected_files, dict):
        for name in sorted(set(cached_files) | set(expected_files)):
            if cached_files.get(name) != expected_files.get(name):
                print(
                    f" -> cache file hash mismatch for {name}: "
                    f"cache={cached_files.get(name)}, expected={expected_files.get(name)}"
                )


def _write_benchmark_authenticity_logs(
    *,
    log_dir: Path,
    results: dict[str, Any],
    args: argparse.Namespace,
    runtime_root: Path,
    results_path: Path,
    cache_path: Path,
    cache_manifest: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).isoformat()
    command_line = " ".join(sys.argv)
    artifacts = {
        name: {
            "path": str(path),
            "sha256": _file_sha256_or_none(path),
            "exists": path.exists(),
        }
        for name, path in artifact_paths.items()
    }
    payload = {
        "schema": "dynamo.igbh.non_loadgen_runtime_benchmark.measurements.v1",
        "generated_utc": generated_utc,
        "benchmark": "IGBH_dynamo_benchmark.py",
        "official_loadgen_run": False,
        "authenticity_note": (
            "This benchmark does not invoke MLPerf LoadGen. It records the Dynamo "
            "runtime-authoritative graph workload, measured live through "
            "dynamo_lattix.integrated_runtime, with hashes and provenance for review."
        ),
        "command_line": command_line,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "runtime_root": str(runtime_root),
        "runtime_authority": "dynamo_lattix.integrated_runtime.run_integrated_forward_workload",
        "transient_steps": int(args.transient_steps),
        "validation_start": int(args.validation_start),
        "requested_count": int(args.count),
        "subgraph_hops": int(args.subgraph_hops),
        "used_rebuild_l1_cache": bool(args.rebuild_l1_cache),
        "cache_path": str(cache_path),
        "cache_manifest": cache_manifest,
        "artifacts": artifacts,
        "results_json": str(results_path),
        "results_sha256": _file_sha256_or_none(results_path),
        "results": results,
    }
    measurements_path = log_dir / "dynamo_measurements.json"
    measurements_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        "Dynamo IGBH Runtime Benchmark Summary",
        "====================================",
        f"Generated UTC: {generated_utc}",
        "Official MLPerf LoadGen run: no",
        "Runtime authority: dynamo_lattix.integrated_runtime.run_integrated_forward_workload",
        f"Command: {command_line}",
        f"Samples: {results.get('samples')}",
        f"Accuracy: {float(results.get('accuracy', 0.0)) * 100.0:.3f}%",
        f"Average runtime tile latency: {float(results.get('avg_latency_ps', 0.0)):.6f} ps",
        f"Physical runtime QPS: {float(results.get('physical_qps', 0.0)):,.0f}",
        f"Full workload Dynamo NPU energy: {float(results.get('full_workload_npu_energy_j', 0.0)) * 1e9:.9f} nJ",
        f"Full workload Dynamo NPU cost: {float(results.get('full_workload_npu_fj_per_op', 0.0)):.9f} fJ/op",
        f"Full workload Dynamo NPU efficiency: {float(results.get('full_workload_npu_tops_w', 0.0)):,.6f} TOPS/W",
        f"Target-path material efficiency: {float(results.get('target_path_material_tops_w', 0.0)):,.6f} TOPS/W",
        f"Operation basis: {json.dumps(results.get('operation_basis'), sort_keys=True)}",
        f"Energy scope: {results.get('energy_scope')}",
        f"Measurements JSON: {measurements_path}",
        f"Results JSON SHA256: {payload['results_sha256']}",
    ]
    summary_path = log_dir / "dynamo_mlperf_log_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    detail_lines = [
        "Dynamo IGBH Runtime Benchmark Detail",
        "===================================",
        f"Generated UTC: {generated_utc}",
        "This is a non-LoadGen Dynamo runtime benchmark artifact for MLCommons review.",
        "It is intentionally not named mlperf_log_summary.txt/mlperf_log_detail.txt to avoid implying an official LoadGen run.",
        "",
        "Command",
        command_line,
        "",
        "Runtime Root",
        str(runtime_root),
        "",
        "Cache Manifest",
        json.dumps(cache_manifest, indent=2, sort_keys=True),
        "",
        "Artifacts",
        json.dumps(artifacts, indent=2, sort_keys=True),
        "",
        "Results",
        json.dumps(results, indent=2, sort_keys=True),
    ]
    detail_path = log_dir / "dynamo_mlperf_log_detail.txt"
    detail_path.write_text("\n".join(detail_lines) + "\n", encoding="utf-8")
    return {
        "summary": str(summary_path),
        "detail": str(detail_path),
        "measurements": str(measurements_path),
    }


def _load_weight_tensor(path: Path) -> torch.Tensor | np.ndarray:
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(weights, (torch.Tensor, np.ndarray)):
        return weights
    raise TypeError(f"Expected tensor weights in {path}, got {type(weights).__name__}")


def _logical_tile_output(input_np: np.ndarray, tile) -> np.ndarray:
    x = np.asarray(input_np, dtype=np.float64)
    lattix = tile.lattix
    if hasattr(lattix, "positive") and hasattr(lattix, "negative"):
        weight = _as_numpy(lattix.weight_matrix()).astype(np.float64)
    else:
        weight = _as_numpy(tile.read_weight_tensor()).astype(np.float64)
    out = x @ weight.T
    if tile.bias is not None:
        out += tile.bias.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64)


def _event_energy_and_latency(trace: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    arrays = trace["forward_cycle_arrays"]
    source_power_w = np.asarray(arrays["event_source_power_w"], dtype=np.float64)
    analog_ready_ns = np.asarray(arrays["event_analog_ready_ns"], dtype=np.float64)
    service_ns = np.asarray(arrays["event_service_ns"], dtype=np.float64)
    # Match integrated_runtime's material VMM energy definition:
    # source-side power integrated over the live analog read aperture.
    energy_j = source_power_w * analog_ready_ns * 1.0e-9
    return energy_j, service_ns


class RuntimeTileSimulator:
    """Small benchmark context; physics execution lives in integrated_runtime."""

    def __init__(self, runtime_state) -> None:
        self.runtime_state = runtime_state

    def conductance_matrix(self, tile) -> np.ndarray:
        lattix = tile.lattix
        if hasattr(lattix, "positive") and hasattr(lattix, "negative"):
            return _as_numpy(lattix.positive.g_total()).astype(np.float64).T
        return _as_numpy(lattix.g_total()).astype(np.float64).T

    def ops_per_sample(self, tiles: list) -> int:
        return int(
            2 * sum(np.prod(self.conductance_matrix(tile).shape) for tile in tiles)
        )

    def tile_ops_mac(self, tile) -> int:
        rows, cols = self.conductance_matrix(tile).shape
        return int(rows * cols)

    def tile_interface_energy_j(self, tile) -> float:
        rows, cols = self.conductance_matrix(tile).shape
        cmos = self.runtime_state.static.cmos_interface
        c_interface_f = float(rows + cols) * float(cmos.c_gate_f)
        return 0.5 * c_interface_f * float(cmos.v_dd) ** 2 * float(cmos.logic_activity_factor)


def _build_l1_cache_manifest(
    run_dir: Path,
    runtime_state: Any,
    omega_vmm1_path: Path,
    pca_path: Path,
    pca_mean_path: Path,
) -> dict:
    constants_dir = run_dir / "runtime_constants"
    tracked_files = {
        "omega_vmm1.pt": omega_vmm1_path,
        "igbh_pca_components.npy": pca_path,
        "igbh_pca_mean.npy": pca_mean_path,
        "runtime_constants/manifest.json": constants_dir / "manifest.json",
        "runtime_constants/integrated_runtime_summary.json": (
            constants_dir / "integrated_runtime_summary.json"
        ),
        "runtime_constants/realistic_parasitics.json": (
            constants_dir / "realistic_parasitics.json"
        ),
        "runtime_constants/crossbar_capacitance_map.pt": (
            constants_dir / "crossbar_capacitance_map.pt"
        ),
        "runtime_constants/live_capacitance_calibration.json": (
            constants_dir / "live_capacitance_calibration.json"
        ),
    }
    return {
        "schema": "dynamo.igbh.l1_runtime_cache.v3",
        "run_root": str(run_dir),
        "runtime_shape": list(runtime_state.static.shape),
        "read_v": float(runtime_state.static.read_v),
        "weight_mapping": "conductance_sustained",
        "runtime_authority": "dynamo_lattix.integrated_runtime.run_integrated_forward_workload",
        "runtime_forward_workload_adapter": True,
        "sustained_branches_physicalized": True,
        "energy_accounting_version": "live_npu_full_v4_analog_ready_energy_window",
        "files": {
            name: _file_sha256(path) if path.exists() else None
            for name, path in tracked_files.items()
        },
    }


def _load_l1_cache(cache_path: Path, expected_manifest: dict, num_nodes: int):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_manifest = json.loads(str(cached["manifest_json"].item()))
            if cached_manifest != expected_manifest:
                print(" -> Ignoring stale Layer 1 runtime cache manifest.")
                _print_l1_manifest_mismatch(cached_manifest, expected_manifest)
                return None
            pre_aggregated_x1 = cached["pre_aggregated_x1"]
            l1_energy_j = cached["l1_energy_j"]
            l1_latency_ns = cached["l1_latency_ns"]
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f" -> Ignoring unreadable Layer 1 runtime cache: {exc}")
        return None

    if pre_aggregated_x1.shape[0] != num_nodes:
        print(" -> Ignoring Layer 1 runtime cache with wrong node count.")
        return None
    print(f" -> Loaded runtime-validated Layer 1 cache from {cache_path.name}")
    return pre_aggregated_x1, l1_energy_j, l1_latency_ns


def _precompute_l1(
    *,
    model,
    simulator: RuntimeTileSimulator,
    features: torch.Tensor,
    pca_matrix: torch.Tensor,
    pca_mean: torch.Tensor,
    adj_hat: torch.Tensor,
    cache_path: Path,
    cache_manifest: dict,
    node_indices: np.ndarray | None = None,
    transient_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = int(features.shape[0])
    solve_node_indices = (
        np.arange(num_nodes, dtype=np.int64)
        if node_indices is None
        else np.asarray(node_indices, dtype=np.int64)
    )
    batch_size = 500
    vmm1_cols = simulator.conductance_matrix(model.vmm1).shape[1]
    x1_vmm = np.zeros((num_nodes, vmm1_cols), dtype=np.float32)
    l1_energy_j = np.zeros(num_nodes, dtype=np.float64)
    l1_latency_ns = np.zeros(num_nodes, dtype=np.float64)
    features_cpu = features.cpu()
    pca_cpu = pca_matrix.cpu()
    pca_mean_cpu = pca_mean.cpu()

    from tqdm import tqdm

    for i in tqdm(
        range(0, solve_node_indices.size, batch_size), desc="Physical Solve (L1)"
    ):
        batch_nodes = solve_node_indices[i : i + batch_size]
        batch_raw = features_cpu[torch.from_numpy(batch_nodes).long()]
        batch_x = ((batch_raw - pca_mean_cpu) @ pca_cpu).numpy()
        batch_out = _logical_tile_output(batch_x, model.vmm1)
        trace = run_integrated_forward_workload(
            state=simulator.runtime_state,
            input_vectors=batch_x,
            reference_vectors=batch_out,
            lattix=model.vmm1.lattix,
            workload_name="igbh_l1_subgraph",
            transient_steps=transient_steps,
        )
        batch_energy_j, batch_latency_ns = _event_energy_and_latency(trace)
        x1_vmm[batch_nodes] = batch_out.astype(np.float32)
        l1_energy_j[batch_nodes] = batch_energy_j
        l1_latency_ns[batch_nodes] = batch_latency_ns

    x1_aggregated = torch.sparse.mm(adj_hat, torch.from_numpy(x1_vmm).float())
    pre_aggregated_x1 = torch.relu(x1_aggregated).numpy()

    np.savez_compressed(
        cache_path,
        pre_aggregated_x1=pre_aggregated_x1,
        l1_energy_j=l1_energy_j,
        l1_latency_ns=l1_latency_ns,
        manifest_json=np.array(json.dumps(cache_manifest, sort_keys=True)),
    )
    print(f" -> Cached runtime-validated Layer 1 precompute to {cache_path}")
    return pre_aggregated_x1, l1_energy_j, l1_latency_ns


def main():
    parser = argparse.ArgumentParser(
        description="Runtime-authoritative IGBH Dynamo benchmark"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1024,
        help="Number of validation nodes to benchmark from the validation window.",
    )
    parser.add_argument("--validation-start", type=int, default=60000)
    parser.add_argument(
        "--subgraph-hops",
        type=int,
        default=2,
        help="Neighborhood hops to retain around the selected validation nodes.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small validation-window pipeline test instead of sign-off.",
    )
    parser.add_argument(
        "--smoke-nodes",
        type=int,
        default=16,
        help="Number of validation nodes to run in --smoke mode.",
    )
    parser.add_argument(
        "--rebuild-l1-cache",
        action="store_true",
        help="Force rebuilding the subset L1 physical cache even if one exists.",
    )
    parser.add_argument(
        "--transient-steps",
        type=int,
        default=10,
        help="Internal transient increments per runtime read event.",
    )
    parser.add_argument(
        "--benchmark-log-dir",
        type=Path,
        default=ROOT / "outputs" / "igbh_dynamo_benchmark_logs",
        help="Directory for non-LoadGen Dynamo authenticity logs.",
    )
    args = parser.parse_args()
    if args.transient_steps < 1:
        raise ValueError(f"--transient-steps must be >= 1, got {args.transient_steps}")
    sample_count = args.smoke_nodes if args.smoke else args.count

    print("============================================================")
    print("REALISTIC DYNAMO-NPU PERFORMANCE BENCHMARK (RUNTIME-AUTHORITATIVE)")
    print("Target: IGBH-Tiny Dataset (Validation Window, Performance Only)")
    print(f"Environment: {RUN_DIR.name}")
    print("============================================================")

    runtime = load_integrated_runtime_state(RUN_DIR)
    simulator = RuntimeTileSimulator(runtime)

    print("\n[1/4] Loading IGBH-Tiny graph and extracting validation subgraph...")
    x_np = np.load(IGBH_DATA_PATH / "paper" / "node_feat.npy", mmap_mode="r")
    y_np = np.load(IGBH_DATA_PATH / "paper" / "node_label_19.npy", mmap_mode="r")
    edge_np = np.load(
        IGBH_DATA_PATH / "paper__cites__paper" / "edge_index.npy",
        mmap_mode="r",
    )
    val_start = int(args.validation_start)
    val_end = val_start + int(sample_count)
    full_num_nodes = int(x_np.shape[0])
    if val_start < 0 or val_end > full_num_nodes:
        raise ValueError(
            f"Requested validation slice [{val_start}, {val_end}) exceeds node count {full_num_nodes}"
        )
    edge_index = torch.from_numpy(edge_np[:].copy()).long().t().contiguous()
    target_global_nodes = torch.arange(val_start, val_end, dtype=torch.long)
    subgraph_nodes, subgraph_edge_index, target_local_nodes, _ = k_hop_subgraph(
        target_global_nodes,
        num_hops=int(args.subgraph_hops),
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=full_num_nodes,
    )
    features = torch.from_numpy(x_np[subgraph_nodes.cpu().numpy()].copy()).float()
    data = pyg_data.Data(
        x=features.to(DEVICE), edge_index=subgraph_edge_index.to(DEVICE)
    )
    pca_path = ROOT / "igbh_pca_components.npy"
    pca_mean_path = ROOT / "igbh_pca_mean.npy"
    if not pca_path.exists():
        raise FileNotFoundError(
            f"PCA matrix not found at {pca_path}. Run training first."
        )
    if not pca_mean_path.exists():
        raise FileNotFoundError(
            f"PCA mean not found at {pca_mean_path}. Run training first."
        )
    pca_matrix = torch.from_numpy(np.load(pca_path)).float()
    pca_mean = torch.from_numpy(np.load(pca_mean_path)).float()
    data.x = ((features - pca_mean) @ pca_matrix).to(DEVICE)
    print(f" -> Applied synchronized PCA projection (1024 -> {data.x.shape[1]})")
    print(
        " -> Subgraph extraction: "
        f"{int(target_local_nodes.numel())} target nodes, "
        f"{int(subgraph_nodes.numel())} total nodes, "
        f"{int(args.subgraph_hops)} hops"
    )

    print("\n[2/4] Instantiating Analog NPU and Physicalizing ALL Layers...")
    trained_dir = ROOT / "outputs" / "igbh_trained_model"
    model = DecomposedAnalogGCN(
        in_features=int(data.x.shape[1]),
        num_classes=19,
        hidden_dim=64,
        n_blocks=runtime.static.spec.n_blocks,
        spec=runtime.static.spec,
        weight_mapping="conductance_sustained",
    )
    resolved_mapping = getattr(model.vmm1, "weight_mapping", "affine_sigma")

    omega_paths = {
        "vmm1": trained_dir / "omega_vmm1.pt",
        "mid": trained_dir / "omega_mid.pt",
        "vmm2": trained_dir / "omega_vmm2.pt",
    }
    missing_omegas = [str(path) for path in omega_paths.values() if not path.exists()]
    if missing_omegas:
        raise FileNotFoundError(
            "Missing trained analog tile weights required for sign-off: "
            + ", ".join(missing_omegas)
        )

    with torch.no_grad():
        model.vmm1.set_weights(_load_weight_tensor(omega_paths["vmm1"]), None)
        model.vmm_mid.set_weights(_load_weight_tensor(omega_paths["mid"]), None)
        model.vmm2.set_weights(_load_weight_tensor(omega_paths["vmm2"]), None)
    model = model.to(DEVICE)
    model.eval()
    model.set_adjacency(
        subgraph_edge_index, int(features.shape[0]), torch.device("cpu")
    )
    adj_hat = model.adj_hat
    if adj_hat is None:
        raise RuntimeError("Adjacency precomputation failed for the selected subgraph.")

    print(
        f" -> Analog Core: N={runtime.static.spec.n_blocks}, mapping={resolved_mapping}"
    )
    print(f" -> Runtime constants: {RUN_DIR / 'runtime_constants'}")

    print("\n[3/4] Running runtime-backed physical inference on validation split...")
    cache_stem = (
        f"igbh_l1_runtime_cache_smoke_{val_start}_{sample_count}_{int(args.subgraph_hops)}hop"
        if args.smoke
        else f"igbh_l1_runtime_cache_subset_{val_start}_{sample_count}_{int(args.subgraph_hops)}hop"
    )
    cache_path = ROOT / "outputs" / f"{cache_stem}.npz"
    cache_manifest = _build_l1_cache_manifest(
        RUN_DIR, runtime, omega_paths["vmm1"], pca_path, pca_mean_path
    )
    cache_manifest = {
        **cache_manifest,
        "cache_mode": "smoke" if args.smoke else "subset",
        "validation_start": val_start,
        "validation_end": val_end,
        "sample_count": int(sample_count),
        "subgraph_hops": int(args.subgraph_hops),
        "transient_steps": int(args.transient_steps),
        "subgraph_node_count": int(subgraph_nodes.numel()),
        "target_node_count": int(target_local_nodes.numel()),
        "target_nodes_sha256": _node_ids_sha256(
            target_global_nodes.cpu().numpy().astype(np.int64)
        ),
        "subgraph_nodes_sha256": _node_ids_sha256(
            subgraph_nodes.cpu().numpy().astype(np.int64)
        ),
    }
    if args.rebuild_l1_cache and cache_path.exists():
        cache_path.unlink()
    cached_l1 = _load_l1_cache(cache_path, cache_manifest, int(features.shape[0]))
    if cached_l1 is None:
        pre_aggregated_x1, l1_energy_j, l1_latency_ns = _precompute_l1(
            model=model,
            simulator=simulator,
            features=features,
            pca_matrix=pca_matrix,
            pca_mean=pca_mean,
            adj_hat=adj_hat,
            cache_path=cache_path,
            cache_manifest=cache_manifest,
            transient_steps=int(args.transient_steps),
        )
    else:
        pre_aggregated_x1, l1_energy_j, l1_latency_ns = cached_l1

    target_indices = np.asarray(target_local_nodes.cpu().tolist(), dtype=np.int64)
    target_x1 = np.asarray(pre_aggregated_x1[target_indices], dtype=np.float64)
    out_mid = _logical_tile_output(target_x1, model.vmm_mid)
    mid_trace = run_integrated_forward_workload(
        state=simulator.runtime_state,
        input_vectors=target_x1,
        reference_vectors=out_mid,
        lattix=model.vmm_mid.lattix,
        workload_name="igbh_mid_target_samples",
        transient_steps=int(args.transient_steps),
    )
    mid_energy_j, mid_latency_ns = _event_energy_and_latency(mid_trace)
    x2_np = np.maximum(out_mid + target_x1, 0.0)
    out_vmm2 = _logical_tile_output(x2_np, model.vmm2)
    out_trace = run_integrated_forward_workload(
        state=simulator.runtime_state,
        input_vectors=x2_np,
        reference_vectors=out_vmm2,
        lattix=model.vmm2.lattix,
        workload_name="igbh_output_target_samples",
        transient_steps=int(args.transient_steps),
    )
    out_energy_j, out_latency_ns = _event_energy_and_latency(out_trace)

    target_mid_output_material_energy_j = float(np.sum(mid_energy_j) + np.sum(out_energy_j))
    target_path_material_energy_j = float(
        np.sum(l1_energy_j[target_indices]) + target_mid_output_material_energy_j
    )
    total_latency_ns = float(
        np.sum(l1_latency_ns[target_indices]) + np.sum(mid_latency_ns) + np.sum(out_latency_ns)
    )
    samples = int(target_local_nodes.numel())
    correct = 0
    total_valid = 0
    for local_idx in range(samples):
        label = y_np[int(target_global_nodes[local_idx])]
        if np.isfinite(label) and label >= 0:
            pred = np.argmax(out_vmm2[local_idx])
            if int(pred) == int(label):
                correct += 1
            total_valid += 1

    print(" -> Runtime physical performance loop complete.")

    print("\n[4/4] Dynamic Physics-Derived Performance Scorecard")
    print("============================================================")
    avg_latency_ps = (total_latency_ns / samples) * 1000.0
    physical_qps = 1.0 / (avg_latency_ps * 1.0e-12) if avg_latency_ps > 0 else 0.0
    physical_qps_billion = physical_qps / 1.0e9
    l1_shape = tuple(int(v) for v in simulator.conductance_matrix(model.vmm1).shape)
    mid_shape = tuple(int(v) for v in simulator.conductance_matrix(model.vmm_mid).shape)
    out_shape = tuple(int(v) for v in simulator.conductance_matrix(model.vmm2).shape)
    l1_ops_mac = int(np.prod(l1_shape))
    mid_ops_mac = int(np.prod(mid_shape))
    out_ops_mac = int(np.prod(out_shape))
    ops_per_sample = int(2 * (l1_ops_mac + mid_ops_mac + out_ops_mac))
    op_basis = {
        "macs_per_target_sample": {
            "vmm1": l1_ops_mac,
            "vmm_mid": mid_ops_mac,
            "vmm2": out_ops_mac,
            "total": l1_ops_mac + mid_ops_mac + out_ops_mac,
        },
        "ops_per_target_sample": ops_per_sample,
        "op_convention": "1 MAC = 2 ops",
        "tile_shapes_input_by_output": {
            "vmm1": list(l1_shape),
            "vmm_mid": list(mid_shape),
            "vmm2": list(out_shape),
        },
    }
    target_path_ops = samples * ops_per_sample
    target_path_tops_w = (target_path_ops / 1.0e12) / target_path_material_energy_j if target_path_material_energy_j > 0 else 0.0
    target_path_fj_per_op = (target_path_material_energy_j * 1.0e15) / max(target_path_ops, 1)

    full_workload_ops_mac = float(int(subgraph_nodes.numel()) * l1_ops_mac + samples * (mid_ops_mac + out_ops_mac))
    full_workload_ops = 2.0 * full_workload_ops_mac
    subgraph_l1_material_energy_j = float(np.sum(l1_energy_j))
    full_workload_material_energy_j = subgraph_l1_material_energy_j + target_mid_output_material_energy_j
    interface_cmos_energy_j = (
        int(subgraph_nodes.numel()) * simulator.tile_interface_energy_j(model.vmm1)
        + samples * (
            simulator.tile_interface_energy_j(model.vmm_mid)
            + simulator.tile_interface_energy_j(model.vmm2)
        )
    )
    full_workload_npu_energy_j = full_workload_material_energy_j + interface_cmos_energy_j
    full_workload_npu_tops_w = (full_workload_ops / 1.0e12) / full_workload_npu_energy_j if full_workload_npu_energy_j > 0 else 0.0
    full_workload_npu_fj_per_op = (full_workload_npu_energy_j * 1.0e15) / max(full_workload_ops, 1.0)
    accuracy = float(correct) / total_valid if total_valid > 0 else 0.0
    results = {
        "mode": "smoke" if args.smoke else "performance",
        "samples": samples,
        "accuracy": accuracy,
        "validation_start": val_start,
        "validation_end": val_end,
        "subgraph_hops": int(args.subgraph_hops),
        "transient_steps": int(args.transient_steps),
        "subgraph_node_count": int(subgraph_nodes.numel()),
        "avg_latency_ps": avg_latency_ps,
        "avg_latency_ns": avg_latency_ps / 1000.0,
        "physical_qps": physical_qps,
        "qps_billion": physical_qps_billion,
        "target_path_material_energy_j": target_path_material_energy_j,
        "target_path_fj_per_op": target_path_fj_per_op,
        "target_path_material_tops_w": target_path_tops_w,
        "target_path_ops": target_path_ops,
        "full_workload_material_energy_j": full_workload_material_energy_j,
        "full_workload_interface_cmos_energy_j": interface_cmos_energy_j,
        "full_workload_npu_energy_j": full_workload_npu_energy_j,
        "full_workload_npu_fj_per_op": full_workload_npu_fj_per_op,
        "full_workload_npu_tops_w": full_workload_npu_tops_w,
        "full_workload_ops": full_workload_ops,
        "full_workload_ops_mac": full_workload_ops_mac,
        "subgraph_l1_material_energy_j": subgraph_l1_material_energy_j,
        "target_mid_output_material_energy_j": target_mid_output_material_energy_j,
        "energy_scope": "Dynamo NPU only; live benchmark material energy plus live-modeled NPU interface CMOS energy; excludes host CPU/GPU/platform power",
        "transient_steps": int(args.transient_steps),
        "ops_per_sample": ops_per_sample,
        "operation_basis": op_basis,
        "total_energy_j": full_workload_npu_energy_j,
        "fj_per_op": full_workload_npu_fj_per_op,
        "tops_w": full_workload_npu_tops_w,
    }
    results_path = ROOT / "igbh_dynamo_benchmark_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    benchmark_log_paths = _write_benchmark_authenticity_logs(
        log_dir=args.benchmark_log_dir,
        results=results,
        args=args,
        runtime_root=RUN_DIR,
        results_path=results_path,
        cache_path=cache_path,
        cache_manifest=cache_manifest,
        artifact_paths={
            "results_json": results_path,
            "l1_cache": cache_path,
            "igbh_features": IGBH_DATA_PATH / "paper" / "node_feat.npy",
            "igbh_labels": IGBH_DATA_PATH / "paper" / "node_label_19.npy",
            "igbh_edges": IGBH_DATA_PATH / "paper__cites__paper" / "edge_index.npy",
            "pca_components": pca_path,
            "pca_mean": pca_mean_path,
            "omega_vmm1": omega_paths["vmm1"],
            "omega_mid": omega_paths["mid"],
            "omega_vmm2": omega_paths["vmm2"],
            "runtime_manifest": RUN_DIR / "runtime_constants" / "manifest.json",
            "runtime_summary": RUN_DIR / "runtime_constants" / "integrated_runtime_summary.json",
            "runtime_parasitics": RUN_DIR / "runtime_constants" / "realistic_parasitics.json",
            "runtime_capacitance_map": RUN_DIR / "runtime_constants" / "crossbar_capacitance_map.pt",
            "runtime_capacitance_calibration": RUN_DIR / "runtime_constants" / "live_capacitance_calibration.json",
        },
    )

    print(f"Accuracy:                                     {accuracy * 100:.2f}%")
    print(f"Samples:                                      {samples:,}")
    print(f"Average Runtime Tile Latency:                 {avg_latency_ps:.2f} ps")
    print(f"Physical Runtime QPS:                         {physical_qps:,.0f}")
    print(
        f"Target-Path Material Energy:                  {target_path_material_energy_j * 1e9:.6f} nJ"
    )
    print(
        f"Full Workload Dynamo NPU Energy:              {full_workload_npu_energy_j * 1e9:.6f} nJ"
    )
    print(f"Full Workload Dynamo NPU Cost:                {full_workload_npu_fj_per_op:.4f} fJ/op")
    print(f"Full Workload Dynamo NPU Efficiency:          {full_workload_npu_tops_w:,.0f} TOPS/W")
    print(f"Target-Path Material Efficiency:              {target_path_tops_w:,.0f} TOPS/W")
    print(
        "Operation Basis:                              "
        f"{ops_per_sample} ops/target sample "
        f"(2x[{l1_shape[0]}x{l1_shape[1]} + {mid_shape[0]}x{mid_shape[1]} + {out_shape[0]}x{out_shape[1]}]); "
        f"{full_workload_ops:,.0f} full-workload ops"
    )
    print(f"Results JSON:                                 {results_path}")
    print(f"Benchmark Summary Log:                       {benchmark_log_paths['summary']}")
    print(f"Benchmark Detail Log:                        {benchmark_log_paths['detail']}")
    print(f"Benchmark Measurements JSON:                 {benchmark_log_paths['measurements']}")
    print("Provenance: IGBH workload events executed through dynamo_lattix.integrated_runtime")
    print("============================================================")


if __name__ == "__main__":
    main()
