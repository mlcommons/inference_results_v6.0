import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast

import mlperf_loadgen as lg
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

# IGBH-Tiny Path (Absolute path to the processed files)
IGBH_DATA_PATH = "/home/lattix/Dynamo2SwappingVersions/Dynamo2/DistilledDynamo2/data/IGBH/tiny/processed"

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ruff: noqa: E402
from dynamo_lattix.dynamo_cora import DecomposedAnalogGCN
from dynamo_lattix.integrated_runtime import (
    run_integrated_forward_workload,
)

from dynamo_lattix.integrated_loader import load_integrated_runtime_state

def _load_weight_tensor(path: Path) -> torch.Tensor | np.ndarray:
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(weights, (torch.Tensor, np.ndarray)):
        return weights
    raise TypeError(f"Expected tensor weights in {path}, got {type(weights).__name__}")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _node_ids_sha256(node_ids: np.ndarray | torch.Tensor) -> str:
    if isinstance(node_ids, torch.Tensor):
        node_ids = node_ids.detach().cpu().numpy()
    node_array = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    return hashlib.sha256(node_array.tobytes()).hexdigest()


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        tensor_like = cast(Any, value)
        return tensor_like.detach().cpu().numpy()
    return np.asarray(cast(object, value))


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
    return source_power_w * analog_ready_ns * 1.0e-9, service_ns


def _print_l1_manifest_mismatch(
    cached_manifest: dict, expected_manifest: dict, prefix: str
) -> None:
    for key in ("schema", "run_root", "runtime_shape", "read_v"):
        if cached_manifest.get(key) != expected_manifest.get(key):
            print(
                f"{prefix}: cache {key}={cached_manifest.get(key)!r}, "
                f"expected {expected_manifest.get(key)!r}"
            )

    cached_files = cached_manifest.get("files", {})
    expected_files = expected_manifest.get("files", {})
    if isinstance(cached_files, dict) and isinstance(expected_files, dict):
        for name in sorted(set(cached_files) | set(expected_files)):
            if cached_files.get(name) != expected_files.get(name):
                print(
                    f"{prefix}: cache file hash mismatch for {name}: "
                    f"cache={cached_files.get(name)}, expected={expected_files.get(name)}"
                )


class DynamoIGBHSUT:
    def __init__(
        self,
        run_root_path,
        test_mode="accuracy",
        sample_count=10000,
        validation_start=60000,
        rebuild_l1_cache=False,
        smoke_mode=False,
        subgraph_hops=2,
        transient_steps=10,
    ):
        self.run_root = Path(run_root_path).resolve()
        self.experiment_root = self.run_root / "dynamo_mid_experiment"
        self.test_mode = test_mode
        self.requested_sample_count = int(sample_count)
        self.validation_start = int(validation_start)
        self.rebuild_l1_cache = bool(rebuild_l1_cache)
        self.smoke_mode = bool(smoke_mode)
        self.subgraph_hops = int(subgraph_hops)
        self.transient_steps = int(transient_steps)
        if self.transient_steps < 1:
            raise ValueError(f"transient_steps must be >= 1, got {self.transient_steps}")
        if self.smoke_mode:
            print(
                "SUT: SMOKE MODE enabled. This is a small pipeline completion test, "
                "not a sign-off accuracy run."
            )

        print(
            "SUT: Initializing Suprascend NPU (Option 2: Custom Feature Extractor)..."
        )

        # 1. Load Hardware State
        self.runtime_state = load_integrated_runtime_state(self.run_root)
        self._calculate_physics_params()

        # 2. Load Native Model (All 3 layers on crossbar)
        self.model = DecomposedAnalogGCN(
            in_features=64,
            num_classes=19,
            hidden_dim=64,
            spec=self.runtime_state.static.spec,
            weight_mapping="conductance_sustained",
        )

        repo_root_path = Path(__file__).resolve().parent
        trained_model_dir = repo_root_path / "outputs" / "igbh_trained_model"

        # Load the 3 programmed omegas
        omega_vmm1_path = trained_model_dir / "omega_vmm1.pt"
        omega_mid_path = trained_model_dir / "omega_mid.pt"
        omega_vmm2_path = trained_model_dir / "omega_vmm2.pt"

        required_omegas = [omega_vmm1_path, omega_mid_path, omega_vmm2_path]
        missing_omegas = [str(p) for p in required_omegas if not p.exists()]
        if missing_omegas:
            raise FileNotFoundError(
                "Missing trained analog tile weights required for sign-off: "
                + ", ".join(missing_omegas)
            )

        print("SUT: Loading trained analog weights for all 3 tiles...")
        # PHYSICAL MAPPING: Clamp to [0.01, 0.99] to provide safety margin on physical rails
        omega_vmm1 = _load_weight_tensor(omega_vmm1_path).clamp(0.01, 0.99)
        omega_mid = _load_weight_tensor(omega_mid_path).clamp(0.01, 0.99)
        omega_vmm2 = _load_weight_tensor(omega_vmm2_path).clamp(0.01, 0.99)
        
        with torch.no_grad():
            self.model.vmm1.set_weights(omega_vmm1, None)
            self.model.vmm_mid.set_weights(omega_mid, None)
            self.model.vmm2.set_weights(omega_vmm2, None)

        self.model.eval()
        self.ops_per_sample = int(
            2
            * sum(
                np.prod(self._tile_conductance_matrix(tile).shape)
                for tile in [self.model.vmm1, self.model.vmm_mid, self.model.vmm2]
            )
        )

        # 4. MANUAL DATA LOAD (with PCA compression for 64-wordline hardware)
        print(f"SUT: Loading IGBH-Tiny features (1024-dim) from {IGBH_DATA_PATH}...")
        raw_node_features = np.load(
            os.path.join(IGBH_DATA_PATH, "paper/node_feat.npy"), mmap_mode="r"
        )

        print(
            "SUT: Utilizing synchronized PCA projection (1024 -> 64) for Suprascend NPU parity..."
        )
        self.node_features = raw_node_features
        print(f"SUT: Raw features loaded. Shape: {self.node_features.shape}")

        labels_path = Path(IGBH_DATA_PATH) / "paper" / "node_label_19.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"IGBH labels not found at {labels_path}; accuracy mode requires labels."
            )
        self.labels_path = labels_path
        self.node_labels = np.load(labels_path, mmap_mode="r")
        self.sample_to_node_idx = self._build_validation_qsl_mapping()
        self.num_samples = int(self.sample_to_node_idx.shape[0])
        self.progress_interval = max(1, min(100, self.num_samples // 10 or 1))
        self.validation_end = int(self.sample_to_node_idx[-1]) + 1
        print(
            "SUT: Mapped LoadGen to labeled validation paper nodes "
            f"[{self.validation_start}, {self.validation_end}) "
            f"({self.num_samples} samples)."
        )

        print("SUT: Loading IGBH-Tiny graph connectivity...")
        edge_np = np.load(
            os.path.join(IGBH_DATA_PATH, "paper__cites__paper/edge_index.npy"),
            mmap_mode="r",
        )
        self.edge_index = torch.from_numpy(edge_np[:].copy()).long().t().contiguous()

        target_nodes = torch.from_numpy(self.sample_to_node_idx.astype(np.int64)).long()
        (
            self.subgraph_nodes,
            self.subgraph_edge_index,
            target_local_nodes,
            _,
        ) = k_hop_subgraph(
            target_nodes,
            num_hops=self.subgraph_hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=int(self.node_features.shape[0]),
        )
        self.qsl_to_subgraph_local_idx = (
            target_local_nodes.cpu().numpy().astype(np.int64)
        )
        self.subgraph_node_count = int(self.subgraph_nodes.numel())
        self._write_qsl_mapping()
        print(
            "SUT: Extracted LoadGen validation subgraph "
            f"({self.subgraph_hops} hops, {self.subgraph_node_count:,} local nodes)."
        )

        # 5. PCA Load (Matches training Stage 0)
        pca_path = repo_root_path / "igbh_pca_components.npy"
        pca_mean_path = repo_root_path / "igbh_pca_mean.npy"
        if not pca_path.exists():
            raise FileNotFoundError(
                f"PCA matrix not found at {pca_path}. Run training first."
            )
        if not pca_mean_path.exists():
            raise FileNotFoundError(
                f"PCA mean not found at {pca_mean_path}. Run training first."
            )
        self.pca_matrix = torch.from_numpy(np.load(pca_path)).float()
        self.pca_mean = torch.from_numpy(np.load(pca_mean_path)).float()
        print(f"SUT: Loaded synchronized PCA matrix from {pca_path}")
        print(f"SUT: Loaded synchronized PCA mean from {pca_mean_path}")

        print(
            "SUT: Pre-computing normalized adjacency matrix on LoadGen validation subgraph..."
        )
        self.model.set_adjacency(
            self.subgraph_edge_index, self.subgraph_node_count, torch.device("cpu")
        )
        self.adj_hat = self.model.adj_hat

        # 6. PHYSICAL PRE-COMPUTE (Layer 1 + Aggregation)
        cache_tag = f"{self.validation_start}_{self.num_samples}_{self.subgraph_hops}hop"
        if self.smoke_mode:
            cache_path = (
                repo_root_path
                / "outputs"
                / f"igbh_l1_runtime_cache_loadgen_smoke_{cache_tag}.npz"
            )
        else:
            cache_path = (
                repo_root_path
                / "outputs"
                / f"igbh_l1_runtime_cache_loadgen_{cache_tag}.npz"
            )
        cache_manifest = self._build_l1_cache_manifest(
            omega_vmm1_path, pca_path, pca_mean_path
        )
        cache_manifest = {
            **cache_manifest,
            "cache_mode": "loadgen_validation_subgraph_smoke"
            if self.smoke_mode
            else "loadgen_validation_subgraph",
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "sample_count": self.num_samples,
            "subgraph_hops": self.subgraph_hops,
            "target_node_count": int(self.sample_to_node_idx.size),
            "subgraph_node_count": self.subgraph_node_count,
            "target_nodes_sha256": _node_ids_sha256(self.sample_to_node_idx),
            "subgraph_nodes_sha256": _node_ids_sha256(self.subgraph_nodes.cpu().numpy()),
        }

        cache_loaded = False if self.rebuild_l1_cache else self._load_l1_cache(cache_path, cache_manifest)
        if not cache_loaded:
            reason = "rebuild requested" if self.rebuild_l1_cache else "missing/stale"
            print(
                "SUT: Cache " + reason + ". Building LoadGen subgraph Layer 1 runtime cache."
            )
            solve_node_indices = np.arange(self.subgraph_node_count, dtype=np.int64)
            print(
                "SUT: Pre-calculating physical Layer 1 "
                f"(Full Kirchhoff Solve) for {solve_node_indices.size:,} subgraph nodes..."
            )

            with torch.no_grad():
                batch_size = 500  # Adjust based on RAM
                vmm1_cols = self._tile_conductance_matrix(self.model.vmm1).shape[1]
                x1_vmm = np.zeros((self.subgraph_node_count, vmm1_cols), dtype=np.float32)
                l1_energy_j = np.zeros(self.subgraph_node_count, dtype=np.float64)
                l1_latency_ns = np.zeros(self.subgraph_node_count, dtype=np.float64)

                from tqdm import tqdm

                for i in tqdm(
                    range(0, solve_node_indices.size, batch_size),
                    desc="Physical Solve (L1)",
                ):
                    batch_local_nodes = solve_node_indices[i : i + batch_size]
                    batch_global_nodes = (
                        self.subgraph_nodes[torch.from_numpy(batch_local_nodes)]
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                    )
                    batch_raw = np.asarray(self.node_features[batch_global_nodes]).copy()
                    batch_x = (
                        (torch.from_numpy(batch_raw).float() - self.pca_mean)
                        @ self.pca_matrix
                    ).numpy()
                    batch_out = _logical_tile_output(batch_x, self.model.vmm1)
                    trace = run_integrated_forward_workload(
                        state=self.runtime_state,
                        input_vectors=batch_x,
                        reference_vectors=batch_out,
                        lattix=self.model.vmm1.lattix,
                        workload_name="igbh_loadgen_l1_subgraph",
                        transient_steps=self.transient_steps,
                    )
                    batch_energy_j, batch_latency_ns = _event_energy_and_latency(trace)
                    x1_vmm[batch_local_nodes] = batch_out.astype(np.float32)
                    l1_energy_j[batch_local_nodes] = batch_energy_j
                    l1_latency_ns[batch_local_nodes] = batch_latency_ns

                print("SUT: Performing digital neighbor aggregation...")
                x1_aggregated = torch.sparse.mm(
                    self.adj_hat, torch.from_numpy(x1_vmm).float()
                )
                self.pre_aggregated_x1 = torch.relu(x1_aggregated).numpy()
                self.pre_l1_energy_j = l1_energy_j
                self.pre_l1_latency_ns = l1_latency_ns

                np.savez_compressed(
                    cache_path,
                    pre_aggregated_x1=self.pre_aggregated_x1,
                    l1_energy_j=self.pre_l1_energy_j,
                    l1_latency_ns=self.pre_l1_latency_ns,
                    manifest_json=np.array(json.dumps(cache_manifest, sort_keys=True)),
                )
                print(f"SUT: Cached runtime-validated L1 precompute to {cache_path}")

        self.total_energy_j = 0.0
        self.total_mid_output_material_energy_j = 0.0
        self.total_latency_ns = 0.0
        self.processed_count = 0

        self.sut = lg.ConstructSUT(self.issue_query, self.flush_queries)
        self.qsl = lg.ConstructQSL(
            self.num_samples, self.num_samples, self.load_samples, self.unload_samples
        )

    def _build_l1_cache_manifest(
        self, omega_vmm1_path: Path, pca_path: Path, pca_mean_path: Path
    ) -> dict:
        constants_dir = self.run_root / "runtime_constants"
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
            "run_root": str(self.run_root),
            "runtime_shape": list(self.runtime_state.static.shape),
            "read_v": float(self.runtime_state.static.read_v),
            "transient_steps": self.transient_steps,
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

    def _load_l1_cache(self, cache_path: Path, expected_manifest: dict) -> bool:
        if not cache_path.exists():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_manifest = json.loads(str(cached["manifest_json"].item()))
                if cached_manifest != expected_manifest:
                    print("SUT: Ignoring stale Layer 1 runtime cache manifest.")
                    _print_l1_manifest_mismatch(
                        cached_manifest, expected_manifest, "SUT"
                    )
                    return False
                self.pre_aggregated_x1 = cached["pre_aggregated_x1"]
                self.pre_l1_energy_j = cached["l1_energy_j"]
                self.pre_l1_latency_ns = cached["l1_latency_ns"]
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"SUT: Ignoring unreadable Layer 1 runtime cache: {exc}")
            return False

        expected_nodes = self.subgraph_node_count
        if self.pre_aggregated_x1.shape[0] != expected_nodes:
            print("SUT: Ignoring Layer 1 runtime cache with wrong node count.")
            return False
        print(f"SUT: Loaded runtime-validated Layer 1 cache from {cache_path}")
        return True

    def _build_validation_qsl_mapping(self) -> np.ndarray:
        num_nodes = int(self.node_features.shape[0])
        label_count = int(self.node_labels.shape[0])
        start = self.validation_start
        stop = min(start + self.requested_sample_count, num_nodes, label_count)
        if start < 0 or start >= stop:
            raise ValueError(
                f"Invalid validation range start={start}, stop={stop}, "
                f"nodes={num_nodes}, labels={label_count}"
            )

        candidate_nodes = np.arange(start, stop, dtype=np.int64)
        candidate_labels = np.asarray(
            self.node_labels[candidate_nodes], dtype=np.float64
        )
        labeled_mask = np.isfinite(candidate_labels) & (candidate_labels >= 0)
        labeled_nodes = candidate_nodes[labeled_mask]
        if labeled_nodes.size == 0:
            raise ValueError(
                f"No labeled validation nodes found in [{start}, {stop}) from {self.labels_path}"
            )
        if labeled_nodes.size < self.requested_sample_count:
            print(
                "SUT: Requested "
                f"{self.requested_sample_count} validation samples, but only "
                f"{labeled_nodes.size} labeled nodes were available in [{start}, {stop})."
            )
        return labeled_nodes

    def _write_qsl_mapping(self) -> None:
        mapping = {
            "schema": "dynamo.igbh.qsl_validation_mapping.v1",
            "split": "labeled_validation_paper_nodes",
            "mode": "smoke" if self.smoke_mode else "signoff",
            "labels_path": str(self.labels_path),
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "sample_count": self.num_samples,
            "node_indices": self.sample_to_node_idx.astype(int).tolist(),
            "subgraph_hops": self.subgraph_hops,
            "subgraph_node_count": self.subgraph_node_count,
            "target_local_node_indices": self.qsl_to_subgraph_local_idx.astype(int).tolist(),
        }
        with open("igbh_qsl_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)

    def _calculate_physics_params(self):
        static = self.runtime_state.static
        physical = self.runtime_state.physical

        self.physics_params = {
            "read_v": float(static.read_v),
            "read_source_r_ohm": float(static.read_source_r),
            "via_resistance_ohm": float(static.via_resistance_ohm),
            "read_timing_floor_ns": float(static.read_timing_floor_ns),
            "wordline_seg_r_matrix": np.asarray(
                static.wire_resistances_h, dtype=np.float64
            ),
            "bitline_seg_r_matrix": np.asarray(
                static.wire_resistances_v, dtype=np.float64
            ),
            "wordline_seg_r_ohm": float(static.wordline_seg_r),
            "bitline_seg_r_ohm": float(static.bitline_seg_r),
            "wordline_seg_c_f": float(static.wordline_seg_c),
            "bitline_seg_c_f": float(static.bitline_seg_c),
            "cell_node_c_f": float(static.cell_node_c),
            "wordline_node_caps_f": np.asarray(
                static.wordline_node_caps_f, dtype=np.float64
            ),
            "bitline_node_caps_f": np.asarray(
                static.bitline_node_caps_f, dtype=np.float64
            ),
            "ambient_temperature_k": float(physical.ambient_temperature_k),
            "temperature_map_k": np.asarray(
                physical.current_temperature_map, dtype=np.float64
            ),
            "cumulative_joule_heating_j": np.asarray(
                physical.cumulative_joule_heating_j, dtype=np.float64
            ),
            "cumulative_voltage_stress_v2s": np.asarray(
                physical.cumulative_voltage_stress_v2s, dtype=np.float64
            ),
            "readout_contract": static.cmos_service_timing_contract,
            "capacitance_provenance": static.capacitance_provenance,
            "spec": static.spec,
        }

        print(
            "SUT: Runtime physics loaded from integrated state "
            f"({self.run_root / 'runtime_constants'})."
        )

    def _tile_runtime_slices(self, rows: int, cols: int) -> dict[str, np.ndarray]:
        return {
            "wire_resistances_h": self.physics_params["wordline_seg_r_matrix"][
                :rows, : max(0, cols - 1)
            ],
            "wire_resistances_v": self.physics_params["bitline_seg_r_matrix"][
                : max(0, rows - 1), :cols
            ],
            "wordline_node_caps_f": self.physics_params["wordline_node_caps_f"][
                :rows, :cols
            ],
            "bitline_node_caps_f": self.physics_params["bitline_node_caps_f"][
                :rows, :cols
            ],
            "temperature_map_k": self.physics_params["temperature_map_k"][:rows, :cols],
            "cumulative_joule_heating_j": self.physics_params[
                "cumulative_joule_heating_j"
            ][:rows, :cols],
            "cumulative_voltage_stress_v2s": self.physics_params[
                "cumulative_voltage_stress_v2s"
            ][:rows, :cols],
        }

    def _tile_occupancy_fraction(self, tile, rows: int, cols: int) -> np.ndarray:
        return self._occupancy_fraction_for_lattix(tile.lattix, rows, cols)

    def _occupancy_fraction_for_lattix(self, lattix, rows: int, cols: int) -> np.ndarray:
        if hasattr(lattix, "n_high") and hasattr(lattix, "n_active"):
            n_high = _as_numpy(lattix.n_high).astype(np.float64).T
            n_active = _as_numpy(lattix.n_active).astype(np.float64).T
            occupancy = np.divide(n_high, np.maximum(n_active, 1.0))
            return occupancy[:rows, :cols]
        return self.runtime_state.physical.occupancy_fraction[:rows, :cols]

    def _tile_conductance_matrix(self, tile) -> np.ndarray:
        lattix = tile.lattix
        if hasattr(lattix, "positive") and hasattr(lattix, "negative"):
            conductance = _as_numpy(lattix.positive.g_total()).astype(np.float64)
        else:
            conductance = _as_numpy(lattix.g_total()).astype(np.float64)
        return conductance.T

    def _tile_ops_mac(self, tile) -> int:
        rows, cols = self._tile_conductance_matrix(tile).shape
        return int(rows * cols)

    def _tile_interface_energy_j(self, tile) -> float:
        rows, cols = self._tile_conductance_matrix(tile).shape
        cmos = self.runtime_state.static.cmos_interface
        c_interface_f = float(rows + cols) * float(cmos.c_gate_f)
        return 0.5 * c_interface_f * float(cmos.v_dd) ** 2 * float(cmos.logic_activity_factor)

    def load_samples(self, samples):
        pass

    def unload_samples(self, samples):
        pass

    def issue_query(self, query_samples):
        responses = []
        logits_keepalive = []

        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull):
                for sample in query_samples:
                    qsl_idx = sample.index
                    idx = int(self.qsl_to_subgraph_local_idx[qsl_idx])

                    x1 = torch.from_numpy(self.pre_aggregated_x1[idx : idx + 1]).float()
                    e1 = float(self.pre_l1_energy_j[idx])
                    l1 = float(self.pre_l1_latency_ns[idx])

                    x1_np = x1.numpy().astype(np.float64)
                    out_mid = _logical_tile_output(x1_np, self.model.vmm_mid)
                    mid_trace = run_integrated_forward_workload(
                        state=self.runtime_state,
                        input_vectors=x1_np,
                        reference_vectors=out_mid,
                        lattix=self.model.vmm_mid.lattix,
                        workload_name="igbh_loadgen_mid_sample",
                        transient_steps=self.transient_steps,
                    )
                    mid_energy_j, mid_latency_ns = _event_energy_and_latency(mid_trace)
                    x2_np = np.maximum(out_mid + x1_np, 0.0)
                    out2 = _logical_tile_output(x2_np, self.model.vmm2)
                    out_trace = run_integrated_forward_workload(
                        state=self.runtime_state,
                        input_vectors=x2_np,
                        reference_vectors=out2,
                        lattix=self.model.vmm2.lattix,
                        workload_name="igbh_loadgen_output_sample",
                        transient_steps=self.transient_steps,
                    )
                    out_energy_j, out_latency_ns = _event_energy_and_latency(out_trace)

                    sample_mid_output_energy_j = float(mid_energy_j[0] + out_energy_j[0])
                    self.total_mid_output_material_energy_j += sample_mid_output_energy_j
                    self.total_energy_j += e1 + sample_mid_output_energy_j
                    self.total_latency_ns += l1 + float(mid_latency_ns[0]) + float(out_latency_ns[0])
                    self.processed_count += 1

                    logits = out2.astype(np.float32)
                    logits_keepalive.append(logits)
                    responses.append(
                        lg.QuerySampleResponse(
                            sample.id, logits.ctypes.data, logits.nbytes
                        )
                    )

                    if (
                        self.processed_count == 1
                        or self.processed_count == self.num_samples
                        or self.processed_count % self.progress_interval == 0
                    ):
                        import sys as sys_lib

                        sys_lib.stderr.write(
                            f"SUT: Progress - {self.processed_count}/{self.num_samples} physicalized...\n"
                        )
                        sys_lib.stderr.flush()

            lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass

    def save_metrics(self):
        if self.processed_count == 0:
            print("SUT: No samples processed, skipping metrics save.")
            return

        avg_latency_ps = (self.total_latency_ns / self.processed_count) * 1000
        avg_latency_ns = avg_latency_ps / 1000.0
        physical_qps = 1.0 / (avg_latency_ps * 1.0e-12) if avg_latency_ps > 0 else 0.0
        physical_qps_billion = physical_qps / 1.0e9
        l1_shape = tuple(int(v) for v in self._tile_conductance_matrix(self.model.vmm1).shape)
        mid_shape = tuple(int(v) for v in self._tile_conductance_matrix(self.model.vmm_mid).shape)
        out_shape = tuple(int(v) for v in self._tile_conductance_matrix(self.model.vmm2).shape)
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
        target_path_ops = self.processed_count * ops_per_sample
        target_path_tops_w = (
            (target_path_ops / 1e12) / self.total_energy_j if self.total_energy_j > 0 else 0
        )
        target_path_fj_per_op = (self.total_energy_j * 1.0e15) / max(target_path_ops, 1)

        full_workload_ops_mac = float(
            self.subgraph_node_count * l1_ops_mac
            + self.processed_count * (mid_ops_mac + out_ops_mac)
        )
        full_workload_ops = 2.0 * full_workload_ops_mac
        subgraph_l1_material_energy_j = float(np.sum(self.pre_l1_energy_j))
        full_workload_material_energy_j = (
            subgraph_l1_material_energy_j + self.total_mid_output_material_energy_j
        )
        interface_cmos_energy_j = (
            self.subgraph_node_count * self._tile_interface_energy_j(self.model.vmm1)
            + self.processed_count
            * (
                self._tile_interface_energy_j(self.model.vmm_mid)
                + self._tile_interface_energy_j(self.model.vmm2)
            )
        )
        full_workload_npu_energy_j = full_workload_material_energy_j + interface_cmos_energy_j
        full_workload_npu_tops_w = (
            (full_workload_ops / 1.0e12) / full_workload_npu_energy_j
            if full_workload_npu_energy_j > 0
            else 0.0
        )
        full_workload_npu_fj_per_op = (
            full_workload_npu_energy_j * 1.0e15 / max(full_workload_ops, 1.0)
        )

        workload_metrics = {
            "samples": self.processed_count,
            "mode": "smoke" if self.smoke_mode else "signoff",
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_sample_count": self.num_samples,
            "target_path_material_energy_j": self.total_energy_j,
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
            "ops_per_sample": ops_per_sample,
            "operation_basis": op_basis,
            "subgraph_l1_material_energy_j": subgraph_l1_material_energy_j,
            "target_mid_output_material_energy_j": self.total_mid_output_material_energy_j,
            "energy_scope": "Dynamo NPU only; live benchmark material energy plus live-modeled NPU interface CMOS energy; excludes host CPU/GPU/platform power",
            "transient_steps": self.transient_steps,
            "total_energy_j": full_workload_npu_energy_j,
            "avg_latency_ps": avg_latency_ps,
            "avg_latency_ns": avg_latency_ns,
            "physical_qps": physical_qps,
            "qps_billion": physical_qps_billion,
            "tops_w": full_workload_npu_tops_w,
        }

        with open("igbh_workload_metrics.json", "w") as f:
            json.dump(workload_metrics, f, indent=2)
        print(
            "SUT: Workload metrics saved to igbh_workload_metrics.json "
            f"(QPS: {physical_qps_billion:.3f}B, full NPU TOPS/W: {full_workload_npu_tops_w:.2f})"
        )
        print(f"SUT: Finalized processing {self.processed_count} samples.")


def main():
    parser = argparse.ArgumentParser(
        description="IGBH MLPerf Harness for Suprascend NPU"
    )
    parser.add_argument(
        "--mode", type=str, choices=["accuracy", "performance"], default="accuracy"
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default="/home/lattix/Dynamo2SwappingVersions/Dynamo2/DistilledDynamo2/outputs/private_end_to_end_runs/integrated_runtime_audit_gf180_rebuilt_recalibrated_fresh_final2",
    )
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--validation_start", type=int, default=60000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small validation-window pipeline test instead of sign-off.",
    )
    parser.add_argument(
        "--smoke_nodes",
        type=int,
        default=16,
        help="Number of validation nodes to run in --smoke mode.",
    )
    parser.add_argument(
        "--rebuild_l1_cache",
        action="store_true",
        help="Allow rebuilding the expensive Layer 1 physical cache.",
    )
    parser.add_argument(
        "--subgraph_hops",
        type=int,
        default=2,
        help="K-hop validation subgraph radius used for LoadGen physical L1 precompute.",
    )
    parser.add_argument(
        "--transient_steps",
        type=int,
        default=10,
        help="Internal transient increments per runtime read event.",
    )
    args = parser.parse_args()
    sample_count = args.smoke_nodes if args.smoke else args.count

    sut = DynamoIGBHSUT(
        args.run_root,
        test_mode=args.mode,
        sample_count=sample_count,
        validation_start=args.validation_start,
        rebuild_l1_cache=args.rebuild_l1_cache,
        smoke_mode=args.smoke,
        subgraph_hops=args.subgraph_hops,
        transient_steps=args.transient_steps,
    )

    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Offline
    settings.min_duration_ms = 0
    settings.max_duration_ms = 0
    settings.min_query_count = 1
    settings.max_query_count = sut.num_samples

    if args.mode == "accuracy":
        settings.mode = lg.TestMode.AccuracyOnly
        settings.accuracy_log_sampling_target = sut.num_samples
        if args.smoke:
            print(
                "Running IGBH/RGAT smoke pipeline test "
                f"({sut.num_samples} validation samples) - ACCURACY MODE..."
            )
        else:
            print(
                "Running IGBH/RGAT Sign-off (Option 2: Native Extractor) - ACCURACY MODE..."
            )
    else:
        settings.mode = lg.TestMode.PerformanceOnly
        settings.offline_expected_qps = 1000
        settings.min_query_count = 1000
        settings.performance_sample_count_override = sut.num_samples
        settings.min_duration_ms = 60000
        settings.max_duration_ms = 60000
        print(
            "Running IGBH/RGAT Sign-off (Option 2: Native Extractor) - PERFORMANCE MODE (60s)..."
        )

    lg.StartTest(sut.sut, sut.qsl, settings)

    sut.save_metrics()
    lg.DestroySUT(sut.sut)
    lg.DestroyQSL(sut.qsl)

    if args.mode == "accuracy":
        if os.path.exists("mlperf_log_accuracy.json"):
            shutil.copy("mlperf_log_accuracy.json", "mlperf_log_accuracy_igbh.json")
            print(
                "Saved accuracy log to mlperf_log_accuracy_igbh.json to prevent overwrite."
            )


if __name__ == "__main__":
    main()
