"""Topology-conditioned adapter for cross-network PhaseVLM adaptation.

Provides IntersectionTopologyEncoder: a lightweight MLP that maps static
intersection features (grid position, neighbor existence, network scale) to
a topology embedding that conditions the prefix projection.

Design rationale (per README/stage3_cross_network_adaptation_20260810.md §4):
  - For 12-16 node grids, a full GNN is overkill — the prefix tokens already
    capture dynamic traffic state via the TS2I image.
  - Static topology features injected as additive conditioning on the pooled
    vision features before prefix projection.
  - The encoder handles arbitrary grid sizes via normalized position features
    and network-scale metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---- Road network topology registry ----
# (rows, cols): number of controlled intersections in each dimension
# neighbor_ENWS[r][c] = [has_east, has_north, has_west, has_south] (4 bools)
ROADNET_TOPOLOGY: Dict[str, Dict] = {
    "Hangzhou-4_4": {
        "rows": 4,
        "cols": 4,
        # Hangzhou 4x4: full grid, all interior intersections have all 4 neighbors
        # Border intersections: missing exterior-facing neighbors
        "neighbors": {},  # auto-computed: interior full, border partial
    },
    "Jinan-3_4": {
        "rows": 3,
        "cols": 4,
        # Jinan 4cols × 3rows: border intersections miss some neighbors
        "neighbors": {},
    },
}


def _auto_neighbors(rows: int, cols: int) -> Dict[str, List[bool]]:
    """Compute neighbor_ENWS existence for a rows×cols grid.

    E,N,W,S order, True if a neighbor intersection exists in that direction.
    """
    result: Dict[str, List[bool]] = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            iname = f"intersection_{r}_{c}"
            result[iname] = [
                c < cols,   # has East  (col+1 exists)
                r < rows,   # has North (row+1 exists)
                c > 1,      # has West  (col-1 exists)
                r > 1,      # has South (row-1 exists)
            ]
    return result


# Pre-compute neighbor tables
for _name, _cfg in ROADNET_TOPOLOGY.items():
    _cfg["neighbors"] = _auto_neighbors(_cfg["rows"], _cfg["cols"])


class IntersectionTopologyEncoder(nn.Module):
    """MLP that maps intersection topology features to a conditioning embedding.

    Features per intersection (8-dim):
      - row / rows          (normalized row position)
      - col / cols          (normalized col position)
      - has_E, has_N, has_W, has_S  (binary neighbor existence)
      - rows, cols          (network scale, log-normalized)

    Output: topology_embedding [vision_dim] for additive conditioning.
    """

    def __init__(
        self,
        vision_dim: int,
        hidden_dim: int = 32,
        roadnet_configs: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.hidden_dim = int(hidden_dim)

        # Use provided configs or defaults
        self.roadnet_configs: Dict[str, Dict] = {}
        if roadnet_configs:
            for name, cfg in roadnet_configs.items():
                c = dict(cfg)
                if "neighbors" not in c:
                    c["neighbors"] = _auto_neighbors(
                        int(c.get("rows", 1)), int(c.get("cols", 1))
                    )
                self.roadnet_configs[name] = c
        else:
            self.roadnet_configs = {
                name: dict(cfg) for name, cfg in ROADNET_TOPOLOGY.items()
            }

        # Build embedding table: intersection_id → feature vector
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_features: List[List[float]] = []
        self._dataset_of_intersection: Dict[str, str] = {}
        self._build_feature_table()

        num_intersections = len(self._id_to_idx)
        if num_intersections == 0:
            raise ValueError("Topology encoder requires at least one roadnet config")

        self.feature_dim = len(self._idx_to_features[0]) if self._idx_to_features else 8
        # Use a small Embedding as lookup: intersection → learnable representation
        self.intersection_embed = nn.Embedding(num_intersections, self.feature_dim)
        # Initialize with static features
        with torch.no_grad():
            feat_tensor = torch.tensor(self._idx_to_features, dtype=torch.float32)
            self.intersection_embed.weight.copy_(feat_tensor)

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.vision_dim),
        )

    def _build_feature_table(self) -> None:
        """Build static feature vectors for all intersections across all networks.

        Uses qualified keys "dataset_name:intersection_id" to disambiguate
        intersections that share the same grid coordinates across networks.
        """
        import math

        idx = 0
        for dataset_name, cfg in sorted(self.roadnet_configs.items()):
            rows = int(cfg.get("rows", 1))
            cols = int(cfg.get("cols", 1))
            neighbors = cfg.get("neighbors", {})
            log_rows = math.log(max(1, rows))
            log_cols = math.log(max(1, cols))

            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    iname = f"intersection_{r}_{c}"
                    qualified = f"{dataset_name}:{iname}"
                    if qualified in self._id_to_idx:
                        continue
                    nbr = neighbors.get(iname, [False, False, False, False])
                    feat = [
                        float(r) / max(1, rows),     # normalized row
                        float(c) / max(1, cols),     # normalized col
                        1.0 if nbr[0] else 0.0,      # has East
                        1.0 if nbr[1] else 0.0,      # has North
                        1.0 if nbr[2] else 0.0,      # has West
                        1.0 if nbr[3] else 0.0,      # has South
                        log_rows,                      # network scale (log)
                        log_cols,                      # network scale (log)
                    ]
                    self._id_to_idx[qualified] = idx
                    self._idx_to_features.append(feat)
                    self._dataset_of_intersection[qualified] = dataset_name
                    idx += 1

    def _resolve_qualified_id(
        self,
        intersection_id: str,
        dataset_hint: str = "",
    ) -> str:
        """Resolve an intersection_id + optional dataset hint to a qualified key."""
        iid = str(intersection_id)
        # Try qualified lookup first
        if dataset_hint:
            qualified = f"{dataset_hint}:{iid}"
            if qualified in self._id_to_idx:
                return qualified
        # Try all known networks
        for ds_name in self.roadnet_configs:
            qualified = f"{ds_name}:{iid}"
            if qualified in self._id_to_idx:
                return qualified
        # Fallback: bare ID (backward compat)
        if iid in self._id_to_idx:
            return iid
        return ""

    def forward(
        self,
        intersection_ids: List[str],
        dataset_hints: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Map intersection IDs to topology embeddings.

        Args:
            intersection_ids: list of strings like ["intersection_1_1", ...]
            dataset_hints: optional per-sample dataset names for disambiguation
                           (e.g. ["Hangzhou-4_4", ...])

        Returns:
            topology_embedding [B, vision_dim]
        """
        device = self.intersection_embed.weight.device
        indices: List[int] = []
        for i, iid in enumerate(intersection_ids):
            hint = dataset_hints[i] if dataset_hints and i < len(dataset_hints) else ""
            qualified = self._resolve_qualified_id(str(iid), str(hint or ""))
            if qualified:
                indices.append(self._id_to_idx[qualified])
            else:
                # Unknown intersection — use index 0 as fallback
                indices.append(0)

        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        static_feats = self.intersection_embed(idx_tensor)  # [B, feature_dim]
        return self.mlp(static_feats)  # [B, vision_dim]

    def get_dataset_name(self, intersection_id: str) -> str:
        """Return the dataset/network name for an intersection."""
        return self._dataset_of_intersection.get(str(intersection_id), "unknown")

    @staticmethod
    def dataset_hint_from_traffic_file(traffic_file: str) -> str:
        """Map a traffic_file name to a dataset hint for the topology encoder.

        Examples:
            anon_4_4_hangzhou_real → Hangzhou-4_4
            anon_3_4_jinan_real → Jinan-3_4
            anon_4_4_hangzhou_real_5816 → Hangzhou-4_4
        """
        tf = str(traffic_file).lower()
        if "hangzhou" in tf:
            return "Hangzhou-4_4"
        if "jinan" in tf:
            return "Jinan-3_4"
        if "newyork" in tf:
            return "NewYork-28_7"
        return ""

    @classmethod
    def dataset_hints_from_batch(
        cls,
        traffic_files: List[str],
        intersection_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Convert a batch of traffic_file values to dataset hints."""
        return [cls.dataset_hint_from_traffic_file(tf) for tf in traffic_files]

    def extra_repr(self) -> str:
        return (
            f"vision_dim={self.vision_dim}, hidden_dim={self.hidden_dim}, "
            f"num_intersections={len(self._id_to_idx)}, "
            f"networks={list(self.roadnet_configs)}"
        )
