# topology_io.py
# One combined module: topology dataclass + NPZ save/load + builder functions.
# No "crossbar" terminology anywhere.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import numpy as np


# -------------------------
# Core topology container
# -------------------------

@dataclass(frozen=True)
class Topology:
    """
    Directed input->output wiring description suitable for netlist generation.

    Nodes are numeric (ints) so they work cleanly with ngspice "print allv".
    Parallel edges are represented by duplicate (D,S) pairs in edges_D/edges_S.
    """
    negref: int
    posref: int
    input_nodes: np.ndarray  # shape (Nin,)
    out_nodes: np.ndarray    # shape (K,)
    edges_D: np.ndarray      # shape (E,)
    edges_S: np.ndarray      # shape (E,)
    meta: Dict[str, Any]

    @property
    def Nin(self) -> int:
        return int(self.input_nodes.size)

    @property
    def K(self) -> int:
        return int(self.out_nodes.size)

    @property
    def num_edges(self) -> int:
        return int(self.edges_D.size)


# -------------------------
# I/O (NPZ)
# -------------------------

def save_topology_npz(path: Union[str, Path], topo: Topology) -> None:
    """
    Saves a topology artifact as a single .npz file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        negref=np.int64(topo.negref),
        posref=np.int64(topo.posref),
        input_nodes=np.asarray(topo.input_nodes, dtype=np.int64),
        out_nodes=np.asarray(topo.out_nodes, dtype=np.int64),
        edges_D=np.asarray(topo.edges_D, dtype=np.int64),
        edges_S=np.asarray(topo.edges_S, dtype=np.int64),
        meta_json=np.array([json.dumps(topo.meta)], dtype=object),
    )


def load_topology_npz(path: Union[str, Path]) -> Topology:
    """
    Loads a topology artifact saved by save_topology_npz().
    """
    path = Path(path)
    z = np.load(path, allow_pickle=True)

    meta: Dict[str, Any] = {}
    if "meta_json" in z:
        try:
            meta = json.loads(str(z["meta_json"][0]))
        except Exception:
            meta = {}

    topo = Topology(
        negref=int(z["negref"]),
        posref=int(z["posref"]),
        input_nodes=np.asarray(z["input_nodes"], dtype=int),
        out_nodes=np.asarray(z["out_nodes"], dtype=int),
        edges_D=np.asarray(z["edges_D"], dtype=int),
        edges_S=np.asarray(z["edges_S"], dtype=int),
        meta=meta,
    )
    validate_topology(topo)
    return topo


def validate_topology(topo: Topology) -> None:
    """
    Basic structural sanity checks. Allows parallel edges.
    """
    if topo.input_nodes.ndim != 1 or topo.out_nodes.ndim != 1:
        raise ValueError("input_nodes and out_nodes must be 1D arrays")

    if topo.edges_D.ndim != 1 or topo.edges_S.ndim != 1:
        raise ValueError("edges_D and edges_S must be 1D arrays")

    if topo.edges_D.shape != topo.edges_S.shape:
        raise ValueError("edges_D and edges_S must have the same shape")

    if topo.input_nodes.size == 0:
        raise ValueError("input_nodes is empty")
    if topo.out_nodes.size == 0:
        raise ValueError("out_nodes is empty")

    # Check that every edge goes from an input node to an output node
    in_set = set(topo.input_nodes.tolist())
    out_set = set(topo.out_nodes.tolist())

    # (O(E)) membership checks; fine for typical sizes
    for d, s in zip(topo.edges_D.tolist(), topo.edges_S.tolist()):
        if d not in in_set:
            raise ValueError(f"Edge drain node {d} not present in input_nodes")
        if s not in out_set:
            raise ValueError(f"Edge source node {s} not present in out_nodes")


# -------------------------
# Builders
# -------------------------

def _allocate_nodes(Nin: int, K: int, negref: int = 1, posref: int = 2) -> Tuple[int, int, np.ndarray, np.ndarray]:
    if Nin <= 0:
        raise ValueError("Nin must be > 0")
    if K <= 0:
        raise ValueError("K must be > 0")

    out_nodes = np.arange(3, 3 + K, dtype=int)
    input_nodes = np.arange(out_nodes[-1] + 1, out_nodes[-1] + 1 + Nin, dtype=int)
    return int(negref), int(posref), input_nodes, out_nodes


def build_fully_connected_input_output(
    Nin: int,
    K: int = 10,
    copies_per_pair: int = 3,
    negref: int = 1,
    posref: int = 2,
    meta: Optional[Dict[str, Any]] = None,
) -> Topology:
    """
    Builder type (1):
      Fully connected input->output, but each (input i, output k) appears copies_per_pair times.
      Parallel edges are represented by duplicate (D,S) pairs.

    Edge ordering is deterministic:
      uid = ((i*K) + k)*copies_per_pair + c
      where c in [0, copies_per_pair-1]
    """
    if copies_per_pair <= 0:
        raise ValueError("copies_per_pair must be > 0")

    negref, posref, input_nodes, out_nodes = _allocate_nodes(Nin=Nin, K=K, negref=negref, posref=posref)

    # For each input node, we list outputs with repetition:
    # [o0,o0,o0,o1,o1,o1,...] if copies_per_pair=3
    out_repeated = np.repeat(out_nodes, copies_per_pair)              # shape (K*copies,)
    edges_S = np.tile(out_repeated, Nin)                              # shape (Nin*K*copies,)
    edges_D = np.repeat(input_nodes, K * copies_per_pair)             # shape (Nin*K*copies,)

    m = {
        "type": "fully_connected_input_output",
        "Nin": int(Nin),
        "K": int(K),
        "copies_per_pair": int(copies_per_pair),
        "edges": int(edges_D.size),
        "ordering": "uid=((i*K)+k)*copies + c",
    }
    if meta:
        m.update(meta)

    topo = Topology(
        negref=negref,
        posref=posref,
        input_nodes=input_nodes,
        out_nodes=out_nodes,
        edges_D=edges_D,
        edges_S=edges_S,
        meta=m,
    )
    validate_topology(topo)
    return topo


# ---- Pruning helpers ----

# You can pass pruned_edges in a few convenient forms:
#   A) (i, k)           -> remove ALL copies for that (input, output)
#   B) (i, k, c)        -> remove ONLY copy c (0-based)
#   C) orig_index int   -> treat as orig = i*K + k; remove ALL copies for that pair
#
# This is intentionally flexible because different scripts store prune lists differently.

PruneSpec = Union[int, Tuple[int, int], Tuple[int, int, int]]


def _uids_for_prune_spec(spec: PruneSpec, K: int, copies: int) -> List[int]:
    if isinstance(spec, (int, np.integer)):
        orig = int(spec)
        if orig < 0:
            raise ValueError(f"pruned edge orig index must be >=0, got {orig}")
        base = orig * copies
        return list(range(base, base + copies))

    if isinstance(spec, tuple) and len(spec) == 2:
        i, k = int(spec[0]), int(spec[1])
        if i < 0 or k < 0:
            raise ValueError(f"(i,k) must be >=0, got {(i,k)}")
        base = ((i * K) + k) * copies
        return list(range(base, base + copies))

    if isinstance(spec, tuple) and len(spec) == 3:
        i, k, c = int(spec[0]), int(spec[1]), int(spec[2])
        if i < 0 or k < 0 or c < 0:
            raise ValueError(f"(i,k,c) must be >=0, got {(i,k,c)}")
        if c >= copies:
            raise ValueError(f"copy index c={c} out of range for copies={copies}")
        uid = ((i * K) + k) * copies + c
        return [uid]

    raise TypeError(f"Unsupported prune spec: {spec!r}")


def build_pruned_input_output(
    Nin: int,
    K: int = 10,
    copies_per_pair: int = 3,
    pruned_edges: Sequence[PruneSpec] = (),
    negref: int = 1,
    posref: int = 2,
    meta: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Topology:
    """
    Builder type (2):
      Start from the fully-connected input->output topology (with copies_per_pair parallel edges),
      then delete specified edges.

    pruned_edges can contain:
      - (i, k): delete ALL copies between input i and output k
      - (i, k, c): delete ONLY copy c (0-based) between input i and output k
      - orig int: treat orig = i*K + k; delete ALL copies for that pair

    strict=True:
      - raises if a prune spec addresses an out-of-range input/output index.
    strict=False:
      - ignores out-of-range prune specs.
    """
    base = build_fully_connected_input_output(
        Nin=Nin, K=K, copies_per_pair=copies_per_pair, negref=negref, posref=posref, meta=None
    )

    E = base.num_edges
    keep = np.ones(E, dtype=bool)

    # Validate bounds if strict
    def _check_bounds_i_k(i: int, k: int) -> None:
        if not (0 <= i < Nin) or not (0 <= k < K):
            raise ValueError(f"prune (i,k)=({i},{k}) out of range: Nin={Nin}, K={K}")

    for spec in pruned_edges:
        if isinstance(spec, (int, np.integer)):
            orig = int(spec)
            i = orig // K
            k = orig % K
            if strict:
                _check_bounds_i_k(i, k)
            else:
                if not (0 <= i < Nin and 0 <= k < K):
                    continue
        elif isinstance(spec, tuple) and (len(spec) == 2 or len(spec) == 3):
            i = int(spec[0])
            k = int(spec[1])
            if strict:
                _check_bounds_i_k(i, k)
            else:
                if not (0 <= i < Nin and 0 <= k < K):
                    continue
        else:
            raise TypeError(f"Unsupported prune spec: {spec!r}")

        uids = _uids_for_prune_spec(spec, K=K, copies=copies_per_pair)
        for uid in uids:
            if 0 <= uid < E:
                keep[uid] = False

    edges_D = base.edges_D[keep]
    edges_S = base.edges_S[keep]

    m = dict(base.meta)
    m.update(
        {
            "type": "pruned_input_output",
            "pruned_count": int(np.sum(~keep)),
            "kept_edges": int(edges_D.size),
        }
    )
    if meta:
        m.update(meta)

    topo = Topology(
        negref=base.negref,
        posref=base.posref,
        input_nodes=base.input_nodes,
        out_nodes=base.out_nodes,
        edges_D=edges_D,
        edges_S=edges_S,
        meta=m,
    )
    validate_topology(topo)
    return topo


# -------------------------
# Convenience: quick summary string
# -------------------------

def topology_summary(topo: Topology) -> str:
    return (
        f"Nin={topo.Nin} K={topo.K} edges={topo.num_edges} "
        f"negref={topo.negref} posref={topo.posref} meta_keys={sorted(list(topo.meta.keys()))}"
    )
