from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class Topology:
    Nin: int
    K: int
    negref: int
    posref: int
    input_nodes: np.ndarray   # (Nin,)
    out_nodes: np.ndarray     # (K,)
    edges_D: np.ndarray       # (E,)
    edges_S: np.ndarray       # (E,)
    num_edges: int
    meta: Dict[str, Any]

def save_topology_npz(path: str | Path, topo: Topology):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        Nin=np.array(topo.Nin, dtype=int),
        K=np.array(topo.K, dtype=int),
        negref=np.array(topo.negref, dtype=int),
        posref=np.array(topo.posref, dtype=int),
        input_nodes=np.asarray(topo.input_nodes, dtype=int),
        out_nodes=np.asarray(topo.out_nodes, dtype=int),
        edges_D=np.asarray(topo.edges_D, dtype=int),
        edges_S=np.asarray(topo.edges_S, dtype=int),
        num_edges=np.array(topo.num_edges, dtype=int),
        meta=np.array(dict(topo.meta), dtype=object),
    )

def load_topology_npz(path: str | Path) -> Topology:
    z = np.load(str(path), allow_pickle=True)
    meta = {}
    if "meta" in z:
        try:
            meta = z["meta"].item()
        except Exception:
            meta = {}
    input_nodes = np.asarray(z["input_nodes"], dtype=int).ravel()
    out_nodes   = np.asarray(z["out_nodes"], dtype=int).ravel()
    edges_D     = np.asarray(z["edges_D"], dtype=int).ravel()
    edges_S     = np.asarray(z["edges_S"], dtype=int).ravel()

    Nin = int(z["Nin"]) if "Nin" in z else int(input_nodes.size)
    K   = int(z["K"])   if "K"   in z else int(out_nodes.size)
    negref = int(z["negref"])
    posref = int(z["posref"])
    num_edges = int(z["num_edges"]) if "num_edges" in z else int(edges_D.size)

    return Topology(
        Nin=Nin, K=K, negref=negref, posref=posref,
        input_nodes=input_nodes, out_nodes=out_nodes,
        edges_D=edges_D, edges_S=edges_S, num_edges=num_edges,
        meta=meta,
    )