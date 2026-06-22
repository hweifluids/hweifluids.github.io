#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Wedge
from numpy.polynomial.legendre import Legendre
from PIL import Image, ImageDraw, ImageFont

try:
    import pyvista as pv
    import vtk
except Exception:  # pragma: no cover - reported at runtime.
    pv = None
    vtk = None


THRESHOLD = math.pi
LOG_EDGES = np.linspace(-5.0, 5.0, 801)
DEFAULT_RE_R = 1370.0
DEFAULT_CHUNK_ELEMENTS = 384
DEFAULT_VOXEL_GRID = (360, 128, 96)
WINDOW_SIZE = (1800, 1200)
GROUND_PLANE_COLOR = "#bcbcbc"
GROUND_PLANE_OPACITY = 0.24
FONT_REG = Path("C:/Windows/Fonts/times.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")
FONT_ITALIC = Path("C:/Windows/Fonts/timesi.ttf")
MIN_CONNECTED_COMPONENT_CELLS = 20
CLEAN_ABSOLUTE_TOLERANCE = 1.0e-7

LEVELS = (
    ("half_threshold", 0.5 * THRESHOLD, "0.5 pi", "#2c7fb8", 0.18),
    ("threshold", THRESHOLD, "pi", "#fdae61", 0.32),
    ("double_threshold", 2.0 * THRESHOLD, "2 pi", "#d7191c", 0.54),
    ("quad_threshold", 4.0 * THRESHOLD, "4 pi", "#542788", 0.74),
)

FIELD_ARRAYS_FLOAT32 = (
    "x",
    "y",
    "z",
    "epsilon",
    "eta",
    "delta_i",
    "delta_j",
    "delta_k",
    "delta_geom",
    "delta_max",
    "ratio_i",
    "ratio_j",
    "ratio_k",
    "ratio_geom",
    "ratio_max",
    "volume",
)


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class TeeLogging:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.handle = None
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = TeeStream(sys.stdout, self.handle)
        sys.stderr = TeeStream(sys.stderr, self.handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            traceback.print_exception(exc_type, exc, tb)
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()


@dataclass(frozen=True)
class FieldMeta:
    path: Path
    wdsiz: int
    nx: int
    ny: int
    nz: int
    nel: int
    nxyz: int
    time: float
    code: str
    endian: str
    base: int
    labels: tuple[str, ...]
    data_end: int


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Self-contained refined Groetzbach validation for one Nek field. "
            "The script computes epsilon, Kolmogorov eta, GLL-subcell spacings, "
            "full check fields, statistics, and full-domain 3D isosurfaces."
        )
    )
    parser.add_argument("--field", type=Path, required=True, help="Single Nek field file containing U, and preferably X.")
    parser.add_argument("--coord-field", type=Path, default=None, help="Optional coordinate field containing X.")
    parser.add_argument("--sequence-header", type=Path, default=None, help="Optional .nek5000 header used to locate the sequence head mesh.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for all outputs.")
    parser.add_argument("--re-r", type=float, default=DEFAULT_RE_R, help="Reynolds number based on hemisphere radius; nu=1/Re_R when --nu is omitted.")
    parser.add_argument("--nu", type=float, default=None, help="Kinematic viscosity. Overrides --re-r.")
    parser.add_argument("--chunk-elements", type=int, default=DEFAULT_CHUNK_ELEMENTS, help="Elements processed per chunk.")
    parser.add_argument("--voxel-grid", type=str, default="x".join(str(v) for v in DEFAULT_VOXEL_GRID), help="Voxel grid for 3D contours as nx,ny,nz or nxXnyXnz.")
    parser.add_argument("--top-n", type=int, default=300, help="Rows retained in worst-subcell CSV.")
    parser.add_argument("--max-elements", type=int, default=None, help="Debug option: process only the first N elements.")
    parser.add_argument("--skip-3d", action="store_true", help="Skip PyVista 3D isosurface visualization.")
    parser.add_argument("--skip-2d", action="store_true", help="Skip matplotlib 2D summary plots.")
    return parser.parse_args()


def parse_voxel_grid(text: str) -> tuple[int, int, int]:
    parts = [p for p in re.split(r"[xX, ]+", text.strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"--voxel-grid must contain three integers, got {text!r}")
    dims = tuple(int(p) for p in parts)
    if min(dims) < 4:
        raise ValueError(f"--voxel-grid dimensions are too small: {dims}")
    return dims


def labels_from_code(code: str) -> tuple[str, ...]:
    labels: list[str] = []
    if "X" in code:
        labels.extend(("x", "y", "z"))
    if "U" in code:
        labels.extend(("u", "v", "w"))
    if "P" in code:
        labels.append("p")
    if "T" in code:
        labels.append("t")
    sm = re.search(r"S(\d\d)", code)
    if sm:
        labels.extend(f"s{i + 1:02d}" for i in range(int(sm.group(1))))
    return tuple(labels)


def read_header(path: Path) -> FieldMeta:
    path = path.resolve()
    with path.open("rb") as fh:
        header = fh.read(132).decode("ascii", errors="replace")
        tag = fh.read(4)
    parts = header.split()
    if len(parts) < 12 or parts[0] != "#std":
        raise ValueError(f"not a Nek #std field file: {path}")
    wdsiz = int(parts[1])
    nx = int(parts[2])
    ny = int(parts[3])
    nz = int(parts[4])
    nel = int(parts[5])
    time_value = float(parts[7].replace("D", "E"))
    code = parts[11]
    endian = "<" if abs(struct.unpack("<f", tag)[0] - 6.54321) < 1.0e-4 else ">"
    labels = labels_from_code(code)
    nxyz = nx * ny * nz
    base = 132 + 4 + 4 * nel
    data_end = base + len(labels) * nel * nxyz * wdsiz
    actual = path.stat().st_size
    if actual < data_end:
        raise ValueError(f"{path.name} is shorter than its header-declared payload: actual={actual}, data_end={data_end}")
    if wdsiz not in (4, 8):
        raise ValueError(f"unsupported Nek word size {wdsiz} in {path}")
    return FieldMeta(path, wdsiz, nx, ny, nz, nel, nxyz, time_value, code, endian, base, labels, data_end)


def dtype_for(meta: FieldMeta) -> np.dtype:
    return np.dtype(meta.endian + ("f4" if meta.wdsiz == 4 else "f8"))


def component_offset(meta: FieldMeta, label: str) -> int:
    if label not in meta.labels:
        raise ValueError(f"{meta.path.name} has no component {label!r}; code={meta.code}")
    idx = meta.labels.index(label)
    return meta.base + idx * meta.nel * meta.nxyz * meta.wdsiz


def vector_offset(meta: FieldMeta, block: str) -> int:
    if block == "X":
        if "X" not in meta.code:
            raise ValueError(f"{meta.path.name} has no coordinate block; code={meta.code}")
        return meta.base
    if block == "U":
        if "U" not in meta.code:
            raise ValueError(f"{meta.path.name} has no velocity block; code={meta.code}")
        before = 3 if "X" in meta.code else 0
        return meta.base + before * meta.nel * meta.nxyz * meta.wdsiz
    raise ValueError(block)


def numeric_suffix(path: Path) -> int:
    m = re.search(r"\.f(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def parse_sequence_header(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        data[key.strip().lower()] = value.strip()
    return data


def expand_nek_template(template: str, first_timestep: int) -> list[str]:
    names: list[str] = []
    for args in ((0, first_timestep), (first_timestep,), (first_timestep, 0)):
        try:
            names.append(template % args)
        except Exception:
            pass
    return list(dict.fromkeys(names))


def same_layout(a: FieldMeta, b: FieldMeta) -> bool:
    return (a.nx, a.ny, a.nz, a.nel, a.nxyz) == (b.nx, b.ny, b.nz, b.nel, b.nxyz)


def valid_coord_candidate(path: Path, data_meta: FieldMeta) -> bool:
    try:
        meta = read_header(path)
    except Exception:
        return False
    return "X" in meta.code and same_layout(meta, data_meta)


def coord_candidates_from_headers(field: Path, data_meta: FieldMeta, sequence_header: Path | None) -> list[Path]:
    headers: list[Path] = []
    if sequence_header is not None:
        headers.append(sequence_header.resolve())
    headers.extend(sorted(field.parent.glob("*.nek5000")))
    seen: set[Path] = set()
    out: list[Path] = []
    for header in headers:
        if header in seen or not header.exists():
            continue
        seen.add(header)
        try:
            info = parse_sequence_header(header)
        except Exception:
            continue
        template = info.get("filetemplate")
        if not template:
            continue
        first = int(info.get("firsttimestep", "0"))
        for name in expand_nek_template(template, first):
            candidate = (header.parent / name).resolve()
            if candidate.exists():
                out.append(candidate)
    return list(dict.fromkeys(out))


def resolve_coord_field(field: Path, data_meta: FieldMeta, coord_arg: Path | None, sequence_header: Path | None) -> Path:
    if coord_arg is not None:
        coord = coord_arg.resolve()
        coord_meta = read_header(coord)
        if "X" not in coord_meta.code:
            raise ValueError(f"--coord-field has no coordinate block: {coord}")
        if not same_layout(coord_meta, data_meta):
            raise ValueError(f"--coord-field layout does not match --field: {coord}")
        return coord
    if "X" in data_meta.code:
        return field.resolve()

    candidates: list[Path] = []
    m = re.search(r"^(.*)\.f\d+$", field.name)
    if m:
        candidates.append((field.parent / f"{m.group(1)}.f00000").resolve())
    candidates.extend(coord_candidates_from_headers(field, data_meta, sequence_header))
    candidates.extend(sorted(field.parent.glob("*.f*"), key=lambda p: (numeric_suffix(p), p.name)))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists() or not candidate.is_file():
            continue
        seen.add(candidate)
        if valid_coord_candidate(candidate, data_meta):
            return candidate.resolve()
    raise FileNotFoundError(
        f"{field.name} does not contain X and no matching coordinate field was found in the sequence head or directory."
    )


def gll_nodes(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("GLL node count must be >= 2")
    if n == 2:
        return np.array([-1.0, 1.0], dtype=np.float64)
    roots = Legendre.basis(n - 1).deriv().roots()
    return np.r_[-1.0, np.sort(roots), 1.0].astype(np.float64)


def derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    n = nodes.size
    weights = np.ones(n, dtype=np.float64)
    for j in range(n):
        for k in range(n):
            if j != k:
                weights[j] /= nodes[j] - nodes[k]
    dmat = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            value = weights[j] / (weights[i] * (nodes[i] - nodes[j]))
            dmat[i, j] = value
            row_sum += value
        dmat[i, i] = -row_sum
    return dmat


def ref_derivatives(values: np.ndarray, di: np.ndarray, dj: np.ndarray, dk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # values shape is (element, k, j, i). Local i is the fastest Nek field index.
    dr = np.einsum("im,ekjm->ekji", di, values, optimize=True)
    ds = np.einsum("jm,ekmi->ekji", dj, values, optimize=True)
    dt = np.einsum("km,emji->ekji", dk, values, optimize=True)
    return dr, ds, dt


def average_vertices(a: np.ndarray) -> np.ndarray:
    return (
        a[:, :-1, :-1, :-1]
        + a[:, :-1, :-1, 1:]
        + a[:, :-1, 1:, :-1]
        + a[:, :-1, 1:, 1:]
        + a[:, 1:, :-1, :-1]
        + a[:, 1:, :-1, 1:]
        + a[:, 1:, 1:, :-1]
        + a[:, 1:, 1:, 1:]
    ) * 0.125


def edge_i(a: np.ndarray) -> np.ndarray:
    return 0.25 * (
        a[:, :-1, :-1, 1:] - a[:, :-1, :-1, :-1]
        + a[:, :-1, 1:, 1:] - a[:, :-1, 1:, :-1]
        + a[:, 1:, :-1, 1:] - a[:, 1:, :-1, :-1]
        + a[:, 1:, 1:, 1:] - a[:, 1:, 1:, :-1]
    )


def edge_j(a: np.ndarray) -> np.ndarray:
    return 0.25 * (
        a[:, :-1, 1:, :-1] - a[:, :-1, :-1, :-1]
        + a[:, :-1, 1:, 1:] - a[:, :-1, :-1, 1:]
        + a[:, 1:, 1:, :-1] - a[:, 1:, :-1, :-1]
        + a[:, 1:, 1:, 1:] - a[:, 1:, :-1, 1:]
    )


def edge_k(a: np.ndarray) -> np.ndarray:
    return 0.25 * (
        a[:, 1:, :-1, :-1] - a[:, :-1, :-1, :-1]
        + a[:, 1:, :-1, 1:] - a[:, :-1, :-1, 1:]
        + a[:, 1:, 1:, :-1] - a[:, :-1, 1:, :-1]
        + a[:, 1:, 1:, 1:] - a[:, :-1, 1:, 1:]
    )


def compute_epsilon_and_subcells(
    xyz_flat: np.ndarray,
    vel_flat: np.ndarray,
    d_i: np.ndarray,
    d_j: np.ndarray,
    d_k: np.ndarray,
    dims: tuple[int, int, int],
    nu: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    nx, ny, nz = dims
    ne = xyz_flat.shape[0]
    xyz = np.asarray(xyz_flat, dtype=np.float64).reshape(ne, 3, nz, ny, nx)
    vel = np.asarray(vel_flat, dtype=np.float64).reshape(ne, 3, nz, ny, nx)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xr, xs, xt = ref_derivatives(x, d_i, d_j, d_k)
    yr, ys, yt = ref_derivatives(y, d_i, d_j, d_k)
    zr, zs, zt = ref_derivatives(z, d_i, d_j, d_k)

    jmat = np.empty((ne, nz, ny, nx, 3, 3), dtype=np.float64)
    jmat[..., 0, 0] = xr
    jmat[..., 0, 1] = xs
    jmat[..., 0, 2] = xt
    jmat[..., 1, 0] = yr
    jmat[..., 1, 1] = ys
    jmat[..., 1, 2] = yt
    jmat[..., 2, 0] = zr
    jmat[..., 2, 1] = zs
    jmat[..., 2, 2] = zt

    flat_j = jmat.reshape(-1, 3, 3)
    det = np.linalg.det(flat_j)
    good = np.isfinite(det) & (np.abs(det) > 1.0e-13)
    bad_count = int(det.size - np.count_nonzero(good))
    inv_j = np.full_like(flat_j, np.nan)
    if np.any(good):
        inv_j[good] = np.linalg.inv(flat_j[good])
    inv_j = inv_j.reshape(ne, nz, ny, nx, 3, 3)

    grad = np.empty((ne, nz, ny, nx, 3, 3), dtype=np.float64)
    for comp in range(3):
        fr, fs, ft = ref_derivatives(vel[:, comp], d_i, d_j, d_k)
        gref = np.stack((fr, fs, ft), axis=-1)
        grad[..., comp, :] = np.einsum("...rp,...r->...p", inv_j, gref, optimize=True)

    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    eps_node = 2.0 * nu * np.sum(strain * strain, axis=(-2, -1))
    eps_node[~good.reshape(ne, nz, ny, nx)] = np.nan
    eps = average_vertices(eps_node)

    cx = average_vertices(x)
    cy = average_vertices(y)
    cz = average_vertices(z)

    ei = np.stack((edge_i(x), edge_i(y), edge_i(z)), axis=-1)
    ej = np.stack((edge_j(x), edge_j(y), edge_j(z)), axis=-1)
    ek = np.stack((edge_k(x), edge_k(y), edge_k(z)), axis=-1)
    d1 = np.linalg.norm(ei, axis=-1)
    d2 = np.linalg.norm(ej, axis=-1)
    d3 = np.linalg.norm(ek, axis=-1)
    delta_geom = np.cbrt(np.maximum(d1 * d2 * d3, 0.0))
    delta_max = np.maximum.reduce((d1, d2, d3))
    volume = np.abs(np.einsum("...i,...i->...", ei, np.cross(ej, ek)))

    eps_safe = np.where(np.isfinite(eps) & (eps > 0.0), eps, np.nan)
    eta = (nu**3 / eps_safe) ** 0.25
    eta = np.where(np.isfinite(eta), eta, np.inf)
    return (cx, cy, cz), eps, eta, delta_geom, delta_max, d1, d2, d3, volume, bad_count


class Hist:
    def __init__(self) -> None:
        self.edges = LOG_EDGES
        self.counts = np.zeros(self.edges.size - 1, dtype=np.int64)
        self.volume = np.zeros(self.edges.size - 1, dtype=np.float64)

    def add(self, values: np.ndarray, volume: np.ndarray) -> None:
        finite = np.isfinite(values) & (values > 0.0) & np.isfinite(volume) & (volume > 0.0)
        if not np.any(finite):
            return
        idx = np.searchsorted(self.edges, np.log10(values[finite]), side="right") - 1
        idx = np.clip(idx, 0, self.counts.size - 1)
        np.add.at(self.counts, idx, 1)
        np.add.at(self.volume, idx, volume[finite].astype(np.float64, copy=False))

    def quantile(self, q: float, weighted: bool) -> float:
        vals = self.volume if weighted and np.sum(self.volume) > 0.0 else self.counts.astype(np.float64)
        total = float(np.sum(vals))
        if total <= 0.0:
            return math.nan
        idx = int(np.searchsorted(np.cumsum(vals), q * total, side="left"))
        idx = max(0, min(idx, vals.size - 1))
        return float(10.0 ** (0.5 * (self.edges[idx] + self.edges[idx + 1])))

    def as_dict(self) -> dict[str, object]:
        return {"log10_edges": self.edges.tolist(), "counts": self.counts.tolist(), "volume": self.volume.tolist()}


class ScalarStats:
    def __init__(self) -> None:
        self.count = 0
        self.volume = 0.0
        self.sum = 0.0
        self.weighted_sum = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.hist = Hist()

    def add(self, values: np.ndarray, volume: np.ndarray) -> None:
        finite = np.isfinite(values) & np.isfinite(volume) & (volume > 0.0)
        if not np.any(finite):
            return
        v = values[finite].astype(np.float64, copy=False)
        w = volume[finite].astype(np.float64, copy=False)
        self.count += int(v.size)
        self.volume += float(np.sum(w))
        self.sum += float(np.sum(v))
        self.weighted_sum += float(np.sum(v * w))
        self.min = min(self.min, float(np.min(v)))
        self.max = max(self.max, float(np.max(v)))
        self.hist.add(v, w)

    def as_dict(self) -> dict[str, float]:
        return {
            "count": int(self.count),
            "volume": self.volume,
            "mean": self.sum / self.count if self.count else math.nan,
            "volume_mean": self.weighted_sum / self.volume if self.volume > 0.0 else math.nan,
            "min": self.min if self.count else math.nan,
            "max": self.max if self.count else math.nan,
            "p50_count": self.hist.quantile(0.50, weighted=False),
            "p95_count": self.hist.quantile(0.95, weighted=False),
            "p99_count": self.hist.quantile(0.99, weighted=False),
            "p50_volume": self.hist.quantile(0.50, weighted=True),
            "p95_volume": self.hist.quantile(0.95, weighted=True),
            "p99_volume": self.hist.quantile(0.99, weighted=True),
        }


class Profile:
    def __init__(self, edges: np.ndarray) -> None:
        self.edges = edges.astype(np.float64)
        n = self.edges.size - 1
        self.count = np.zeros(n, dtype=np.int64)
        self.volume = np.zeros(n, dtype=np.float64)
        self.fail_volume = np.zeros(n, dtype=np.float64)
        self.weighted_sum = np.zeros(n, dtype=np.float64)
        self.max_ratio = np.full(n, np.nan, dtype=np.float64)

    def add(self, coord: np.ndarray, ratio: np.ndarray, volume: np.ndarray) -> None:
        finite = np.isfinite(coord) & np.isfinite(ratio) & np.isfinite(volume) & (volume > 0.0)
        if not np.any(finite):
            return
        idx = np.searchsorted(self.edges, coord[finite], side="right") - 1
        ok = (idx >= 0) & (idx < self.count.size)
        if not np.any(ok):
            return
        idx = idx[ok]
        r = ratio[finite][ok].astype(np.float64, copy=False)
        w = volume[finite][ok].astype(np.float64, copy=False)
        np.add.at(self.count, idx, 1)
        np.add.at(self.volume, idx, w)
        np.add.at(self.weighted_sum, idx, r * w)
        fail = r > THRESHOLD
        if np.any(fail):
            np.add.at(self.fail_volume, idx[fail], w[fail])
        current = np.nan_to_num(self.max_ratio, nan=-math.inf)
        np.maximum.at(current, idx, r)
        self.max_ratio = np.where(np.isfinite(current), current, np.nan)

    def center(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def mean(self) -> np.ndarray:
        return np.divide(self.weighted_sum, self.volume, out=np.full_like(self.volume, np.nan), where=self.volume > 0.0)

    def fail_fraction(self) -> np.ndarray:
        return np.divide(self.fail_volume, self.volume, out=np.full_like(self.volume, np.nan), where=self.volume > 0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "edges": self.edges.tolist(),
            "count": self.count.tolist(),
            "volume": self.volume.tolist(),
            "volume_mean_ratio": self.mean().tolist(),
            "max_ratio": self.max_ratio.tolist(),
            "fail_volume_fraction": self.fail_fraction().tolist(),
        }


class SectionMap:
    def __init__(self, a_edges: np.ndarray, b_edges: np.ndarray) -> None:
        self.a_edges = a_edges.astype(np.float64)
        self.b_edges = b_edges.astype(np.float64)
        self.max_ratio = np.full((self.b_edges.size - 1, self.a_edges.size - 1), np.nan, dtype=np.float64)
        self.count = np.zeros_like(self.max_ratio, dtype=np.int64)

    def add(self, a: np.ndarray, b: np.ndarray, ratio: np.ndarray) -> None:
        finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(ratio)
        if not np.any(finite):
            return
        ia = np.searchsorted(self.a_edges, a[finite], side="right") - 1
        ib = np.searchsorted(self.b_edges, b[finite], side="right") - 1
        ok = (ia >= 0) & (ia < self.a_edges.size - 1) & (ib >= 0) & (ib < self.b_edges.size - 1)
        if not np.any(ok):
            return
        flat = ib[ok] * (self.a_edges.size - 1) + ia[ok]
        current = np.nan_to_num(self.max_ratio.ravel(), nan=-math.inf)
        np.maximum.at(current, flat, ratio[finite][ok].astype(np.float64, copy=False))
        self.max_ratio = current.reshape(self.max_ratio.shape)
        self.max_ratio[np.isneginf(self.max_ratio)] = np.nan
        np.add.at(self.count.ravel(), flat, 1)

    def extent(self) -> tuple[float, float, float, float]:
        return (float(self.a_edges[0]), float(self.a_edges[-1]), float(self.b_edges[0]), float(self.b_edges[-1]))

    def as_dict(self) -> dict[str, object]:
        return {
            "a_edges": self.a_edges.tolist(),
            "b_edges": self.b_edges.tolist(),
            "count": self.count.tolist(),
            "max_ratio": self.max_ratio.tolist(),
        }


class Accumulator:
    def __init__(self, bounds: tuple[float, float, float, float, float, float], top_n: int) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        self.top_n = top_n
        self.stats = {key: ScalarStats() for key in ("max", "geom", "i", "j", "k", "eta", "epsilon", "delta_max")}
        self.total_subcells = 0
        self.valid_subcells = 0
        self.total_volume = 0.0
        self.valid_volume = 0.0
        self.bad_jacobian_nodes = 0
        self.fail = {
            key: {label: {"count": 0, "volume": 0.0} for label, *_ in LEVELS}
            for key in ("max", "geom", "i", "j", "k")
        }
        self.profiles = {
            "x": Profile(np.linspace(xmin, xmax, 181)),
            "y": Profile(np.linspace(ymin, ymax, 141)),
            "z": Profile(np.linspace(zmin, zmax, 141)),
        }
        self.sections = {
            "xy_max_over_z": SectionMap(np.linspace(xmin, xmax, 321), np.linspace(ymin, ymax, 181)),
            "xz_max_over_y": SectionMap(np.linspace(xmin, xmax, 321), np.linspace(zmin, zmax, 181)),
            "yz_max_over_x": SectionMap(np.linspace(ymin, ymax, 181), np.linspace(zmin, zmax, 181)),
        }
        self.top_rows: list[dict[str, float | str | int]] = []

    def add_top(
        self,
        e0: int,
        ratios: dict[str, np.ndarray],
        center: tuple[np.ndarray, np.ndarray, np.ndarray],
        eta: np.ndarray,
        eps: np.ndarray,
        deltas: dict[str, np.ndarray],
    ) -> None:
        flat = ratios["max"].reshape(-1)
        finite = np.isfinite(flat)
        if not np.any(finite):
            return
        take = min(self.top_n, int(np.count_nonzero(finite)))
        finite_idx = np.flatnonzero(finite)
        local = finite_idx[np.argpartition(flat[finite_idx], -take)[-take:]]
        order = local[np.argsort(flat[local])[::-1]]
        subcells_per_element = ratios["max"].shape[1] * ratios["max"].shape[2] * ratios["max"].shape[3]
        x, y, z = (arr.reshape(-1) for arr in center)
        for idx in order:
            vals = [ratios["i"].reshape(-1)[idx], ratios["j"].reshape(-1)[idx], ratios["k"].reshape(-1)[idx]]
            dominant = ("i", "j", "k")[int(np.nanargmax(vals))]
            self.top_rows.append(
                {
                    "ratio_max": float(flat[idx]),
                    "ratio_i": float(vals[0]),
                    "ratio_j": float(vals[1]),
                    "ratio_k": float(vals[2]),
                    "dominant_direction": dominant,
                    "x": float(x[idx]),
                    "y": float(y[idx]),
                    "z": float(z[idx]),
                    "eta": float(eta.reshape(-1)[idx]),
                    "epsilon": float(eps.reshape(-1)[idx]),
                    "delta_i": float(deltas["i"].reshape(-1)[idx]),
                    "delta_j": float(deltas["j"].reshape(-1)[idx]),
                    "delta_k": float(deltas["k"].reshape(-1)[idx]),
                    "element": int(e0 + idx // subcells_per_element),
                    "local_subcell": int(idx % subcells_per_element),
                }
            )
        self.top_rows.sort(key=lambda row: float(row["ratio_max"]), reverse=True)
        del self.top_rows[self.top_n :]

    def add(
        self,
        e0: int,
        center: tuple[np.ndarray, np.ndarray, np.ndarray],
        eps: np.ndarray,
        eta: np.ndarray,
        deltas: dict[str, np.ndarray],
        volume: np.ndarray,
    ) -> None:
        x, y, z = center
        ratios = {
            "max": deltas["max"] / eta,
            "geom": deltas["geom"] / eta,
            "i": deltas["i"] / eta,
            "j": deltas["j"] / eta,
            "k": deltas["k"] / eta,
        }
        valid = np.isfinite(ratios["max"]) & np.isfinite(volume) & (volume > 0.0)
        self.total_subcells += int(ratios["max"].size)
        self.total_volume += float(np.nansum(np.where(np.isfinite(volume), volume, 0.0)))
        if not np.any(valid):
            return
        self.valid_subcells += int(np.count_nonzero(valid))
        self.valid_volume += float(np.sum(volume[valid]))

        self.stats["eta"].add(eta[valid], volume[valid])
        self.stats["epsilon"].add(eps[valid], volume[valid])
        self.stats["delta_max"].add(deltas["max"][valid], volume[valid])
        for key in ("max", "geom", "i", "j", "k"):
            self.stats[key].add(ratios[key][valid], volume[valid])
            for label, level, *_ in LEVELS:
                mask = valid & (ratios[key] > level)
                count = int(np.count_nonzero(mask))
                if count:
                    self.fail[key][label]["count"] += count
                    self.fail[key][label]["volume"] += float(np.sum(volume[mask]))

        xv = x[valid]
        yv = y[valid]
        zv = z[valid]
        rv = ratios["max"][valid]
        vv = volume[valid]
        self.profiles["x"].add(xv, rv, vv)
        self.profiles["y"].add(yv, rv, vv)
        self.profiles["z"].add(zv, rv, vv)
        self.sections["xy_max_over_z"].add(xv, yv, rv)
        self.sections["xz_max_over_y"].add(xv, zv, rv)
        self.sections["yz_max_over_x"].add(yv, zv, rv)
        self.add_top(e0, ratios, center, eta, eps, deltas)

    def fail_summary(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, levels in self.fail.items():
            out[key] = {}
            for label, item in levels.items():
                out[key][label] = {
                    "count": int(item["count"]),
                    "count_fraction": item["count"] / self.valid_subcells if self.valid_subcells else math.nan,
                    "volume": float(item["volume"]),
                    "volume_fraction": item["volume"] / self.valid_volume if self.valid_volume > 0.0 else math.nan,
                }
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "total_subcells": int(self.total_subcells),
            "valid_subcells": int(self.valid_subcells),
            "total_volume_approx": self.total_volume,
            "valid_volume_approx": self.valid_volume,
            "bad_jacobian_nodes": int(self.bad_jacobian_nodes),
            "metrics": {key: value.as_dict() for key, value in self.stats.items()},
            "fail": self.fail_summary(),
            "profiles": {key: value.as_dict() for key, value in self.profiles.items()},
            "sections": {key: value.as_dict() for key, value in self.sections.items()},
            "top_worst_subcells": self.top_rows,
        }


class FullFieldWriter:
    def __init__(self, out_dir: Path, shape: tuple[int, int, int, int]) -> None:
        self.out_dir = out_dir
        self.shape = shape
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.arrays: dict[str, np.memmap] = {}
        for name in FIELD_ARRAYS_FLOAT32:
            self.arrays[name] = np.lib.format.open_memmap(self.out_dir / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        self.arrays["dominant_direction"] = np.lib.format.open_memmap(
            self.out_dir / "dominant_direction.npy", mode="w+", dtype=np.uint8, shape=shape
        )

    def write(
        self,
        e0: int,
        e1: int,
        center: tuple[np.ndarray, np.ndarray, np.ndarray],
        eps: np.ndarray,
        eta: np.ndarray,
        deltas: dict[str, np.ndarray],
        ratios: dict[str, np.ndarray],
        volume: np.ndarray,
    ) -> None:
        target = slice(e0, e1)
        self.arrays["x"][target] = center[0].astype(np.float32, copy=False)
        self.arrays["y"][target] = center[1].astype(np.float32, copy=False)
        self.arrays["z"][target] = center[2].astype(np.float32, copy=False)
        self.arrays["epsilon"][target] = eps.astype(np.float32, copy=False)
        self.arrays["eta"][target] = eta.astype(np.float32, copy=False)
        self.arrays["delta_i"][target] = deltas["i"].astype(np.float32, copy=False)
        self.arrays["delta_j"][target] = deltas["j"].astype(np.float32, copy=False)
        self.arrays["delta_k"][target] = deltas["k"].astype(np.float32, copy=False)
        self.arrays["delta_geom"][target] = deltas["geom"].astype(np.float32, copy=False)
        self.arrays["delta_max"][target] = deltas["max"].astype(np.float32, copy=False)
        self.arrays["ratio_i"][target] = ratios["i"].astype(np.float32, copy=False)
        self.arrays["ratio_j"][target] = ratios["j"].astype(np.float32, copy=False)
        self.arrays["ratio_k"][target] = ratios["k"].astype(np.float32, copy=False)
        self.arrays["ratio_geom"][target] = ratios["geom"].astype(np.float32, copy=False)
        self.arrays["ratio_max"][target] = ratios["max"].astype(np.float32, copy=False)
        self.arrays["volume"][target] = volume.astype(np.float32, copy=False)
        stacked = np.stack((ratios["i"], ratios["j"], ratios["k"]), axis=0)
        dom = (np.nanargmax(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0) + 1).astype(np.uint8)
        dom[~np.isfinite(ratios["max"])] = 0
        self.arrays["dominant_direction"][target] = dom

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()

    def metadata(self, meta: FieldMeta, coord_meta: FieldMeta, nu: float, re_r: float | None) -> dict[str, object]:
        paths = {name: str(self.out_dir / f"{name}.npy") for name in FIELD_ARRAYS_FLOAT32}
        paths["dominant_direction"] = str(self.out_dir / "dominant_direction.npy")
        return {
            "shape": list(self.shape),
            "shape_order": ["element", "k_subcell", "j_subcell", "i_subcell"],
            "dtype_float_arrays": "float32",
            "dominant_direction_encoding": {"0": "invalid", "1": "local_i", "2": "local_j", "3": "local_k"},
            "paths": paths,
            "data_field": str(meta.path),
            "coord_field": str(coord_meta.path),
            "field_time": meta.time,
            "nu": nu,
            "re_r": re_r,
            "threshold": THRESHOLD,
            "definition": "ratio_direction = Delta_direction / eta_K; ratio_max = max(local i,j,k spacing)/eta_K.",
        }


class VoxelMaxGrid:
    def __init__(self, bounds: tuple[float, float, float, float, float, float], dims: tuple[int, int, int]) -> None:
        self.bounds = bounds
        self.dims = dims
        nx, ny, nz = dims
        self.values = np.full((nz, ny, nx), -np.inf, dtype=np.float32)
        self.count = np.zeros((nz, ny, nx), dtype=np.uint32)

    def add(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, ratio: np.ndarray) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.dims
        flat_ratio = ratio.reshape(-1)
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        zf = z.reshape(-1)
        finite = np.isfinite(flat_ratio) & np.isfinite(xf) & np.isfinite(yf) & np.isfinite(zf)
        if not np.any(finite):
            return
        ix = np.floor((xf[finite] - xmin) / max(xmax - xmin, 1.0e-30) * nx).astype(np.int64)
        iy = np.floor((yf[finite] - ymin) / max(ymax - ymin, 1.0e-30) * ny).astype(np.int64)
        iz = np.floor((zf[finite] - zmin) / max(zmax - zmin, 1.0e-30) * nz).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        if not np.any(ok):
            return
        flat = iz[ok] * (ny * nx) + iy[ok] * nx + ix[ok]
        current = self.values.ravel()
        np.maximum.at(current, flat, flat_ratio[finite][ok].astype(np.float32, copy=False))
        np.add.at(self.count.ravel(), flat, 1)

    def finite_values(self) -> np.ndarray:
        return np.where(np.isfinite(self.values), self.values, np.nan)

    def save_npz(self, path: Path) -> None:
        np.savez_compressed(path, bounds=np.array(self.bounds, dtype=np.float64), dims=np.array(self.dims), ratio_max=self.finite_values(), sample_count=self.count)

    def to_pyvista_grid(self):
        if pv is None:
            raise RuntimeError("pyvista is not available")
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.dims
        dx = (xmax - xmin) / max(nx - 1, 1)
        dy = (ymax - ymin) / max(ny - 1, 1)
        dz = (zmax - zmin) / max(nz - 1, 1)
        grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dy, dz), origin=(xmin, ymin, zmin))
        values = self.finite_values()
        fill_value = float(np.nanmin(values)) if np.any(np.isfinite(values)) else 0.0
        arr = np.where(np.isfinite(values), values, fill_value).astype(np.float32, copy=False)
        grid.point_data["ratio_max"] = arr.ravel(order="C")
        return grid


def scan_bounds(coord_meta: FieldMeta, xyz: np.memmap, chunk: int) -> tuple[float, float, float, float, float, float]:
    mins = np.full(3, math.inf, dtype=np.float64)
    maxs = np.full(3, -math.inf, dtype=np.float64)
    for e0 in range(0, coord_meta.nel, chunk):
        e1 = min(coord_meta.nel, e0 + chunk)
        chunk_xyz = np.asarray(xyz[e0:e1], dtype=np.float64)
        mins = np.minimum(mins, np.nanmin(chunk_xyz, axis=(0, 2)))
        maxs = np.maximum(maxs, np.nanmax(chunk_xyz, axis=(0, 2)))
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


def iter_chunks(nel: int, chunk: int, max_elements: int | None) -> Iterable[tuple[int, int]]:
    stop = min(nel, max_elements) if max_elements is not None else nel
    for e0 in range(0, stop, chunk):
        yield e0, min(stop, e0 + chunk)


def refinement_suggestions(acc: Accumulator, nel: int, nxyz: int) -> dict[str, object]:
    directions: dict[str, object] = {}
    for key in ("i", "j", "k"):
        stats = acc.stats[key].as_dict()
        p99_count = stats["p99_count"]
        p99_volume = stats["p99_volume"]
        p100 = stats["max"]
        directions[key] = {
            "p99_count_ratio": p99_count,
            "p99_volume_ratio": p99_volume,
            "p100_ratio": p100,
            "p99_count_refine_factor": max(1.0, p99_count / THRESHOLD) if math.isfinite(p99_count) else math.nan,
            "p99_volume_refine_factor": max(1.0, p99_volume / THRESHOLD) if math.isfinite(p99_volume) else math.nan,
            "p100_refine_factor": max(1.0, p100 / THRESHOLD) if math.isfinite(p100) else math.nan,
            "p99_count_integer_factor": math.ceil(max(1.0, p99_count / THRESHOLD)) if math.isfinite(p99_count) else None,
            "p99_volume_integer_factor": math.ceil(max(1.0, p99_volume / THRESHOLD)) if math.isfinite(p99_volume) else None,
            "p100_integer_factor": math.ceil(max(1.0, p100 / THRESHOLD)) if math.isfinite(p100) else None,
        }

    def combined(label: str) -> dict[str, object]:
        factors = [float(directions[k][f"{label}_refine_factor"]) for k in ("i", "j", "k")]
        integer = [int(directions[k][f"{label}_integer_factor"]) for k in ("i", "j", "k")]
        continuous_elements = nel * math.prod(factors)
        integer_elements = nel * math.prod(integer)
        return {
            "continuous_direction_factors": {"i": factors[0], "j": factors[1], "k": factors[2]},
            "integer_direction_factors": {"i": integer[0], "j": integer[1], "k": integer[2]},
            "estimated_spectral_elements_continuous": continuous_elements,
            "estimated_spectral_elements_integer": int(integer_elements),
            "estimated_stored_gll_points_continuous": continuous_elements * nxyz,
            "estimated_stored_gll_points_integer": int(integer_elements * nxyz),
        }

    return {
        "interpretation": (
            "Factors are local spectral-element directional factors needed to reduce Delta_direction/eta_K to <= pi. "
            "p99_count is count-weighted; p99_volume is volume-weighted; p100 uses the exact observed maximum."
        ),
        "directions": directions,
        "p99_count_plan": combined("p99_count"),
        "p99_volume_plan": combined("p99_volume"),
        "p100_plan": combined("p100"),
    }


def write_worst_rows(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(summary: dict[str, object], path: Path) -> None:
    metrics = summary["accumulator"]["metrics"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["name", "count", "volume", "mean", "volume_mean", "min", "max", "p95_count", "p99_count", "p95_volume", "p99_volume"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, row in metrics.items():
            writer.writerow({field: (name if field == "name" else row.get(field, "")) for field in fields})


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def hist_curve(hist: Hist, weighted: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 10.0 ** (0.5 * (hist.edges[:-1] + hist.edges[1:]))
    values = hist.volume if weighted and np.sum(hist.volume) > 0.0 else hist.counts.astype(np.float64)
    pdf = values / np.sum(values) if np.sum(values) > 0.0 else values
    return centers, pdf, np.cumsum(pdf)


def plot_histograms(acc: Accumulator, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    for key, label, color in (
        ("geom", r"$\Delta_{geom}/\eta_K$", "#2468a8"),
        ("max", r"$\Delta_{max}/\eta_K$", "#c43b3b"),
    ):
        x, pdf, cdf = hist_curve(acc.stats[key].hist, weighted=True)
        axes[0].semilogx(x, pdf, color=color, lw=1.9, label=label)
        axes[1].semilogx(x, cdf, color=color, lw=1.9, label=label)
    for ax in axes:
        ax.axvline(THRESHOLD, color="black", ls="--", lw=1.1, label=r"$\pi$")
        ax.axvline(0.5 * THRESHOLD, color="0.3", ls=":", lw=0.9)
        ax.axvline(2.0 * THRESHOLD, color="0.3", ls=":", lw=0.9)
        ax.axvline(4.0 * THRESHOLD, color="0.3", ls=":", lw=0.9)
        ax.grid(True, which="both", alpha=0.24)
        ax.set_xlabel("ratio")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("volume-weighted PDF/bin")
    axes[1].set_ylabel("volume-weighted CDF")
    axes[0].set_title("Refined Groetzbach ratio distribution")
    axes[1].set_title("Cumulative distribution")
    axes[1].legend(loc="lower right")
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_directional(acc: Accumulator, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.5), constrained_layout=True)
    for key, label, color in (
        ("i", "local i", "#2f80c1"),
        ("j", "local j", "#2ca25f"),
        ("k", "local k", "#d95f02"),
        ("max", "max(i,j,k)", "#222222"),
    ):
        x, _pdf, cdf = hist_curve(acc.stats[key].hist, weighted=True)
        ax.semilogx(x, cdf, lw=1.9, color=color, label=label)
    for level in (0.5 * THRESHOLD, THRESHOLD, 2.0 * THRESHOLD, 4.0 * THRESHOLD):
        ax.axvline(level, color="black" if level == THRESHOLD else "0.35", ls="--" if level == THRESHOLD else ":", lw=1.0)
    ax.grid(True, which="both", alpha=0.24)
    ax.set_xlabel(r"$\Delta_{direction}/\eta_K$")
    ax.set_ylabel("volume-weighted CDF")
    ax.set_title("Directional SEM GLL-subcell resolution")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_profile(profile: Profile, xlabel: str, out: Path) -> None:
    c = profile.center()
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.3), sharex=True, constrained_layout=True)
    axes[0].semilogy(c, profile.mean(), color="#2468a8", lw=1.7, label="volume mean")
    axes[0].semilogy(c, profile.max_ratio, color="#c43b3b", lw=1.0, label="max")
    axes[0].axhline(THRESHOLD, color="black", ls="--", lw=1.0, label=r"$\pi$")
    axes[0].grid(True, which="both", alpha=0.24)
    axes[0].set_ylabel(r"$\Delta_{max}/\eta_K$")
    axes[0].legend(loc="best")
    axes[1].plot(c, profile.fail_fraction(), color="#7b3294", lw=1.8)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.24)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("failing volume fraction")
    fig.savefig(out, dpi=240)
    plt.close(fig)


def draw_body(ax: plt.Axes, key: str) -> None:
    if key == "xz_max_over_y":
        ax.add_patch(Wedge((0.0, 0.0), 1.0, 0.0, 180.0, facecolor="0.72", edgecolor="0.1", lw=0.8))
    elif key == "xy_max_over_z":
        ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="0.72", edgecolor="0.1", lw=0.8))
    elif key == "yz_max_over_x":
        ax.add_patch(Wedge((0.0, 0.0), 1.0, 0.0, 180.0, facecolor="0.72", edgecolor="0.1", lw=0.8))


def plot_section(key: str, section: SectionMap, title: str, xlabel: str, ylabel: str, out: Path) -> None:
    data = section.max_ratio
    finite = data[np.isfinite(data) & (data > 0.0)]
    vmax = max(THRESHOLD * 2.0, float(np.nanpercentile(finite, 99.5)) if finite.size else THRESHOLD * 2.0)
    vmin = max(0.05, float(np.nanpercentile(finite, 1.0)) if finite.size else 0.05)
    fig, ax = plt.subplots(figsize=(12.0, 6.0), constrained_layout=True)
    image = ax.imshow(
        data,
        origin="lower",
        extent=section.extent(),
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    aa = 0.5 * (section.a_edges[:-1] + section.a_edges[1:])
    bb = 0.5 * (section.b_edges[:-1] + section.b_edges[1:])
    try:
        ax.contour(aa, bb, data, levels=[THRESHOLD, 2.0 * THRESHOLD, 4.0 * THRESHOLD], colors=["cyan", "white", "lime"], linewidths=[0.8, 0.7, 0.7])
    except Exception:
        pass
    draw_body(ax, key)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, label=r"max $\Delta_{max}/\eta_K$")
    fig.savefig(out, dpi=240)
    plt.close(fig)


def save_2d_plots(acc: Accumulator, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    outputs: list[str] = []
    path = out_dir / "fig_01_ratio_hist_cdf.png"
    plot_histograms(acc, path)
    outputs.append(str(path))
    path = out_dir / "fig_02_directional_cdf.png"
    plot_directional(acc, path)
    outputs.append(str(path))
    for name, xlabel in (("x", "x*"), ("y", "y*"), ("z", "z*")):
        path = out_dir / f"fig_03_profile_{name}.png"
        plot_profile(acc.profiles[name], xlabel, path)
        outputs.append(str(path))
    section_specs = {
        "xy_max_over_z": ("Top projection: max ratio over z", "x*", "y*"),
        "xz_max_over_y": ("Side projection: max ratio over y", "x*", "z*"),
        "yz_max_over_x": ("Cross projection: max ratio over x", "y*", "z*"),
    }
    for idx, (key, labels) in enumerate(section_specs.items(), start=6):
        path = out_dir / f"fig_{idx:02d}_{key}.png"
        plot_section(key, acc.sections[key], labels[0], labels[1], labels[2], path)
        outputs.append(str(path))
    return outputs


def nice_ticks(lo: float, hi: float, max_ticks: int = 8) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return [lo]
    span = hi - lo
    raw_step = span / max(max_ticks - 1, 1)
    power = 10.0 ** math.floor(math.log10(abs(raw_step)))
    candidates = np.array([1.0, 2.0, 2.5, 5.0, 10.0]) * power
    step = float(candidates[np.argmin(np.abs(candidates - raw_step))])
    start = math.ceil(lo / step) * step
    ticks = []
    value = start
    while value <= hi + 1.0e-10 * max(1.0, abs(hi)):
        ticks.append(0.0 if abs(value) < 1.0e-12 else float(value))
        value += step
    return ticks[:max_ticks + 2]


def hemisphere_polydata(n_theta: int = 96, n_phi: int = 32):
    if pv is None:
        raise RuntimeError("pyvista is not available")
    points: list[tuple[float, float, float]] = []
    for ip in range(n_phi + 1):
        phi = 0.5 * math.pi * ip / n_phi
        r = math.sin(phi)
        z = math.cos(phi)
        for it in range(n_theta):
            theta = 2.0 * math.pi * it / n_theta
            points.append((r * math.cos(theta), r * math.sin(theta), z))
    faces: list[int] = []
    for ip in range(n_phi):
        for it in range(n_theta):
            a = ip * n_theta + it
            b = ip * n_theta + (it + 1) % n_theta
            c = (ip + 1) * n_theta + it
            d = (ip + 1) * n_theta + (it + 1) % n_theta
            faces.extend((3, a, c, b))
            faces.extend((3, b, c, d))
    return pv.PolyData(np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int64))


def domain_grid_lines(bounds: tuple[float, float, float, float, float, float]):
    if pv is None:
        raise RuntimeError("pyvista is not available")
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    eps = 0.001 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    x_ticks = nice_ticks(xmin, xmax)
    y_ticks = nice_ticks(ymin, ymax)
    z_ticks = nice_ticks(zmin, zmax)
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for z_face in (zmin + eps, zmax - eps):
        for value in x_ticks:
            segments.append(((value, ymin, z_face), (value, ymax, z_face)))
        for value in y_ticks:
            segments.append(((xmin, value, z_face), (xmax, value, z_face)))
    for y_face in (ymin + eps, ymax - eps):
        for value in x_ticks:
            segments.append(((value, y_face, zmin), (value, y_face, zmax)))
        for value in z_ticks:
            segments.append(((xmin, y_face, value), (xmax, y_face, value)))
    for x_face in (xmin + eps, xmax - eps):
        for value in y_ticks:
            segments.append(((x_face, value, zmin), (x_face, value, zmax)))
        for value in z_ticks:
            segments.append(((x_face, ymin, value), (x_face, ymax, value)))
    points: list[tuple[float, float, float]] = []
    lines: list[int] = []
    for start, end in segments:
        idx = len(points)
        points.extend((start, end))
        lines.extend((2, idx, idx + 1))
    return pv.PolyData(np.asarray(points, dtype=np.float32), lines=np.asarray(lines, dtype=np.int64))


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def box_corners(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return np.asarray([[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)], dtype=np.float64)


def camera_hint_for_view(view: str) -> tuple[np.ndarray, np.ndarray]:
    if view == "parallel":
        return np.asarray((8.0, -10.8, 4.55), dtype=np.float64), np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if view == "head_to_tail":
        return np.asarray((-8.0, -10.8, 4.55), dtype=np.float64), np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if view == "top":
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64), np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if view == "side_yplus":
        return np.asarray((0.0, -1.0, 0.18), dtype=np.float64), np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if view == "upstream":
        return np.asarray((-1.0, 0.0, 0.22), dtype=np.float64), np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if view == "downstream":
        return np.asarray((1.0, 0.0, 0.22), dtype=np.float64), np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    raise ValueError(view)


def fitted_camera(
    bounds: tuple[float, float, float, float, float, float],
    view: str,
    window_size: tuple[int, int] = WINDOW_SIZE,
    margin: float = 1.15,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    pts = box_corners(bounds)
    center = np.asarray([(bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, (bounds[4] + bounds[5]) * 0.5], dtype=np.float64)
    view_hint, up_hint = camera_hint_for_view(view)
    view_dir = normalize(view_hint)
    up = normalize(up_hint - np.dot(up_hint, view_dir) * view_dir)
    if float(np.linalg.norm(up)) == 0.0:
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = normalize(np.cross(up, view_dir))
    rel = pts - center
    px = rel @ right
    py = rel @ up
    aspect = window_size[0] / window_size[1]
    parallel_scale = 0.5 * max(float(px.max() - px.min()) / aspect, float(py.max() - py.min())) * margin
    diagonal = float(np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]))
    eye = center + view_dir * max(28.0, 2.0 * diagonal)
    focal = center
    return tuple(eye), tuple(focal), tuple(up), float(parallel_scale)


def set_text_property(prop, font_size: int) -> None:
    prop.SetFontSize(font_size)
    prop.SetColor(0.0, 0.0, 0.0)
    if FONT_REG.exists():
        prop.SetFontFamily(vtk.VTK_FONT_FILE)
        prop.SetFontFile(str(FONT_REG))
    else:
        prop.SetFontFamilyToTimes()


def set_axes_label_size(axes_actor, font_size: int) -> None:
    for getter_name in ("GetXAxisCaptionActor2D", "GetYAxisCaptionActor2D", "GetZAxisCaptionActor2D"):
        caption = getattr(axes_actor, getter_name)()
        prop = caption.GetCaptionTextProperty()
        set_text_property(prop, font_size)
        caption.SetCaptionTextProperty(prop)
        text_actor = caption.GetTextActor()
        text_actor.SetTextScaleModeToNone()
        text_prop = text_actor.GetTextProperty()
        set_text_property(text_prop, font_size)
        text_actor.SetTextProperty(text_prop)


def crop_to_content(image: Image.Image, margin: int) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return image
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    bottom = min(image.height, int(ys.max()) + margin + 1)
    return image.crop((left, top, right, bottom))


def load_font(path: Path, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def decorate_render(image_array: np.ndarray, output: Path, title: str, time_value: float) -> None:
    image = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(FONT_REG, 42)
    font_small = load_font(FONT_REG, 34)
    font_bold = load_font(FONT_BOLD, 40)
    draw.text((26, 18), f"t* = {time_value:.3f}", font=font, fill=(0, 0, 0), anchor="la")
    draw.text((26, 70), title, font=font_bold, fill=(0, 0, 0), anchor="la")
    legend_x = image.width - 360
    legend_y = image.height - 230
    draw.rectangle((legend_x - 24, legend_y - 24, image.width - 22, image.height - 24), fill=(255, 255, 255), outline=(210, 210, 210))
    for idx, (_key, _level, label, color, opacity) in enumerate(LEVELS):
        y = legend_y + idx * 46
        draw.rectangle((legend_x, y, legend_x + 58, y + 26), fill=color, outline=(0, 0, 0))
        draw.text((legend_x + 76, y - 3), f"{label}  alpha={opacity:.2f}", font=font_small, fill=(0, 0, 0), anchor="la")
    image = crop_to_content(image, 18)
    image.save(output)


def remove_noise_fragments(surface, label: str):
    if vtk is None or surface.n_cells == 0:
        return surface
    cleaner = vtk.vtkStaticCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.SetToleranceIsAbsolute(True)
    cleaner.SetAbsoluteTolerance(CLEAN_ABSOLUTE_TOLERANCE)
    cleaner.Update()
    cleaned = pv.wrap(cleaner.GetOutput()).triangulate()
    if cleaned.n_cells == 0:
        return cleaned
    connected = cleaned.connectivity("all")
    region_ids = np.asarray(connected.cell_data["RegionId"])
    counts = np.bincount(region_ids)
    keep_regions = counts >= MIN_CONNECTED_COMPONENT_CELLS
    keep_cells = np.flatnonzero(keep_regions[region_ids])
    removed = int(connected.n_cells - keep_cells.size)
    if removed:
        print(f"connected-component filter {label}: removed {removed:,}/{connected.n_cells:,} tiny triangles", flush=True)
    if keep_cells.size == connected.n_cells:
        out = cleaned
    elif keep_cells.size:
        out = connected.extract_cells(keep_cells).extract_surface(algorithm="dataset_surface").triangulate()
    else:
        out = pv.PolyData()
    for array_name in ("RegionId",):
        if array_name in out.cell_data:
            del out.cell_data[array_name]
        if array_name in out.point_data:
            del out.point_data[array_name]
    return out


def build_surfaces(voxel: VoxelMaxGrid, out_dir: Path) -> tuple[dict[str, object], dict[str, Path]]:
    if pv is None:
        raise RuntimeError("pyvista is not available; cannot build 3D surfaces")
    vis_dir = out_dir / "visualization_3d"
    vis_dir.mkdir(parents=True, exist_ok=True)
    grid = voxel.to_pyvista_grid()
    grid_path = vis_dir / "groetzbach_ratio_max_voxel_grid.vti"
    grid.save(grid_path)
    finite = voxel.finite_values()
    finite_vals = finite[np.isfinite(finite)]
    surface_info: dict[str, object] = {"voxel_grid": str(grid_path), "levels": {}}
    surfaces: dict[str, Path] = {}
    if finite_vals.size == 0:
        return surface_info, surfaces
    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))
    print(f"voxel ratio_max range: {vmin:.6g} .. {vmax:.6g}", flush=True)
    for key, level, label, _color, _opacity in LEVELS:
        vtp = vis_dir / f"groetzbach_ratio_max_{key}.vtp"
        if level < vmin or level > vmax:
            surface = pv.PolyData()
            surface.save(vtp)
            crossing = False
        else:
            print(f"contouring ratio_max={label} ({level:.8g})", flush=True)
            surface = grid.contour(isosurfaces=[level], scalars="ratio_max", method="contour").triangulate()
            surface = remove_noise_fragments(surface, key)
            surface.save(vtp)
            crossing = surface.n_cells > 0
        surfaces[key] = vtp
        surface_info["levels"][key] = {
            "level": level,
            "label": label,
            "vtp": str(vtp),
            "surface_points": int(surface.n_points),
            "surface_cells": int(surface.n_cells),
            "crossing_present": bool(crossing),
        }
        print(f"wrote {vtp} points={surface.n_points:,} cells={surface.n_cells:,}", flush=True)
    return surface_info, surfaces


def render_isosurface_views(surfaces: dict[str, Path], bounds: tuple[float, float, float, float, float, float], out_dir: Path, time_value: float) -> dict[str, object]:
    if pv is None:
        raise RuntimeError("pyvista is not available; cannot render 3D views")
    vis_dir = out_dir / "visualization_3d"
    vis_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for key, path in surfaces.items():
        try:
            loaded[key] = pv.read(path)
        except Exception:
            loaded[key] = pv.PolyData()
    views = (
        ("parallel", "parallel"),
        ("head_to_tail", "parallel_head_to_tail"),
        ("top", "parallel_top"),
        ("side_yplus", "parallel_side_yplus"),
        ("upstream", "parallel_upstream"),
        ("downstream", "parallel_downstream"),
    )
    pngs: dict[str, str] = {}
    for view_key, suffix in views:
        output = vis_dir / f"groetzbach_ratio_max_threshold_levels_{suffix}.png"
        plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
        plotter.set_background("white")
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        plotter.add_mesh(
            pv.Plane(center=((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, zmin), direction=(0.0, 0.0, 1.0), i_size=xmax - xmin, j_size=ymax - ymin),
            color=GROUND_PLANE_COLOR,
            opacity=GROUND_PLANE_OPACITY,
            show_edges=False,
        )
        plotter.add_mesh(domain_grid_lines(bounds), color="#3e3e3e", line_width=1.2, opacity=0.25)
        if xmin <= 1.0 and xmax >= -1.0 and ymin <= 1.0 and ymax >= -1.0 and zmin <= 1.0 and zmax >= 0.0:
            hemi = hemisphere_polydata().clip_box(bounds=bounds, invert=False)
            plotter.add_mesh(hemi, color="#bdbdb7", opacity=0.64, smooth_shading=True, specular=0.12, specular_power=12)
        for key, _level, _label, color, opacity in LEVELS:
            surface = loaded.get(key)
            if surface is None or surface.n_cells == 0:
                continue
            plotter.add_mesh(surface, color=color, opacity=opacity, smooth_shading=True, specular=0.16, specular_power=14, show_scalar_bar=False)
        plotter.add_mesh(pv.Box(bounds=bounds).outline(), color="#555555", line_width=1.3)
        eye, focal, up, parallel_scale = fitted_camera(bounds, view_key, WINDOW_SIZE, margin=1.15)
        plotter.camera_position = [eye, focal, up]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = parallel_scale
        plotter.camera.clipping_range = (0.001, 10000.0)
        axes_actor = plotter.add_axes(
            interactive=False,
            line_width=1,
            color="#000000",
            x_color="#000000",
            y_color="#000000",
            z_color="#000000",
            xlabel="x",
            ylabel="y",
            zlabel="z",
            viewport=(0.802, 0.194, 0.932, 0.388),
        )
        if vtk is not None:
            set_axes_label_size(axes_actor, 25)
        image = plotter.screenshot(return_img=True)
        plotter.close()
        decorate_render(image, output, "Refined Groetzbach ratio isosurfaces", time_value)
        pngs[view_key] = str(output)
        print(f"wrote {output}", flush=True)
    return {"pngs": pngs, "format_reference": "PyVista off-screen, white background, ground plane, domain grid, hemisphere body, box outline, parallel projection, native axes viewport copied from Q-criterion suite."}


def run_analysis(args: argparse.Namespace, log_path: Path) -> int:
    started = time.perf_counter()
    field = args.field.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.chunk_elements <= 0:
        raise ValueError("--chunk-elements must be positive")
    voxel_dims = parse_voxel_grid(args.voxel_grid)
    nu = float(args.nu) if args.nu is not None else 1.0 / float(args.re_r)
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    re_r = None if args.nu is not None else float(args.re_r)

    print("=== Refined Groetzbach validation ===", flush=True)
    print(f"script: {Path(__file__).resolve()}", flush=True)
    print(f"field: {field}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"log: {log_path}", flush=True)
    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"numpy: {np.__version__}", flush=True)
    print(f"pyvista: {getattr(pv, '__version__', 'not available')}", flush=True)
    print(f"nu={nu:.12g}, Re_R={re_r if re_r is not None else 'specified-by-nu'}, threshold pi={THRESHOLD:.12g}", flush=True)

    shutil.copy2(Path(__file__), output_dir / Path(__file__).name)
    data_meta = read_header(field)
    if "U" not in data_meta.code:
        raise ValueError(f"field must contain velocity U; code={data_meta.code}")
    coord_path = resolve_coord_field(field, data_meta, args.coord_field, args.sequence_header)
    coord_meta = read_header(coord_path)
    if not same_layout(coord_meta, data_meta):
        raise ValueError("coordinate and data fields have different layouts")
    dims = (data_meta.nx, data_meta.ny, data_meta.nz)
    sub_shape = (data_meta.nz - 1, data_meta.ny - 1, data_meta.nx - 1)
    subcells_per_element = sub_shape[0] * sub_shape[1] * sub_shape[2]
    elements_to_process = min(data_meta.nel, args.max_elements) if args.max_elements is not None else data_meta.nel
    full_shape = (elements_to_process, *sub_shape)

    print(f"data field code={data_meta.code}, time={data_meta.time:.12g}, wdsiz={data_meta.wdsiz}", flush=True)
    print(f"coord field: {coord_path}", flush=True)
    print(f"layout: elements={data_meta.nel:,}, GLL=({data_meta.nx},{data_meta.ny},{data_meta.nz}), subcells/element={subcells_per_element}", flush=True)
    print(f"processing elements={elements_to_process:,}/{data_meta.nel:,}, chunk={args.chunk_elements}", flush=True)

    xyz = np.memmap(coord_meta.path, dtype=dtype_for(coord_meta), mode="r", offset=vector_offset(coord_meta, "X"), shape=(coord_meta.nel, 3, coord_meta.nxyz))
    vel = np.memmap(data_meta.path, dtype=dtype_for(data_meta), mode="r", offset=vector_offset(data_meta, "U"), shape=(data_meta.nel, 3, data_meta.nxyz))

    print("scanning coordinate bounds", flush=True)
    bounds = scan_bounds(coord_meta, xyz, max(args.chunk_elements, 512))
    print(f"bounds: x={bounds[0]:.12g}..{bounds[1]:.12g}, y={bounds[2]:.12g}..{bounds[3]:.12g}, z={bounds[4]:.12g}..{bounds[5]:.12g}", flush=True)

    d_i = derivative_matrix(gll_nodes(data_meta.nx))
    d_j = derivative_matrix(gll_nodes(data_meta.ny))
    d_k = derivative_matrix(gll_nodes(data_meta.nz))
    acc = Accumulator(bounds, args.top_n)
    check_dir = output_dir / "check_fields"
    writer = FullFieldWriter(check_dir, full_shape)
    voxel = VoxelMaxGrid(bounds, voxel_dims)

    for e0, e1 in iter_chunks(data_meta.nel, args.chunk_elements, args.max_elements):
        center, eps, eta, delta_geom, delta_max, delta_i, delta_j, delta_k, volume, bad = compute_epsilon_and_subcells(
            xyz[e0:e1], vel[e0:e1], d_i, d_j, d_k, dims, nu
        )
        acc.bad_jacobian_nodes += bad
        deltas = {"geom": delta_geom, "max": delta_max, "i": delta_i, "j": delta_j, "k": delta_k}
        ratios = {
            "geom": delta_geom / eta,
            "max": delta_max / eta,
            "i": delta_i / eta,
            "j": delta_j / eta,
            "k": delta_k / eta,
        }
        acc.add(e0, center, eps, eta, deltas, volume)
        writer.write(e0, e1, center, eps, eta, deltas, ratios, volume)
        voxel.add(center[0], center[1], center[2], ratios["max"])
        if e1 == elements_to_process or e1 % max(args.chunk_elements * 20, 1) == 0:
            elapsed = time.perf_counter() - started
            rate = e1 / elapsed if elapsed > 0.0 else 0.0
            print(f"  processed {e1:,}/{elements_to_process:,} elements; elapsed={elapsed:.1f}s; rate={rate:.1f} elem/s", flush=True)
        del center, eps, eta, delta_geom, delta_max, delta_i, delta_j, delta_k, volume, deltas, ratios
        gc.collect()

    print("flushing complete check fields", flush=True)
    writer.flush()
    field_meta = writer.metadata(data_meta, coord_meta, nu, re_r)
    field_meta_path = check_dir / "metadata.json"
    field_meta_path.write_text(json.dumps(field_meta, indent=2), encoding="utf-8")
    print(f"wrote {field_meta_path}", flush=True)

    acc_dict = acc.as_dict()
    suggestions = refinement_suggestions(acc, data_meta.nel, data_meta.nxyz)
    summary = {
        "run": {
            "script": str(Path(__file__).resolve()),
            "copied_script": str(output_dir / Path(__file__).name),
            "field": str(field),
            "coord_field": str(coord_path),
            "output_dir": str(output_dir),
            "console_log": str(log_path),
            "data_code": data_meta.code,
            "coord_code": coord_meta.code,
            "time": data_meta.time,
            "wdsiz": data_meta.wdsiz,
            "nu": nu,
            "re_r": re_r,
            "threshold": THRESHOLD,
            "elements_processed": elements_to_process,
            "elements_total": data_meta.nel,
            "gll_points_per_element": [data_meta.nx, data_meta.ny, data_meta.nz],
            "subcells_per_element": subcells_per_element,
            "bounds": {"x": [bounds[0], bounds[1]], "y": [bounds[2], bounds[3]], "z": [bounds[4], bounds[5]]},
            "voxel_grid_dims": list(voxel_dims),
            "elapsed_s": time.perf_counter() - started,
            "method": (
                "Instantaneous one-frame velocity U is differentiated spectrally on each Nek GLL element. "
                "The physical velocity gradient is obtained by inverting the coordinate Jacobian. "
                "Dissipation uses epsilon=2*nu*Sij*Sij, eta_K=(nu^3/epsilon)^0.25. "
                "Each GLL subcell is tested with local i, j, k edge lengths and Delta_max/eta_K <= pi."
            ),
        },
        "accumulator": acc_dict,
        "refinement_suggestions": suggestions,
        "check_field_metadata": str(field_meta_path),
    }

    summary_path = output_dir / "groetzbach_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_metrics_csv(summary, output_dir / "groetzbach_metrics.csv")
    write_worst_rows(acc.top_rows, output_dir / "groetzbach_worst_subcells.csv")
    voxel_npz = output_dir / "groetzbach_ratio_max_voxel_grid.npz"
    voxel.save_npz(voxel_npz)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {output_dir / 'groetzbach_metrics.csv'}", flush=True)
    print(f"wrote {output_dir / 'groetzbach_worst_subcells.csv'}", flush=True)
    print(f"wrote {voxel_npz}", flush=True)

    plot_outputs: list[str] = []
    if not args.skip_2d:
        print("writing 2D diagnostic plots", flush=True)
        plot_outputs = save_2d_plots(acc, output_dir / "plots_2d")
        print(f"wrote {len(plot_outputs)} 2D plots", flush=True)

    surface_info: dict[str, object] = {}
    view_info: dict[str, object] = {}
    if not args.skip_3d:
        print("building full-domain 3D isosurfaces", flush=True)
        surface_info, surfaces = build_surfaces(voxel, output_dir)
        print("rendering full-domain 3D views", flush=True)
        view_info = render_isosurface_views(surfaces, bounds, output_dir, data_meta.time)
    else:
        surface_info = {"skipped": True}
        view_info = {"skipped": True}

    summary["plots_2d"] = plot_outputs
    summary["surfaces_3d"] = surface_info
    summary["views_3d"] = view_info
    summary["run"]["elapsed_s"] = time.perf_counter() - started
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fail_pi = summary["accumulator"]["fail"]["max"]["threshold"]
    p99v = summary["accumulator"]["metrics"]["max"]["p99_volume"]
    p100 = summary["accumulator"]["metrics"]["max"]["max"]
    print("=== Verdict ===", flush=True)
    print(f"SEM DNS threshold: Delta_max/eta_K <= pi = {THRESHOLD:.8g}", flush=True)
    print(f"fail count fraction at pi: {fail_pi['count_fraction']:.6%}", flush=True)
    print(f"fail volume fraction at pi: {fail_pi['volume_fraction']:.6%}", flush=True)
    print(f"ratio_max volume p99={p99v:.8g}, max(p100)={p100:.8g}", flush=True)
    for label in ("p99_volume_plan", "p100_plan"):
        plan = suggestions[label]
        f = plan["continuous_direction_factors"]
        print(
            f"{label}: i x{f['i']:.3f}, j x{f['j']:.3f}, k x{f['k']:.3f}; "
            f"integer elements ~= {plan['estimated_spectral_elements_integer']:,}, "
            f"stored GLL points ~= {plan['estimated_stored_gll_points_integer']:,}",
            flush=True,
        )
    print(f"elapsed={summary['run']['elapsed_s']:.1f}s", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"groetzbach_validation_{timestamp()}.run.log"
    with TeeLogging(log_path):
        try:
            return run_analysis(args, log_path)
        except Exception:
            print("ERROR: validation failed", flush=True)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
