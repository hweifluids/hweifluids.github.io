#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
from PIL import Image
import pyvista as pv
import vtk


SCRIPT_DIR = Path(__file__).resolve().parent
HEMISPHERE_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE = HEMISPHERE_ROOT.parent
DEFAULT_INPUT_DIR = HEMISPHERE_ROOT / "5_Results" / "260616_Hemi650" / "260614_GPU_650_Clean"
DEFAULT_OUTPUT_DIR = HEMISPHERE_ROOT / "6_Outputs" / "2_RawFigures" / "HM650"
DEFAULT_CASE_PREFIX = "HM650_avg"

RECIRC_SCRIPT = SCRIPT_DIR / "plot_re650_mean_recirc.py"
Q_LCI_SCRIPT = SCRIPT_DIR / "plot_q_lci_six_views.py"
VIDEO_SCRIPT = SCRIPT_DIR / "render_q_overlay_video.py"

MESH_SCRIPT_DIR = HEMISPHERE_ROOT / "3_DNS_Cases" / "scripts"
sys.path.insert(0, str(MESH_SCRIPT_DIR))
import generate_hemi_msh as meshgen  # noqa: E402
from plot_mesh_sections import plane_intersection_segments  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
import plot_q_lci_six_views as qvis  # noqa: E402


ALL_MODULES = ("geo", "mesh", "slices", "volume", "video")
DEFAULT_MODULES = ("geo", "mesh", "slices", "volume")


def parse_modules(text: str | None) -> tuple[str, ...]:
    if not text:
        return DEFAULT_MODULES
    parts = tuple(part.strip().lower() for part in text.replace(";", ",").split(",") if part.strip())
    if not parts:
        return DEFAULT_MODULES
    if "all" in parts:
        return ALL_MODULES
    unknown = sorted(set(parts) - set(ALL_MODULES))
    if unknown:
        raise SystemExit(f"unknown module(s): {', '.join(unknown)}")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hemisphere visualization suite with fixed published formats.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="case directory containing Nek field files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="root output directory")
    parser.add_argument(
        "--modules",
        default=",".join(DEFAULT_MODULES),
        help="comma-separated modules: geo,mesh,slices,volume,video; use all to include video",
    )
    parser.add_argument("--case-prefix", default=DEFAULT_CASE_PREFIX, help="prefix for average-field slice/volume outputs")
    parser.add_argument("--coord-field", type=Path, default=None, help="instantaneous coordinate field; default <input>/hemi0.f00000")
    parser.add_argument("--data-field", type=Path, default=None, help="single-frame Q/LCI data field; default latest hemi0.f?????")
    parser.add_argument("--avg-coord-field", type=Path, default=None, help="mean/restart coordinate field")
    parser.add_argument("--avg-field", type=Path, default=None, help="mean field for slices and u=0 recirculation")
    parser.add_argument("--video-count", type=int, default=None, help="render the last N available instantaneous frames")
    parser.add_argument("--video-fps", type=int, default=None, help="video frame rate; default rounded snapshot rate")
    parser.add_argument("--surface-chunk-elems", type=int, default=4096, help="video Q-surface chunk size")
    parser.add_argument("--no-clean", action="store_true", help="do not clear selected output category folders before rendering")
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matched {folder / pattern}")
    return files[-1]


def resolve_case_path(case_dir: Path, path: Path | None, default_name: str | None = None) -> Path:
    if path is not None:
        return path if path.is_absolute() else (case_dir / path)
    if default_name is None:
        raise ValueError("default_name is required when path is None")
    return case_dir / default_name


def discover_hemi_indices(case_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in case_dir.glob("hemi0.f?????"):
        try:
            indices.append(int(path.name[-5:]))
        except ValueError:
            continue
    return sorted(indices)


def default_video_fps(case_dir: Path, indices: Sequence[int]) -> int:
    if len(indices) < 2:
        return 20
    first = case_dir / f"hemi0.f{indices[0]:05d}"
    second = case_dir / f"hemi0.f{indices[1]:05d}"
    try:
        t0 = qvis.read_header(first).time
        t1 = qvis.read_header(second).time
        dt = abs(t1 - t0)
        if dt > 0.0:
            return max(1, int(round(1.0 / dt)))
    except Exception:
        pass
    return 20


def run_command(command: Sequence[str], cwd: Path = WORKSPACE) -> None:
    print("running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=str(cwd), check=True)


def field_dtype(meta: qvis.FieldMeta) -> np.dtype:
    return np.dtype(meta.endian + ("f8" if meta.wdsiz == 8 else "f4"))


def coordinate_bounds(path: Path) -> tuple[float, float, float, float, float, float]:
    meta = qvis.read_header(path)
    xyz = np.memmap(
        path,
        dtype=field_dtype(meta),
        mode="r",
        offset=qvis.vector_offset(meta, "X"),
        shape=(meta.nel, 3, meta.nxyz),
    )
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for e0 in range(0, meta.nel, 4096):
        e1 = min(meta.nel, e0 + 4096)
        chunk = np.asarray(xyz[e0:e1], dtype=np.float64)
        mins = np.minimum(mins, np.nanmin(chunk, axis=(0, 2)))
        maxs = np.maximum(maxs, np.nanmax(chunk, axis=(0, 2)))
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


def append_polydata(parts: list[pv.PolyData]) -> pv.PolyData:
    append = vtk.vtkAppendPolyData()
    for part in parts:
        if part.n_points:
            append.AddInputData(part)
    append.Update()
    return pv.wrap(append.GetOutput())


def hemisphere_section_lines(hemisphere: pv.PolyData) -> pv.PolyData:
    sections: list[pv.PolyData] = []
    for z in (0.25, 0.5, 0.75):
        sections.append(hemisphere.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, z)))
    for theta in np.linspace(0.0, np.pi, 4, endpoint=False):
        normal = (-float(np.sin(theta)), float(np.cos(theta)), 0.0)
        sections.append(hemisphere.slice(normal=normal, origin=(0.0, 0.0, 0.0)))
    return append_polydata(sections)


def hemisphere_equator_from_boundary(hemisphere: pv.PolyData) -> pv.PolyData:
    return hemisphere.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )


def crop_to_content(image: Image.Image, margin: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < 248, axis=2)
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return image, (0, 0, image.width, image.height)
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    bottom = min(image.height, int(ys.max()) + margin + 1)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def adjusted_geometry_camera_eye_hint() -> tuple[float, float, float]:
    view_hint = np.asarray(qvis.CAMERA_EYE_HINT, dtype=np.float64) - np.asarray(qvis.CAMERA_TARGET_HINT, dtype=np.float64)
    view_hint[0] *= -1.0
    angle = np.deg2rad(10.0)
    rot = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adjusted = rot @ view_hint
    horizontal = float(np.linalg.norm(adjusted[:2]))
    elevation = np.arctan2(adjusted[2], horizontal) + np.deg2rad(-5.0)
    adjusted[2] = horizontal * np.tan(elevation)
    return tuple(np.asarray(qvis.CAMERA_TARGET_HINT, dtype=np.float64) + adjusted)


def set_text_property(prop: vtk.vtkTextProperty, font_size: int) -> None:
    times = Path("C:/Windows/Fonts/times.ttf")
    prop.SetFontSize(font_size)
    prop.SetColor(0.0, 0.0, 0.0)
    if times.exists():
        prop.SetFontFamily(vtk.VTK_FONT_FILE)
        prop.SetFontFile(str(times))
    else:
        prop.SetFontFamilyToTimes()


def set_axes_label_size(axes_actor: vtk.vtkAxesActor, font_size: int) -> None:
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


def render_geometry(case_dir: Path, out_dir: Path, avg_coord_field: Path) -> dict[str, object]:
    pv.OFF_SCREEN = True
    ensure_dir(out_dir)
    output_png = out_dir / "nekhem_hemi0_full_domain_geometry_parallel_head_to_tail.png"
    output_summary = out_dir / "nekhem_hemi0_full_domain_geometry_parallel_head_to_tail.summary.json"
    content_margin_px = 18
    bounds = coordinate_bounds(avg_coord_field)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    plotter = pv.Plotter(off_screen=True, window_size=qvis.WINDOW_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(
        pv.Plane(
            center=((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, zmin),
            direction=(0.0, 0.0, 1.0),
            i_size=xmax - xmin,
            j_size=ymax - ymin,
        ),
        color=qvis.GROUND_PLANE_COLOR,
        opacity=qvis.GROUND_PLANE_OPACITY,
        show_edges=False,
    )
    hemisphere = qvis.hemisphere_polydata()
    plotter.add_mesh(
        hemisphere,
        color="#bdbdb7",
        opacity=0.60,
        smooth_shading=True,
        specular=0.12,
        specular_power=12,
    )
    plotter.add_mesh(hemisphere_section_lines(hemisphere), color="#000000", line_width=1.0)
    plotter.add_mesh(hemisphere_equator_from_boundary(hemisphere), color="#000000", line_width=1.3)
    plotter.add_mesh(pv.Box(bounds=bounds).outline(), color="#000000", line_width=1.1)

    camera_eye_hint = adjusted_geometry_camera_eye_hint()
    eye, focal, up, parallel_scale = qvis.fitted_camera(
        reverse=False,
        bounds=bounds,
        camera_eye_hint=camera_eye_hint,
    )
    plotter.camera_position = [eye, focal, up]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.clipping_range = (0.01, 200.0)
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
    set_axes_label_size(axes_actor, 25)
    image_array = plotter.screenshot(return_img=True)
    plotter.close()

    raw_image = Image.fromarray(image_array)
    image, content_crop = crop_to_content(raw_image, content_margin_px)
    image.save(output_png)
    summary = {
        "output_png": str(output_png),
        "case_dir": str(case_dir),
        "bounds_source": str(avg_coord_field),
        "domain_bounds": list(bounds),
        "window_size": list(qvis.WINDOW_SIZE),
        "content_crop": list(content_crop),
        "content_margin_px": content_margin_px,
        "final_png_size": list(image.size),
        "camera_eye": list(eye),
        "camera_focal": list(focal),
        "camera_up": list(up),
        "parallel_scale": parallel_scale,
        "hemisphere_opacity": 0.60,
        "hemisphere_section_line_width": 1.0,
        "hemisphere_equator_line_width": 1.3,
        "domain_boundary_line_width": 1.1,
        "native_axes_viewport": [0.802, 0.194, 0.932, 0.388],
        "native_axes_label_font_size": 25,
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_png}", flush=True)
    print(f"wrote {output_summary}", flush=True)
    return {"png": str(output_png), "summary": str(output_summary)}


def set_mesh_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 17,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def render_mesh(out_dir: Path) -> dict[str, object]:
    ensure_dir(out_dir)
    set_mesh_style()
    elements, _spacing_metrics = meshgen.build_structured_elements()
    segments, _ = plane_intersection_segments(elements, axis=1, value=0.0)

    fig, ax = plt.subplots(figsize=(9.8, 2.35))
    collection = LineCollection(segments, colors="0.26", linewidths=0.180, alpha=0.62)
    ax.add_collection(collection)
    theta = np.linspace(0.0, np.pi, 320)
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=0.95)
    ax.axhline(0.0, color="black", lw=0.65)
    ax.set_xlim(meshgen.XMIN, meshgen.XMAX)
    ax.set_ylim(0.0, meshgen.ZTOP_TARGET)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x*", fontsize=17, labelpad=1, fontstyle="italic")
    ax.set_ylabel("z*", fontsize=17, labelpad=5, fontstyle="italic")
    ax.set_xticks([-5, 0, 5, 10, 15, 20])
    ax.set_yticks([0, 2, 4])
    ax.tick_params(labelsize=15, length=4.5, width=0.85, pad=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.25, top=0.975)

    pdf = out_dir / "fig_mesh_section_internal_grid4x.pdf"
    png = out_dir / "fig_mesh_section_internal_grid4x.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    summary = {
        "pdf": str(pdf),
        "png": str(png),
        "segments": len(segments),
        "internal_linewidth": 0.180,
        "png_dpi": 600,
        "format_reference": "fig_mesh_section_internal_grid4x",
    }
    summary_path = out_dir / "fig_mesh_section_internal_grid4x.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"segments={len(segments)}", flush=True)
    print(f"wrote {pdf}", flush=True)
    print(f"wrote {png}", flush=True)
    return {"pdf": str(pdf), "png": str(png), "summary": str(summary_path)}


def run_slices(case_dir: Path, out_dir: Path, avg_coord: Path, avg_field: Path, prefix: str) -> dict[str, object]:
    ensure_dir(out_dir)
    command = [
        sys.executable,
        RECIRC_SCRIPT,
        "--case-dir",
        case_dir,
        "--coord-field",
        avg_coord,
        "--avg-field",
        avg_field,
        "--out-dir",
        out_dir,
        "--prefix",
        prefix,
        "--skip-3d",
    ]
    run_command(command)
    return {"summary": str(out_dir / f"{prefix}_summary.json")}


def run_volume(
    case_dir: Path,
    out_dir: Path,
    avg_coord: Path,
    avg_field: Path,
    inst_coord: Path,
    inst_field: Path,
    prefix: str,
) -> dict[str, object]:
    ensure_dir(out_dir)
    run_command(
        [
            sys.executable,
            RECIRC_SCRIPT,
            "--case-dir",
            case_dir,
            "--coord-field",
            avg_coord,
            "--avg-field",
            avg_field,
            "--out-dir",
            out_dir,
            "--prefix",
            prefix,
            "--skip-sections",
        ]
    )
    run_command(
        [
            sys.executable,
            Q_LCI_SCRIPT,
            "--case-dir",
            case_dir,
            "--coord-field",
            inst_coord,
            "--data-field",
            inst_field,
            "--out-dir",
            out_dir,
        ]
    )
    data_tag = inst_field.name.replace(".", "_")
    return {
        "recirc_summary": str(out_dir / f"{prefix}_summary.json"),
        "q_lci_summary": str(out_dir / f"nekhem_{data_tag}_Q_LCI_six_views.summary.json"),
    }


def run_video(
    case_dir: Path,
    out_dir: Path,
    inst_coord: Path,
    video_count: int | None,
    fps: int | None,
    surface_chunk_elems: int,
) -> dict[str, object]:
    ensure_dir(out_dir)
    frames_dir = out_dir / "frames"
    ensure_dir(frames_dir)
    indices = discover_hemi_indices(case_dir)
    if not indices:
        raise FileNotFoundError(f"no hemi0.f????? frames in {case_dir}")
    if video_count is None:
        selected = indices
    else:
        if video_count <= 0:
            raise SystemExit("--video-count must be positive")
        selected = indices[-video_count:]
    start, end = selected[0], selected[-1]
    expected = list(range(start, end + 1))
    if selected != expected:
        raise SystemExit(f"selected video frames are not contiguous: f{start:05d}..f{end:05d}")
    if fps is None:
        fps = default_video_fps(case_dir, indices)
    video = out_dir / f"nekhem_hemi0_Q0p75_Q0p2overlay_head_to_tail_{fps}fps.mp4"
    progress = out_dir / "progress.json"
    run_command(
        [
            sys.executable,
            VIDEO_SCRIPT,
            "--case-dir",
            case_dir,
            "--coord-field",
            inst_coord,
            "--video-dir",
            out_dir,
            "--start",
            start,
            "--end",
            end,
            "--fps",
            fps,
            "--frames-dir",
            frames_dir,
            "--output-video",
            video,
            "--progress-json",
            progress,
            "--surface-chunk-elems",
            surface_chunk_elems,
        ]
    )
    return {
        "start": start,
        "end": end,
        "fps": fps,
        "frames_dir": str(frames_dir),
        "video": str(video),
        "progress": str(progress),
    }


def prepare_outputs(output_dir: Path, modules: Iterable[str], clean: bool) -> dict[str, Path]:
    paths = {
        "geo_mesh": output_dir / "geo_mesh",
        "slices": output_dir / "slices",
        "volume": output_dir / "volume",
        "video": output_dir / "video",
    }
    if clean:
        if "geo" in modules or "mesh" in modules:
            clear_dir(paths["geo_mesh"])
        if "slices" in modules:
            clear_dir(paths["slices"])
        if "volume" in modules:
            clear_dir(paths["volume"])
        if "video" in modules:
            clear_dir(paths["video"])
    for path in paths.values():
        ensure_dir(path)
    return paths


def main() -> int:
    args = parse_args()
    modules = parse_modules(args.modules)
    case_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inst_coord = resolve_case_path(case_dir, args.coord_field, "hemi0.f00000").resolve()
    inst_field = (args.data_field.resolve() if args.data_field else latest_file(case_dir, "hemi0.f?????").resolve())
    avg_coord = resolve_case_path(case_dir, args.avg_coord_field, "hemi_restart0.f00000").resolve()
    avg_field = (args.avg_field.resolve() if args.avg_field else latest_file(case_dir, "avg0.f?????").resolve())

    if not inst_coord.exists():
        raise FileNotFoundError(inst_coord)
    if not inst_field.exists():
        raise FileNotFoundError(inst_field)
    if not avg_coord.exists():
        raise FileNotFoundError(avg_coord)
    if not avg_field.exists():
        raise FileNotFoundError(avg_field)

    started = time.perf_counter()
    category_dirs = prepare_outputs(output_dir, modules, clean=not args.no_clean)
    shutil.copy2(Path(__file__), output_dir / Path(__file__).name)

    summary: dict[str, object] = {
        "input_dir": str(case_dir),
        "output_dir": str(output_dir),
        "modules": list(modules),
        "case_prefix": args.case_prefix,
        "inst_coord": str(inst_coord),
        "inst_field": str(inst_field),
        "avg_coord": str(avg_coord),
        "avg_field": str(avg_field),
        "category_dirs": {key: str(value) for key, value in category_dirs.items()},
        "outputs": {},
    }

    if "geo" in modules:
        summary["outputs"]["geo"] = render_geometry(case_dir, category_dirs["geo_mesh"], avg_coord)
    if "mesh" in modules:
        summary["outputs"]["mesh"] = render_mesh(category_dirs["geo_mesh"])
    if "slices" in modules:
        summary["outputs"]["slices"] = run_slices(case_dir, category_dirs["slices"], avg_coord, avg_field, args.case_prefix)
    if "volume" in modules:
        summary["outputs"]["volume"] = run_volume(
            case_dir,
            category_dirs["volume"],
            avg_coord,
            avg_field,
            inst_coord,
            inst_field,
            args.case_prefix,
        )
    if "video" in modules:
        summary["outputs"]["video"] = run_video(
            case_dir,
            category_dirs["video"],
            inst_coord,
            args.video_count,
            args.video_fps,
            args.surface_chunk_elems,
        )

    summary["elapsed_seconds"] = time.perf_counter() - started
    summary_path = output_dir / "visualization_suite.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    print(f"elapsed_s={summary['elapsed_seconds']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
