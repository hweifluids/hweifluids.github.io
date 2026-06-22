# Groetzbach Verification Output Guide

This guide tells an agent how to interpret outputs from
`run_groetzbach_validation_standalone.py`.

## Primary Files

Given an output directory, read these first:

- `groetzbach_summary.json`: authoritative run metadata, statistics, failure fractions, refinement suggestions, and visualization file paths.
- `groetzbach_metrics.csv`: compact table of scalar statistics for `ratio_max`, `ratio_i`, `ratio_j`, `ratio_k`, `eta`, `epsilon`, and `delta_max`.
- `groetzbach_worst_subcells.csv`: worst local subcells ranked by `ratio_max`.
- `check_fields/metadata.json`: shape, array names, and meaning of complete subcell fields.
- `visualization_3d/`: full-domain isosurface `.vtp` files and PNG views.
- `plots_2d/`: histograms, directional CDFs, profiles, and projected max-ratio maps.
- `groetzbach_validation_*.run.log`: console transcript from the run.

## Method To Report

State the method explicitly:

- The script validates one instantaneous Nek field, not an averaged field, unless the input file itself is an averaged field.
- It reads velocity `U`; coordinates `X` come from the same file when present, otherwise from the sequence head or an explicitly supplied coordinate field.
- Velocity gradients are evaluated spectrally on GLL nodes inside each spectral element.
- The physical velocity gradient is obtained by inverting the coordinate Jacobian.
- Dissipation is `epsilon = 2 * nu * Sij * Sij`.
- Kolmogorov length is `eta_K = (nu^3 / epsilon)^0.25`.
- Each GLL subcell is checked with local edge lengths:
  - `ratio_i = Delta_i / eta_K`
  - `ratio_j = Delta_j / eta_K`
  - `ratio_k = Delta_k / eta_K`
  - `ratio_max = max(Delta_i, Delta_j, Delta_k) / eta_K`
- The refined Groetzbach SEM DNS threshold is `ratio <= pi`.

Use the words `local i`, `local j`, and `local k`. Do not call them global
`x`, `y`, or `z` directions unless a separate geometric mapping analysis has
been done.

## Verdict Fields

In `groetzbach_summary.json`, use:

- `run.field`: input field.
- `run.coord_field`: coordinate source.
- `run.time`: nondimensional time from the Nek header.
- `run.nu` and `run.re_r`: viscosity and Reynolds number assumption.
- `run.elements_processed`: processed spectral elements.
- `accumulator.valid_subcells`: valid tested GLL subcells.
- `accumulator.bad_jacobian_nodes`: mesh/metric quality warning.
- `accumulator.fail.max.threshold.count_fraction`: fraction of subcells with `ratio_max > pi`.
- `accumulator.fail.max.threshold.volume_fraction`: volume fraction with `ratio_max > pi`.
- `accumulator.metrics.max.p99_volume`: volume-weighted p99 of `ratio_max`.
- `accumulator.metrics.max.max`: p100/max of `ratio_max`.

Recommended conclusion format:

```text
SEM DNS threshold: Delta_max / eta_K <= pi.
The case fails/passes by the strict p100 criterion.
At pi, failing count fraction is ... and failing volume fraction is ....
ratio_max p99_volume is ..., and p100/max is ....
Bad Jacobian nodes: ....
```

## Refinement Suggestions

Use `refinement_suggestions` in `groetzbach_summary.json`.

For a p99 recommendation, prefer `p99_volume_plan` unless the user explicitly
asks for count-weighted p99. Report:

- `continuous_direction_factors.i`
- `continuous_direction_factors.j`
- `continuous_direction_factors.k`
- `integer_direction_factors`
- `estimated_spectral_elements_integer`
- `estimated_stored_gll_points_integer`

For strict p100, use `p100_plan`.

Interpretation:

- A factor of `1.0` means no extra refinement is required in that local direction for the selected target.
- A factor of `2.0` means the local spacing should be roughly halved in that local direction.
- Integer estimates multiply the current spectral-element count by the integer directional factors. They are conservative and can become very large.

## Complete Field Arrays

`check_fields/metadata.json` defines the complete subcell arrays.

The normal shape is:

```text
[element, k_subcell, j_subcell, i_subcell]
```

Important arrays:

- `x.npy`, `y.npy`, `z.npy`: subcell center coordinates.
- `epsilon.npy`: dissipation.
- `eta.npy`: Kolmogorov length.
- `delta_i.npy`, `delta_j.npy`, `delta_k.npy`: local subcell edge lengths.
- `ratio_i.npy`, `ratio_j.npy`, `ratio_k.npy`, `ratio_max.npy`: Groetzbach ratios.
- `volume.npy`: subcell volume approximation.
- `dominant_direction.npy`: `0=invalid`, `1=local_i`, `2=local_j`, `3=local_k`.

Use these arrays when the user asks where the failure is located, which
direction dominates, or wants new custom plots.

## 3D Visualization

The main full-domain 3D outputs are in `visualization_3d/`.

The isosurface levels are:

- `half_threshold`: `0.5 * pi`
- `threshold`: `pi`
- `double_threshold`: `2 * pi`
- `quad_threshold`: `4 * pi`

The rendered PNGs use the Q-criterion suite style:

- white background
- ground plane
- domain grid
- hemisphere body when inside bounds
- full domain box outline
- parallel projection
- small native axes viewport
- semi-transparent nested threshold surfaces

The first three full-domain views normally to inspect are:

- `groetzbach_ratio_max_threshold_levels_parallel.png`
- `groetzbach_ratio_max_threshold_levels_parallel_head_to_tail.png`
- `groetzbach_ratio_max_threshold_levels_parallel_top.png`

Also check side/upstream/downstream views when explaining spatial distribution.

## Caveats

- The 3D surfaces are generated from the voxelized `ratio_max` grid saved as
  `groetzbach_ratio_max_voxel_grid.npz` and `.vti`; the complete original
  subcell data are in `check_fields/`.
- Histogram p99 values are binned estimates. The p100/max statistic is from the
  streaming exact maximum.
- If the input field was produced with regularization, this validation checks
  the resolved field against the Groetzbach spacing criterion; by itself it does
  not prove the simulation is an unregularized DNS.
- If a user asks whether the data are instantaneous or averaged, answer from
  the actual input file and `run.method`; do not infer it from file names alone.

## Example From The 260606 Run

For the run on
`E:\2_Cases\1_CFD\1_Basic_Flow_Mechanics\Hemisphere\5_Results\260606_CPU_SentToArrhenius_1370\hemi_restart0.f00000`,
the completed output in `Z:\Hemisphere\0_Scratch` reported:

- time: `t* = 550`
- spectral elements: `189,114`
- valid subcells: `64,866,102`
- bad Jacobian nodes: `0`
- `ratio_max > pi` count fraction: `12.054433%`
- `ratio_max > pi` volume fraction: `10.561990%`
- `ratio_max` volume p99: `6.2194216`
- `ratio_max` p100/max: `16.226545`
- p99-volume refinement factors: `i x1.816`, `j x1.666`, `k x1.000`
- p100 refinement factors: `i x5.165`, `j x2.560`, `k x2.821`
