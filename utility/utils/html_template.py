"""
HTML template for the camera evaluation tool.

build_html() returns a fully self-contained HTML string embedding:
  - Plotly.js (offline-capable)
  - Camera JSON, distance array, drone defaults, threshold defaults

Physics computed in JavaScript so all controls update the plot and
table live.

Radiometric models
------------------
Thermal (LWIR):
  SNR(D) = ΔT × τ(D) × x_pix² / NETD_K     (unresolved target)
  x_rad  = sqrt(SNR_min × NETD_K / (ΔT × τ))  — radiometric pixel threshold
  D_rad  = S·f / (p × x_rad)

RGB (visible):
  N_bg   = K_VIS × E_lux × p_m² × t_s / (4 × f/#²)   [photons/pixel]
  D_SNR  = S·f_m × N_bg × C² / (p_m × SNR_min²)
  E_cross = SNR_min² / (x_detect × N_per_lux × C²)    [lux at crossover]

Diffraction (both):
  d_airy = 2.44 × λ_um × f/#   [µm]
  λ_vis = 0.55 µm | λ_LWIR = 10.0 µm
"""


def build_html(
    plotlyjs: str,
    camera_json: str,
    distances_json: str,
    drones_json: str,
    defaults_json: str,
) -> str:
    """Assemble and return the complete self-contained HTML document."""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Camera Evaluation — Drone Detection &amp; Tracking</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      display: flex;
      flex-direction: row;
      height: 100vh;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      color: #222;
      background: #f0f0f0;
    }}

    /* ── Controls panel ── */
    #controls {{
      width: 310px;
      min-width: 310px;
      padding: 14px 12px 20px;
      overflow-y: auto;
      border-right: 1px solid #d0d0d0;
      background: #fff;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}

    #controls h2 {{
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 4px;
      color: #111;
    }}

    .section-header {{
      font-weight: 700;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #666;
      margin-top: 14px;
      margin-bottom: 5px;
      padding-bottom: 3px;
      border-bottom: 1px solid #e8e8e8;
    }}

    .control-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 3px 0;
      gap: 6px;
    }}

    .control-row label {{
      color: #333;
      flex: 1;
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: default;
    }}

    .control-row input[type=number] {{
      width: 72px;
      padding: 3px 6px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 13px;
      text-align: right;
      flex-shrink: 0;
    }}

    .control-row input[type=number]:focus {{
      outline: none;
      border-color: #4a90d9;
    }}

    /* Line-style icons for threshold labels */
    .line-dash {{
      display: inline-block;
      width: 26px;
      height: 0;
      flex-shrink: 0;
      border-top-width: 2px;
      border-top-style: dashed;
    }}

    .line-dashdot {{
      display: inline-block;
      width: 26px;
      height: 2px;
      flex-shrink: 0;
      border: none;
      background: repeating-linear-gradient(
        90deg,
        currentColor 0 6px,
        transparent 6px 8px,
        currentColor 8px 10px,
        transparent 10px 13px
      );
    }}

    .col-rgb     {{ color: #1a4d7c; }}
    .col-thermal {{ color: #8b1a0e; }}

    /* Camera checkboxes */
    .cam-check {{
      display: flex;
      align-items: flex-start;
      gap: 5px;
      margin: 2px 0;
      line-height: 1.35;
    }}

    .cam-check input[type=checkbox] {{
      margin-top: 2px;
      flex-shrink: 0;
      cursor: pointer;
    }}

    .cam-check label {{ cursor: pointer; color: #333; }}

    .cam-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      flex-shrink: 0;
      margin-top: 3px;
    }}

    .hint {{
      font-size: 11px;
      color: #888;
      margin-top: 1px;
      padding-left: 2px;
    }}

    /* Atmospheric reference mini-table */
    .ref-table {{
      font-size: 10px;
      border-collapse: collapse;
      width: 100%;
      margin-top: 5px;
    }}

    .ref-table th {{
      background: #f0f0f0;
      padding: 2px 5px;
      text-align: left;
      font-weight: 600;
      color: #555;
    }}

    .ref-table td {{
      padding: 2px 5px;
      border-top: 1px solid #eee;
      color: #555;
    }}

    .ref-table tr:nth-child(even) td {{ background: #f8f8f8; }}

    /* ── Main area ── */
    #main {{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }}

    #plot {{ flex: 0 0 62vh; min-height: 0; }}

    /* ── Table legend ── */
    .table-legend {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 10px 18px;
      margin-top: 12px;
      font-size: 11px;
      color: #444;
    }}

    .legend-section {{ }}

    .legend-title {{
      font-size: 11px;
      font-weight: 700;
      color: #333;
      border-bottom: 1px solid #ddd;
      padding-bottom: 3px;
      margin-bottom: 5px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .legend-body {{
      line-height: 1.7;
    }}

    .legend-body code {{
      background: #f0f0f0;
      border-radius: 3px;
      padding: 0 3px;
      font-size: 10.5px;
      font-family: monospace;
    }}

    /* ── Summary table ── */
    #table-wrapper {{
      flex: 1;
      overflow-y: auto;
      overflow-x: auto;
      background: #fff;
      border-top: 2px solid #ddd;
      padding: 10px 16px 16px;
    }}

    #table-wrapper h3 {{
      font-size: 13px;
      font-weight: 600;
      color: #444;
      margin-bottom: 8px;
    }}

    #range-table {{
      border-collapse: collapse;
      font-size: 11.5px;
      white-space: nowrap;
    }}

    #range-table th {{
      background: #f5f5f5;
      border: 1px solid #ddd;
      padding: 4px 8px;
      text-align: left;
      font-weight: 600;
      color: #444;
      position: sticky;
      top: 0;
      z-index: 1;
    }}

    #range-table th.th-thermal {{ background: #fff0ee; color: #7a1508; }}
    #range-table th.th-rgb     {{ background: #eef3ff; color: #1a3d6b; }}
    #range-table th.th-geo     {{ background: #f0f5e8; color: #2a5018; }}
    #range-table th.th-lim     {{ background: #f0f0f0; color: #555; }}
    #range-table th.th-bind    {{ background: #f0f0f0; color: #555; }}

    #range-table td {{
      border: 1px solid #eee;
      padding: 3px 8px;
    }}

    #range-table tbody tr:nth-child(even) {{ background: #fafafa; }}
    #range-table tbody tr:hover           {{ background: #eef4ff; }}

    .td-num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}

    .badge {{ display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
    .badge-rgb     {{ background: #ddeeff; color: #1a4d7c; }}
    .badge-thermal {{ background: #ffe8e8; color: #8b1a0e; }}
    .badge-geo       {{ background: #f0f5e8; color: #2a5018; }}
    .badge-rad       {{ background: #fff0cc; color: #7a5000; }}
    .badge-pix       {{ background: #a8d898; color: #1a4010; }}
    .badge-diff      {{ background: #fde8cc; color: #7a3800; }}
    .badge-diff-lwir {{ background: #e8eef5; color: #3a5070; }}
  </style>
</head>
<body>

<!-- ═══════════════════════════════════════════════
     Controls Panel
═══════════════════════════════════════════════ -->
<div id="controls">
  <h2>Camera Evaluation</h2>
  <p class="hint">Pixels on target vs. distance.&ensp;Formula:&ensp;x = S·f / (D·p)</p>

  <!-- ── Algorithm Thresholds ── -->
  <div class="section-header">Algorithm Thresholds (px)</div>

  <div class="control-row">
    <label class="col-rgb"><span class="line-dash col-rgb"></span>RGB Detection</label>
    <input type="number" id="rgb_detect" value="10" min="1" step="1">
  </div>
  <div class="control-row">
    <label class="col-rgb"><span class="line-dashdot col-rgb"></span>RGB Tracking</label>
    <input type="number" id="rgb_track" value="40" min="1" step="1">
  </div>
  <div class="control-row">
    <label class="col-thermal"><span class="line-dash col-thermal"></span>Thermal Detection</label>
    <input type="number" id="thermal_detect" value="10" min="1" step="1">
  </div>
  <div class="control-row">
    <label class="col-thermal"><span class="line-dashdot col-thermal"></span>Thermal Tracking</label>
    <input type="number" id="thermal_track" value="40" min="1" step="1">
  </div>

  <!-- ── Atmospheric Transmission ── -->
  <div class="section-header">Atmospheric Transmission</div>
  <div class="control-row">
    <label>Extinction α (km⁻¹)</label>
    <input type="number" id="atm_alpha" value="0.01" min="0" step="0.005">
  </div>
  <p class="hint" id="atm_status">Clear air — τ@500m: 99.5% | τ@1km: 99.0%</p>
  <table class="ref-table">
    <tr><th>Condition</th><th>α (km⁻¹)</th><th>τ@500m</th><th>τ@1km</th></tr>
    <tr><td>Clear</td>      <td>0.005–0.01</td><td>99.5%</td><td>99%</td></tr>
    <tr><td>Hazy</td>       <td>0.02–0.05</td> <td>97.5%</td><td>95%</td></tr>
    <tr><td>Light rain</td> <td>0.1–0.2</td>   <td>94%</td>  <td>82%</td></tr>
    <tr><td>Heavy rain</td> <td>0.5–1.0</td>   <td>78%</td>  <td>61%</td></tr>
    <tr><td>Fog</td>        <td>&gt;1.5</td>   <td>47%</td>  <td>22%</td></tr>
  </table>

  <!-- ── Radiometric — Thermal ── -->
  <div class="section-header">Radiometric — Thermal</div>
  <div class="control-row">
    <label>ΔT drone vs. bg (K)</label>
    <input type="number" id="delta_T" value="5" min="0.001" step="0.5">
  </div>
  <p class="hint">Motors: ~10–30 K | Body: ~2–5 K | Cold: ~0.1–1 K</p>
  <div class="control-row">
    <label>SNR<sub>min</sub></label>
    <input type="number" id="snr_min" value="3" min="1" step="0.5">
  </div>

  <!-- ── Radiometric — RGB ── -->
  <div class="section-header">Radiometric — RGB</div>
  <div class="control-row">
    <label>Illuminance E (lux)</label>
    <input type="number" id="illum_lux" value="10000" min="0.01" step="100">
  </div>
  <p class="hint">Bright sun: 100 000 | Overcast: 5 000 | Dusk: 50 | Night: 1</p>
  <div class="control-row">
    <label>Contrast ΔL/L</label>
    <input type="number" id="contrast_rgb" value="0.1" min="0.001" max="1" step="0.01">
  </div>
  <p class="hint">Drone vs. background luminance difference (0–1).</p>
  <div class="control-row">
    <label>Integration time (ms)</label>
    <input type="number" id="t_int_ms" value="1" min="0.01" step="0.1">
  </div>
  <div class="control-row">
    <label>SNR<sub>min</sub> RGB</label>
    <input type="number" id="snr_min_rgb" value="3" min="1" step="0.5">
  </div>

  <!-- ── Drone Sizes ── -->
  <div class="section-header">Drone Sizes (m — widest span)</div>

  <div class="control-row">
    <label id="label_mini">Mini</label>
    <input type="number" id="size_mini" min="0.01" step="0.001">
  </div>
  <p class="hint" id="hint_mini"></p>
  <div class="control-row">
    <label id="label_medium">Medium</label>
    <input type="number" id="size_medium" min="0.01" step="0.001">
  </div>
  <p class="hint" id="hint_medium"></p>
  <div class="control-row">
    <label id="label_industrial">Industrial</label>
    <input type="number" id="size_industrial" min="0.01" step="0.001">
  </div>
  <p class="hint" id="hint_industrial"></p>

  <!-- ── RGB Cameras ── -->
  <div class="section-header">RGB Cameras</div>
  <div id="rgb_checks"></div>

  <!-- ── Thermal Cameras ── -->
  <div class="section-header">Thermal Cameras</div>
  <div id="thermal_checks"></div>
</div>

<!-- ═══════════════════════════════════════════════
     Main Area (plot + table)
═══════════════════════════════════════════════ -->
<div id="main">
  <div id="plot"></div>

  <div id="table-wrapper">
    <h3>Camera Range Summary — sorted by D_geo Industrial (detection threshold)</h3>
    <table id="range-table">
      <thead>
        <tr>
          <!-- Common specs -->
          <th>Camera</th>
          <th>Type</th>
          <th>f (mm)</th>
          <th>f/#</th>
          <th>d_airy (µm)</th>
          <th>d_eff (µm)</th>
          <th>Pitch (µm)</th>
          <th>Sensor (mm)</th>
          <th>Res</th>
          <th>FoV (°)</th>
          <th class="th-bind">Binding</th>
          <th class="th-lim">NQ</th>
          <th class="th-lim">Lim.</th>
          <!-- Geometric range -->
          <th class="th-geo" id="th_geo_mini">D_geo Mini (m)</th>
          <th class="th-geo" id="th_geo_medium">D_geo Med (m)</th>
          <th class="th-geo" id="th_geo_industrial">D_geo Ind (m)</th>
          <!-- Thermal radiometric -->
          <th class="th-thermal">NETD (mK)</th>
          <th class="th-thermal">ΔT_break (mK)</th>
          <th class="th-thermal" id="th_rad_mini">D_rad Mini (m)</th>
          <th class="th-thermal" id="th_rad_medium">D_rad Med (m)</th>
          <th class="th-thermal" id="th_rad_industrial">D_rad Ind (m)</th>
          <!-- RGB radiometric -->
          <th class="th-rgb">E_cross (lux)</th>
          <th class="th-rgb" id="th_snr_mini">D_SNR Mini (m)</th>
          <th class="th-rgb" id="th_snr_medium">D_SNR Med (m)</th>
          <th class="th-rgb" id="th_snr_industrial">D_SNR Ind (m)</th>
        </tr>
      </thead>
      <tbody id="range-table-body"></tbody>
    </table>
    <div class="table-legend">

      <div class="legend-section">
        <div class="legend-title">Airy Disk, d_eff &amp; Nyquist Ratio</div>
        <div class="legend-body">
          <code>d_airy = 2.44 · λ · f/#</code> &nbsp; (Airy disk diameter, µm)<br>
          λ = 0.55 µm (RGB) &nbsp;|&nbsp; λ = 10 µm (LWIR thermal)<br>
          <code>d_eff = √(d_airy² + p²)</code> &nbsp; RSS effective spot size — combined blur<br>
          <code>NQ = d_airy / (2p)</code> &nbsp; Nyquist sampling ratio:<br>
          &nbsp;&nbsp;NQ &lt; 1 → <span class="badge badge-pix">Pix</span> — pixel-limited (sensor undersamples optics)<br>
          &nbsp;&nbsp;NQ = 1 → critical sampling, p = d_airy/2 — sweet spot<br>
          &nbsp;&nbsp;NQ &gt; 1 → <span class="badge badge-diff">Diff ⚠</span> / <span class="badge badge-diff-lwir">Diff (LWIR)</span> — oversampled, extra pixels are wasted
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">Detection Range &amp; PSF Floor</div>
        <div class="legend-body">
          <code>D_geo = S · f / (x_detect · p)</code> &nbsp; geometric range using raw pitch p<br>
          The PSF spreads signal <em>outward</em> — it does not shrink D_geo.<br>
          <code>N_resolvable = S · f / (D · d_eff)</code> &nbsp; independent information elements<br>
          <b>PSF floor (dotted line on plot):</b> at <code>y = d_airy/p</code> px, the target subtends
          exactly one Airy disk. Below this it appears as a fixed-size PSF blob — no shape content,
          only an intensity anomaly. Since x_detect (10 px) ≫ floor (≈2–3 px for thermal),
          content is always visible at D_geo.
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">Binding Constraint</div>
        <div class="legend-body">
          Which physical limit sets the actual detection range:<br>
          <span class="badge badge-geo">Geo</span> Geometry limits — D_geo is the operative range (normal operating regime)<br>
          <span class="badge badge-rad">Rad ⚠</span> Radiometry limits — SNR insufficient before geometry is reached<br>
          Thermal: radiometry binds only when ΔT ≲ ΔT_break (see NETD column).<br>
          RGB: radiometry binds only below the crossover illuminance E_cross.
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">Thermal Radiometric Range</div>
        <div class="legend-body">
          Unresolved target SNR (fill-factor regime):<br>
          <code>x_rad = √( SNR_min · NETD / (ΔT · τ(D)) )</code><br>
          x_rad = radiometric pixel threshold (px). If x_rad &gt; x_detect, radiometry limits.<br>
          <code>ΔT_break = SNR_min · NETD / x_detect²</code><br>
          ΔT below which the radiometric constraint becomes binding.<br>
          <code>τ(D) = exp(−α · D / 1000)</code> — Beer-Lambert atmospheric transmission.
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">RGB Radiometric Range</div>
        <div class="legend-body">
          Shot-noise limited, background-flux SNR:<br>
          <code>N = K · E · p² · t / (4 · f/#²)</code> &nbsp; photons/pixel<br>
          K = 2·10¹⁵ photons/(lux·m²·s), E = illuminance (lux)<br>
          <code>D_SNR = S · f · N · C² / (p · SNR_min²)</code><br>
          C = contrast ΔL/L, t = integration time.<br>
          <code>E_cross</code> = illuminance at which D_SNR = D_det.<br>
          Below E_cross the camera becomes radiometry-limited.
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">Atmospheric Transmission</div>
        <div class="legend-body">
          <code>τ(D) = exp(−α · D / 1000)</code> &nbsp; (D in m, α in km⁻¹)<br>
          Reduces apparent target contrast and thermal ΔT with range.<br>
          Affects thermal radiometric limit (D_rad) and RGB D_SNR.<br>
          Does <em>not</em> change D_geo (pure geometry).
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">p vs d_eff: CNN Detection</div>
        <div class="legend-body">
          <b>Size targets with p</b> — networks operate on the pixel grid;<br>
          x_detect px maps to D_geo via <code>D_geo = S·f/(x·p)</code>.<br>
          <b>Contrast limits via d_eff</b> — features smaller than d_airy don't
          shrink on the sensor; they blur into background (<em>feature washout</em>).<br>
          Integrated flux is conserved → whole-drone anomaly detection remains viable.<br>
          <code>N_resolvable = S·f/(D·d_eff)</code> — independent resolved elements
          (quality metric for classification &amp; feature-based tracking).
        </div>
      </div>

      <div class="legend-section">
        <div class="legend-title">Tracker Sensitivity to d_eff</div>
        <div class="legend-body">
          <b>BBox trackers</b> (SORT, DeepSORT) — <em>minimal</em>: track box
          positions only; d_eff matters only if detection itself fails from contrast loss.<br>
          <b>Feature / optical flow</b> (KLT, CSRT) — <em>highly impactful</em>:
          need sharp gradients; large d_eff erodes edges &amp; corners → tracking lock
          degrades well before D_geo is reached.<br>
          <b>Centroid / sub-pixel</b> (long-range point source) — d_eff is
          <em>beneficial</em>: PSF spread across 2–4 px creates a light gradient
          that enables sub-pixel centroid fitting. A perfectly sharp 1-px dot gives
          only quantised position estimates.
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Plotly.js -->
<script type="text/javascript">
{plotlyjs}
</script>

<!-- App logic -->
<script type="text/javascript">
"use strict";

// ── Embedded data ──────────────────────────────────────────────────────────
const CAMERAS   = {camera_json};
const DISTANCES = {distances_json};
const DRONES    = {drones_json};
const DEFAULTS  = {defaults_json};

// ── Physics constants ──────────────────────────────────────────────────────
// K_VIS: photons per (lux · m² · s) at 555 nm, QE ≈ 0.5, Lambertian sky bg
const K_VIS = 2.0e15;

// ── Core formula ───────────────────────────────────────────────────────────
/**
 * Geometric pixels on target: x = S·f / (D·p)
 * Uses raw pixel pitch p — this is the blob size the detector (CNN/YOLO) sees.
 * The PSF (d_airy) spreads signal outward, so it does NOT shrink D_geo.
 * Diffraction effects are shown separately via the PSF floor line on the plot.
 */
function computePixels(cam, S_m) {{
  const f_m = cam.focal_length_mm / 1000.0;
  const p_m = cam.pixel_pitch_um  / 1e6;
  return DISTANCES.map(D => S_m * f_m / (D * p_m));
}}

/** D_geo = S·f / (x_thresh · p) — geometric detection range */
function computeMaxRange(cam, S_m, x_thresh) {{
  if (x_thresh <= 0) return Infinity;
  const f_m = cam.focal_length_mm / 1000.0;
  const p_m = cam.pixel_pitch_um  / 1e6;
  return S_m * f_m / (x_thresh * p_m);
}}

// ── Atmospheric transmission ───────────────────────────────────────────────
/** Beer-Lambert: τ = exp(-α_km × D_m / 1000) */
function atmTau(alpha, D_m) {{
  return Math.exp(-alpha * D_m / 1000.0);
}}

// ── Thermal radiometric ────────────────────────────────────────────────────
/**
 * Radiometric pixel threshold for thermal cameras (unresolved target):
 *   x_rad = sqrt(SNR_min × NETD_K / (ΔT × τ(D_rep)))
 * Evaluated at D_rep = 500 m as representative mid-range.
 */
function computeRadThreshThermal(cam, state) {{
  if (!cam.netd_mk) return null;
  const NETD_K = cam.netd_mk / 1000.0;
  const eff_dT = state.delta_T * atmTau(state.atm_alpha, 500);
  if (eff_dT <= 0) return null;
  return Math.sqrt(state.snr_min * NETD_K / eff_dT);
}}

// ── RGB radiometric ────────────────────────────────────────────────────────
/**
 * Background photons per pixel:
 *   N_bg = K_VIS × E_lux × p_m² × t_s / (4 × f/#²)
 * Max SNR-limited detection range:
 *   D_SNR = S × f_m × N_bg × C² / (p_m × SNR_min²)
 */
function computeRgbSNRRange(cam, S_m, state) {{
  if (!cam.f_number) return Infinity;
  const p_m  = cam.pixel_pitch_um / 1e6;
  const f_m  = cam.focal_length_mm / 1000.0;
  const t_s  = state.t_int_ms / 1000.0;
  const N_bg = K_VIS * state.illum_lux * p_m * p_m * t_s / (4.0 * cam.f_number * cam.f_number);
  const C2   = state.contrast_rgb * state.contrast_rgb;
  const snr2 = state.snr_min_rgb * state.snr_min_rgb;
  return S_m * f_m * N_bg * C2 / (p_m * snr2);
}}

/**
 * Crossover illuminance: below E_cross, D_SNR < D_geo → radiometry limits.
 *   E_cross = SNR_min² / (x_detect × N_per_lux_pixel × C²)
 * Independent of drone size S.
 */
function computeRgbEcross(cam, state) {{
  if (!cam.f_number) return null;
  const p_m        = cam.pixel_pitch_um / 1e6;
  const t_s        = state.t_int_ms / 1000.0;
  const N_per_lux  = K_VIS * p_m * p_m * t_s / (4.0 * cam.f_number * cam.f_number);
  if (N_per_lux <= 0) return null;
  const C2  = state.contrast_rgb * state.contrast_rgb;
  const snr2 = state.snr_min_rgb * state.snr_min_rgb;
  return snr2 / (state.rgb_detect * N_per_lux * C2);
}}

// ── Diffraction limit ──────────────────────────────────────────────────────
/** Airy disk diameter in µm: d_airy = 2.44 × λ_um × f/# */
function airyDisk_um(lambda_um, f_number) {{
  return 2.44 * lambda_um * f_number;
}}

// ── Atmospheric status updater ─────────────────────────────────────────────
function updateAtmStatus(alpha) {{
  const t500  = (atmTau(alpha, 500)  * 100).toFixed(1);
  const t1000 = (atmTau(alpha, 1000) * 100).toFixed(1);
  const label = alpha < 0.015 ? "Clear air"
              : alpha < 0.07  ? "Hazy"
              : alpha < 0.35  ? "Light rain"
              : alpha < 1.2   ? "Heavy rain"
              : "Fog";
  document.getElementById("atm_status").textContent =
    label + " — τ@500m: " + t500 + "% | τ@1km: " + t1000 + "%";
}}

// ── Initialise controls from embedded data ─────────────────────────────────
function initDroneInputs() {{
  for (const key of ["mini", "medium", "industrial"]) {{
    const d = DRONES[key];
    document.getElementById("size_" + key).value = d.size_m;
    const lbl = document.getElementById("label_" + key);
    if (lbl) lbl.textContent = d.label.split(" (")[0];
    const hint = document.getElementById("hint_" + key);
    if (hint) hint.textContent = d.label + " — " + d.size_m + " m";
  }}
}}

function initThresholdInputs() {{
  document.getElementById("rgb_detect").value     = DEFAULTS.detect;
  document.getElementById("rgb_track").value      = DEFAULTS.track;
  document.getElementById("thermal_detect").value = DEFAULTS.detect;
  document.getElementById("thermal_track").value  = DEFAULTS.track;
}}

// ── Build checkbox controls ────────────────────────────────────────────────
function buildControls() {{
  buildCheckboxGroup("rgb_checks",     CAMERAS.rgb,     "rgb");
  buildCheckboxGroup("thermal_checks", CAMERAS.thermal, "thermal");
}}

function buildCheckboxGroup(containerId, cams, type) {{
  const container = document.getElementById(containerId);
  cams.forEach(cam => {{
    const div = document.createElement("div");
    div.className = "cam-check";

    const cb = document.createElement("input");
    cb.type    = "checkbox";
    cb.checked = true;
    cb.id      = "cb_" + type + "_" + cam.name;
    cb.addEventListener("change", render);

    const swatch = document.createElement("span");
    swatch.className = "cam-swatch";
    swatch.style.background = cam.color;

    const lbl = document.createElement("label");
    lbl.htmlFor     = cb.id;
    lbl.textContent = cam.name + " — FoV " + cam.fov_deg + "°";

    div.appendChild(cb);
    div.appendChild(swatch);
    div.appendChild(lbl);
    container.appendChild(div);
  }});
}}

// ── Read UI state ──────────────────────────────────────────────────────────
function getCheckedIds(containerId) {{
  const boxes = document.querySelectorAll("#" + containerId + " input[type=checkbox]");
  const s = new Set();
  boxes.forEach(cb => {{ if (cb.checked) s.add(cb.id); }});
  return s;
}}

function getState() {{
  const fv = id => parseFloat(document.getElementById(id).value) || 0;
  return {{
    rgb_detect:     fv("rgb_detect")     || DEFAULTS.detect,
    rgb_track:      fv("rgb_track")      || DEFAULTS.track,
    thermal_detect: fv("thermal_detect") || DEFAULTS.detect,
    thermal_track:  fv("thermal_track")  || DEFAULTS.track,
    atm_alpha:      fv("atm_alpha"),
    delta_T:        fv("delta_T")        || 5,
    snr_min:        fv("snr_min")        || 3,
    illum_lux:      fv("illum_lux")      || 10000,
    contrast_rgb:   fv("contrast_rgb")   || 0.1,
    t_int_ms:       fv("t_int_ms")       || 1.0,
    snr_min_rgb:    fv("snr_min_rgb")    || 3,
    sizes: {{
      mini:       fv("size_mini")       || DRONES.mini.size_m,
      medium:     fv("size_medium")     || DRONES.medium.size_m,
      industrial: fv("size_industrial") || DRONES.industrial.size_m,
    }},
    enabledRgb:     getCheckedIds("rgb_checks"),
    enabledThermal: getCheckedIds("thermal_checks"),
  }};
}}

// ── Threshold style definitions ────────────────────────────────────────────
const THR_STYLES = [
  {{ id: "rgb_detect",     dash: "dash",    color: "#1a4d7c", label: "RGB Detect"     }},
  {{ id: "rgb_track",      dash: "dashdot", color: "#1a4d7c", label: "RGB Track"      }},
  {{ id: "thermal_detect", dash: "dash",    color: "#8b1a0e", label: "Thermal Detect" }},
  {{ id: "thermal_track",  dash: "dashdot", color: "#8b1a0e", label: "Thermal Track"  }},
];

const DRONE_KEYS = ["mini", "medium", "industrial"];
const COL_SUFFIX = ["", "2", "3"];

// ── Build Plotly traces ────────────────────────────────────────────────────
function buildTraces(state) {{
  const traces = [];

  DRONE_KEYS.forEach((cat, colIdx) => {{
    const S   = state.sizes[cat];
    const xax = "x" + COL_SUFFIX[colIdx];
    const yax = "y" + COL_SUFFIX[colIdx];

    // Camera pixel-count curves
    const allCams = [
      ...CAMERAS.rgb.map(c => ({{ ...c, camtype: "rgb" }})),
      ...CAMERAS.thermal.map(c => ({{ ...c, camtype: "thermal" }})),
    ];

    allCams.forEach(cam => {{
      const cbId = "cb_" + cam.camtype + "_" + cam.name;
      const isOn = cam.camtype === "rgb"
        ? state.enabledRgb.has(cbId)
        : state.enabledThermal.has(cbId);
      if (!isOn) return;

      const sw = (cam.resolution_w * cam.pixel_pitch_um / 1000).toFixed(1);
      const sh = (cam.resolution_h * cam.pixel_pitch_um / 1000).toFixed(1);

      // PSF floor: below d_airy/p pixels the target is a point source (< 1 Airy disk)
      const lambda_um_t = cam.camtype === "thermal" ? 10.0 : 0.55;
      const d_airy_t    = airyDisk_um(lambda_um_t, cam.f_number || 2.0);
      const diff_floor  = d_airy_t / cam.pixel_pitch_um;  // px

      const clippedY = computePixels(cam, S).map(px => px >= diff_floor ? px : null);

      traces.push({{
        x: DISTANCES,
        y: clippedY,
        name: cam.name,
        legendgroup: cam.name,
        showlegend: false,
        connectgaps: false,
        mode: "lines",
        line: {{ color: cam.color, width: 1.8 }},
        xaxis: xax,
        yaxis: yax,
        hovertemplate:
          "<b>" + cam.name + "</b><br>" +
          "FoV: " + cam.fov_deg + "°  f/#: " + cam.f_number + "<br>" +
          "f=" + cam.focal_length_mm + "mm  p=" + cam.pixel_pitch_um + "µm  d_airy=" + d_airy_t.toFixed(1) + "µm<br>" +
          "Sensor: " + sw + "×" + sh + " mm  Res: " + cam.resolution_w + "×" + cam.resolution_h + "<br>" +
          "Distance: %{{x}} m | Pixels on target: %{{y:.2f}}<extra></extra>",
      }});

      // Dotted PSF floor line — visible when diff_floor ≥ 1 px
      if (diff_floor >= 1.0) {{
        traces.push({{
          x: [DISTANCES[0], DISTANCES[DISTANCES.length - 1]],
          y: [diff_floor, diff_floor],
          name: cam.name + " floor",
          legendgroup: cam.name,
          showlegend: false,
          mode: "lines",
          line: {{ color: cam.color, dash: "dot", width: 0.8 }},
          xaxis: xax,
          yaxis: yax,
          hovertemplate:
            "<b>" + cam.name + " — PSF floor</b><br>" +
            diff_floor.toFixed(2) + " px (d_airy=" + d_airy_t.toFixed(1) + "µm)<br>" +
            "Below: target &lt; PSF — shape lost, detection is radiometric only" +
            "<extra></extra>",
        }});
      }}
    }});

    // Geometric algorithm threshold lines (4 horizontal lines)
    THR_STYLES.forEach(t => {{
      traces.push({{
        x: [DISTANCES[0], DISTANCES[DISTANCES.length - 1]],
        y: [state[t.id], state[t.id]],
        name: t.label,
        legendgroup: t.id,
        showlegend: false,
        mode: "lines",
        line: {{ color: t.color, dash: t.dash, width: 1.5 }},
        xaxis: xax,
        yaxis: yax,
        hoverinfo: "skip",
      }});
    }});

    // Thermal radiometric threshold lines (per camera, dotted)
    CAMERAS.thermal.forEach(cam => {{
      const cbId = "cb_thermal_" + cam.name;
      if (!state.enabledThermal.has(cbId)) return;
      const x_rad = computeRadThreshThermal(cam, state);
      if (!x_rad || x_rad < 0.05) return;  // below log-plot floor — skip
      traces.push({{
        x: [DISTANCES[0], DISTANCES[DISTANCES.length - 1]],
        y: [x_rad, x_rad],
        name: cam.name + " rad",
        legendgroup: "rad_" + cam.name,
        showlegend: false,
        mode: "lines",
        line: {{ color: cam.color, dash: "dot", width: 1.2 }},
        xaxis: xax,
        yaxis: yax,
        hovertemplate:
          "<b>" + cam.name + " — radiometric limit</b><br>" +
          "x_rad = " + x_rad.toFixed(3) + " px<br>" +
          "ΔT=" + state.delta_T + " K  NETD=" + cam.netd_mk + " mK  SNR≥" + state.snr_min +
          "<extra></extra>",
      }});
    }});
  }});

  return traces;
}}

// ── Build Plotly layout ────────────────────────────────────────────────────
function buildLayout() {{
  const axisY = isFirst => ({{
    type: "log",
    autorange: false,
    range: [Math.log10(1), Math.log10(1000)],
    gridcolor: "#e8e8e8",
    zerolinecolor: "#ccc",
    tickfont: {{ size: 11 }},
    title: isFirst ? {{ text: "Pixels on Target", font: {{ size: 12 }} }} : undefined,
  }});
  const axisX = () => ({{
    title: {{ text: "Distance (m)", font: {{ size: 12 }} }},
    range: [1, 1000],
    gridcolor: "#e8e8e8",
    zerolinecolor: "#ccc",
    tickfont: {{ size: 11 }},
  }});

  const annotations = DRONE_KEYS.map((cat, i) => ({{
    text: "<b>" + DRONES[cat].label + "</b>",
    showarrow: false,
    xref: "paper",
    yref: "paper",
    x: i / 3 + 1 / 6,
    y: 1.055,
    xanchor: "center",
    font: {{ size: 13 }},
  }}));

  return {{
    grid: {{ rows: 1, columns: 3, pattern: "independent" }},
    xaxis:  axisX(),
    xaxis2: axisX(),
    xaxis3: axisX(),
    yaxis:  axisY(true),
    yaxis2: axisY(false),
    yaxis3: axisY(false),
    annotations,
    showlegend: false,
    margin: {{ l: 70, r: 15, t: 65, b: 55 }},
    paper_bgcolor: "#fff",
    plot_bgcolor:  "#fafafa",
    hovermode: "closest",
  }};
}}

// ── Render summary table ───────────────────────────────────────────────────
function renderTable(state) {{
  // Update dynamic column headers
  document.getElementById("th_geo_mini").textContent       = "D_geo " + DRONES.mini.label.split(" (")[0] + " (m)";
  document.getElementById("th_geo_medium").textContent     = "D_geo " + DRONES.medium.label.split(" (")[0] + " (m)";
  document.getElementById("th_geo_industrial").textContent = "D_geo " + DRONES.industrial.label.split(" (")[0] + " (m)";
  document.getElementById("th_rad_mini").textContent       = "D_rad " + DRONES.mini.label.split(" (")[0] + " (m)";
  document.getElementById("th_rad_medium").textContent     = "D_rad " + DRONES.medium.label.split(" (")[0] + " (m)";
  document.getElementById("th_rad_industrial").textContent = "D_rad " + DRONES.industrial.label.split(" (")[0] + " (m)";
  document.getElementById("th_snr_mini").textContent       = "D_SNR " + DRONES.mini.label.split(" (")[0] + " (m)";
  document.getElementById("th_snr_medium").textContent     = "D_SNR " + DRONES.medium.label.split(" (")[0] + " (m)";
  document.getElementById("th_snr_industrial").textContent = "D_SNR " + DRONES.industrial.label.split(" (")[0] + " (m)";

  const allCams = [
    ...CAMERAS.rgb.map(c => ({{ ...c, camtype: "rgb"     }})),
    ...CAMERAS.thermal.map(c => ({{ ...c, camtype: "thermal" }})),
  ].filter(cam => {{
    const id = "cb_" + cam.camtype + "_" + cam.name;
    return cam.camtype === "rgb" ? state.enabledRgb.has(id) : state.enabledThermal.has(id);
  }});

  // Attach sort key: D_geo industrial (detection threshold)
  const x_detect = cam => cam.camtype === "rgb" ? state.rgb_detect : state.thermal_detect;
  allCams.forEach(cam => {{
    cam._dgeo_ind = computeMaxRange(cam, state.sizes.industrial, x_detect(cam));
  }});
  allCams.sort((a, b) => b._dgeo_ind - a._dgeo_ind);

  const fmt  = v => (v === Infinity || v > 999999) ? ">1 Mm"
                  : v > 9999 ? (v/1000).toFixed(0) + " km"
                  : Math.round(v).toString();
  const fmtL = v => (v === null || v === undefined) ? "—"
                  : v > 999999 ? ">1 Mlux"
                  : v > 9999   ? (v/1000).toFixed(0) + " klux"
                  : Math.round(v).toString();

  const tbody = document.getElementById("range-table-body");
  tbody.innerHTML = "";

  allCams.forEach(cam => {{
    const isThermal = cam.camtype === "thermal";
    const isRgb     = cam.camtype === "rgb";

    // Diffraction — Nyquist criterion: critical sampling at p = d_airy/2 (NQ = 1)
    const lambda_um  = isRgb ? 0.55 : 10.0;
    const d_airy     = airyDisk_um(lambda_um, cam.f_number || 2.0);
    const d_eff      = Math.sqrt(d_airy * d_airy + cam.pixel_pitch_um * cam.pixel_pitch_um);
    // NQ = d_airy/(2p): 1=critical, >1=oversampled (Diff), <1=undersampled (Pix)
    const nq_ratio   = d_airy / (2 * cam.pixel_pitch_um);
    const isDiff     = nq_ratio > 1;  // Nyquist: Diff when d_airy > 2p

    const nqColor = nq_ratio > 1.2 ? "#c04000" : nq_ratio > 1.0 ? "#7a5000" : "#1a5010";
    const nq_td   = '<td class="td-num" style="color:' + nqColor + ';font-weight:600">'
                  + nq_ratio.toFixed(2) + '</td>';

    let limBadge;
    if (!isDiff) {{
      limBadge = '<span class="badge badge-pix">Pix</span>';
    }} else if (isThermal) {{
      limBadge = '<span class="badge badge-diff-lwir" title="LWIR: d_airy > 2p — pixels oversample the PSF (NQ=' + nq_ratio.toFixed(2) + ')">Diff (LWIR)</span>';
    }} else {{
      limBadge = '<span class="badge badge-diff" title="d_airy > 2p — pixels oversample the PSF, empty magnification (NQ=' + nq_ratio.toFixed(2) + ')">Diff ⚠</span>';
    }}

    // Sensor physical size
    const sw = (cam.resolution_w * cam.pixel_pitch_um / 1000).toFixed(1);
    const sh = (cam.resolution_h * cam.pixel_pitch_um / 1000).toFixed(1);

    // D_geo for each drone
    const x_thr = x_detect(cam);
    const dgeo  = {{
      mini:       computeMaxRange(cam, state.sizes.mini,       x_thr),
      medium:     computeMaxRange(cam, state.sizes.medium,     x_thr),
      industrial: computeMaxRange(cam, state.sizes.industrial, x_thr),
    }};

    // Thermal radiometric
    let netd_td = "<td>—</td>", dtbreak_td = "<td>—</td>";
    let drad_tds = "<td>—</td><td>—</td><td>—</td>";
    let th_bind_td = "<td>—</td>";
    if (isThermal) {{
      const x_rad = computeRadThreshThermal(cam, state);
      const drad  = {{
        mini:       x_rad ? computeMaxRange(cam, state.sizes.mini,       x_rad) : Infinity,
        medium:     x_rad ? computeMaxRange(cam, state.sizes.medium,     x_rad) : Infinity,
        industrial: x_rad ? computeMaxRange(cam, state.sizes.industrial, x_rad) : Infinity,
      }};
      // dt_break_mk = SNR_min × NETD_mK / x_detect² — already in mK (NETD is in mK, x is dimensionless)
      const dt_break_mk = state.snr_min * cam.netd_mk / (x_thr * x_thr);
      const dt_break_display = dt_break_mk < 1 ? dt_break_mk.toFixed(3) : dt_break_mk.toFixed(1);
      const isRadLim = x_rad && (x_rad > x_thr);
      const thBind = isRadLim
        ? '<span class="badge badge-rad">Rad ⚠</span>'
        : '<span class="badge badge-geo">Geo</span>';

      netd_td   = '<td class="td-num">' + cam.netd_mk + '</td>';
      dtbreak_td = '<td class="td-num">' + dt_break_display + '</td>';
      drad_tds  = '<td class="td-num">' + fmt(drad.mini) + '</td>' +
                  '<td class="td-num">' + fmt(drad.medium) + '</td>' +
                  '<td class="td-num">' + fmt(drad.industrial) + '</td>';
      th_bind_td = '<td>' + thBind + '</td>';
    }}

    // RGB radiometric
    let ecross_td = "<td>—</td>";
    let dsnr_tds  = "<td>—</td><td>—</td><td>—</td>";
    let rgb_bind_td = "<td>—</td>";
    if (isRgb) {{
      const e_cross = computeRgbEcross(cam, state);
      const dsnr    = {{
        mini:       computeRgbSNRRange(cam, state.sizes.mini,       state),
        medium:     computeRgbSNRRange(cam, state.sizes.medium,     state),
        industrial: computeRgbSNRRange(cam, state.sizes.industrial, state),
      }};
      const isRgbRadLim = dsnr.industrial < dgeo.industrial;
      const rgbBind = isRgbRadLim
        ? '<span class="badge badge-rad">Rad ⚠</span>'
        : '<span class="badge badge-geo">Geo</span>';

      ecross_td   = '<td class="td-num">' + fmtL(e_cross) + '</td>';
      dsnr_tds    = '<td class="td-num">' + fmt(dsnr.mini) + '</td>' +
                    '<td class="td-num">' + fmt(dsnr.medium) + '</td>' +
                    '<td class="td-num">' + fmt(dsnr.industrial) + '</td>';
      rgb_bind_td = '<td>' + rgbBind + '</td>';
    }}

    // Badges
    const typeBadge = isThermal
      ? '<span class="badge badge-thermal">Thermal</span>'
      : '<span class="badge badge-rgb">RGB</span>';
    const displayName = cam.name.replace(/\s*\(\d+(\.\d+)?mm\)/i, "").trim();

    const tr = document.createElement("tr");
    tr.innerHTML =
      '<td><span style="color:' + cam.color + ';margin-right:5px">■</span>' + displayName + '</td>' +
      '<td>' + typeBadge + '</td>' +
      '<td class="td-num">' + cam.focal_length_mm + '</td>' +
      '<td class="td-num">' + (cam.f_number || "—") + '</td>' +
      '<td class="td-num">' + d_airy.toFixed(1) + '</td>' +
      '<td class="td-num">' + d_eff.toFixed(1) + '</td>' +
      '<td class="td-num">' + cam.pixel_pitch_um + '</td>' +
      '<td class="td-num">' + sw + '×' + sh + '</td>' +
      '<td class="td-num">' + cam.resolution_w + '×' + cam.resolution_h + '</td>' +
      '<td class="td-num">' + cam.fov_deg + '</td>' +
      // Single binding column — shows the relevant binding for this camera type
      (isThermal ? th_bind_td : rgb_bind_td) +
      nq_td +
      '<td>' + limBadge + '</td>' +
      // D_geo
      '<td class="td-num">' + fmt(dgeo.mini) + '</td>' +
      '<td class="td-num">' + fmt(dgeo.medium) + '</td>' +
      '<td class="td-num">' + fmt(dgeo.industrial) + '</td>' +
      // Thermal rad
      netd_td + dtbreak_td + drad_tds +
      // RGB rad
      ecross_td + dsnr_tds;

    tbody.appendChild(tr);
  }});
}}

// ── Master render ──────────────────────────────────────────────────────────
function render() {{
  const state = getState();
  updateAtmStatus(state.atm_alpha);
  Plotly.react("plot", buildTraces(state), buildLayout(), {{ responsive: true }});
  renderTable(state);
}}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {{
  initDroneInputs();
  initThresholdInputs();
  buildControls();

  document.querySelectorAll("#controls input").forEach(el => {{
    el.addEventListener("input", render);
  }});

  render();
}});
</script>
</body>
</html>
"""
