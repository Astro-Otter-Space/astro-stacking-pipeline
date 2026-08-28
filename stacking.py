#!/usr/bin/env python3
"""
Siril stacking script for EAA & SaaS.

Generates and runs Siril scripts for:
- Automatically detect sensor type (Mono vs Color) via FITS Header.
- Isolate and stack raws by filter (HA, SII, OIII, RED, CLEAR, etc.).
- Clean spaces in filenames to immunize Siril parser.
- Use native Siril 1.2+ commands ('cd', 'convert -out=.', 'stack r_light rej').
- Properly convert FIT containers to master TIFFs (via load/save).
- Merge channels into a chromatic composite (SHO, HOO, RGB, Mono) via ImageMagick
  (channel combine + tone/color finishing chain only — flip, crop, synthetic
  black channels, and star halo reduction now run on pure PIL/numpy/scipy).
- Run 100% in command-line mode (strict headless).

Usage :
    python3 stacking.py <uuid> --format=webp --dso=m31 --verbose
"""

import os
import sys
import argparse
import subprocess
import shutil
import re
import traceback
import json
from pathlib import Path
from datetime import datetime, timedelta
from astropy.io import fits
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
import numpy as np

# Root directory
BASE_DIR = Path(__file__).resolve().parent

# Valid filters accepted and recognized by the system
VALID_FILTERS = [
    'IR_CUT', 'UV_IR_CUT', 'UHC', 'CLS', 'BROADBAND', 'DUAL_NARROWBAND',
    'LRGB', 'LUMINANCE', 'L',
    'RED', 'GREEN', 'BLUE', 'RGB', 'HA', 'H_BETA',
    'OIII', 'SII', 'SOLAR', 'CLEAR'
]

NARROWBAND_FILTERS = {'HA', 'OIII', 'SII', 'HALPHA', 'H_BETA'}
BROADBAND_FILTERS = {'RED', 'GREEN', 'BLUE', 'CLEAR', 'LUMINANCE', 'L', 'R', 'G', 'B'}
VERBOSE = False
DSO_NAME = ''
VALID_EXTENSIONS = {'.fits', '.fit', '.fts', '.nef', '.cr2', '.cr3', '.arw', '.raw', '.dng'}
# Denoise (DA3D) is only ever applied to very short broadband mono stacks,
# where per-frame noise dominates and the stack itself is too shallow to
# average it out. Below this many frames → denoise is considered; at or
# above it, normal stacking noise averaging is trusted instead.
DENOISE_SHORT_STACK_THRESHOLD = 5
USE_PCC = False
# Registration (Siril 'register') rarely produces perfectly-overlapping frames:
# each sub can be shifted/rotated by a fraction of a pixel to several pixels
# relative to the reference, so the stacked master has a thin border where not
# every frame contributed data (visible as a black, noisy, or mis-colored
# fringe — e.g. a stray hue line along one edge). Cropping a small percentage
# off each side removes this without perceptibly cutting into the frame.
# Set to 0.0 to disable.
BORDER_CROP_PERCENT = 1.5

# --- Star halo reduction ---
# Bright, oversaturated stars (e.g. 52 Cygni next to NGC 6960) can bloom
# large enough to eat into nearby faint nebula signal. No StarNet++
# dependency assumed to be available on this machine, so this uses a
# mask+erode technique instead of full star removal/synthesis:
# a bright-pixel mask isolates stars, a min-filtered (eroded) copy of the
# image is blended back in ONLY within that mask — small round bright
# blobs (stars) shrink noticeably under a min filter, large diffuse
# structures (nebula) barely change.
# UNVERIFIED ON REAL DATA — this is a first pass. Tune or disable via the
# constants below after checking a real render; see reduce_star_halos()
# docstring for what each one controls.
ENABLE_STAR_HALO_REDUCTION = True

# --- Flat staleness check ---
# A master flat captured long before/after the light session may no longer
# match the current optical train (back-focus drift, dust motes, filter or
# corrector reinstalled) — found in practice on NGC 4565: corners brighter
# than the frame center (the OPPOSITE of normal optical vignetting, which
# darkens corners), traced to a master flat dated 2022-10-18 used against a
# much more recent light session. This only warns, never blocks — the flat
# might still be fine (some setups are mechanically stable for years).
FLAT_STALENESS_WARNING_DAYS = 90
STAR_HALO_MASK_THRESHOLD = 45   # percent; was 88 — measured on a real render (52 Cygni, NGC 6960) that
                                 # 88% only covers the saturated core (a few dozen px, perceptually invisible
                                 # at full-frame scale): near-white pixel count dropped 60-83%, but the
                                 # visually offending glow at 50-150/255 luma only dropped 2-5%. Lowering the
                                 # threshold brings that mid-brightness glow into the mask so the erosion can
                                 # actually act on it. Re-verify on the next render; if the glow radius still
                                 # barely shrinks, go lower still (e.g. 30) before touching ERODE_RADIUS/BLEND.
STAR_HALO_MASK_FEATHER    = 3   # px blur radius on the mask edge, avoids a visible ring around each treated star
STAR_HALO_ERODE_RADIUS    = 2   # px; size of the min-filter kernel — larger = more shrinkage but more risk of clipping small stars to nothing
STAR_HALO_BLEND_PERCENT   = 45  # 0-100; how much of the eroded copy to mix back in within the mask — lower = gentler

# --- Background depth (autostretch targetbg) tuning ---
# Lower targetbg = darker sky background = more contrast on faint structure,
# at the risk of clipping the faintest signal if the stack's SNR doesn't
# support it. Centralized here (used throughout get_stretch_command) so they
# can be tuned in one place instead of hunting through the function.
# Starting values below are nudged darker than the previous hardcoded ones
# (NB good 0.15→0.12, medium 0.20→0.17, short 0.28→0.26) — re-tune against a
# real test render, these are a starting point, not a measured optimum.
STACK_GOOD_MIN_FRAMES   = 20   # frames (+ masters) required to trust a dark background
STACK_MEDIUM_MIN_FRAMES = 8    # frames (+ masters) required for a moderate background
TARGETBG_NB_GOOD    = 0.12     # narrowband, well-calibrated deep stack
TARGETBG_NB_MEDIUM  = 0.17     # narrowband, medium stack
TARGETBG_NB_SHORT    = 0.26     # narrowband, short stack / no calibration masters
TARGETBG_RGB_GOOD   = 0.18     # mono RGB channel, well-calibrated deep stack (was flat 0.25)
TARGETBG_RGB_MEDIUM = 0.22     # mono RGB channel, medium stack
TARGETBG_RGB_SHORT  = 0.25     # mono RGB channel, short stack / no calibration masters (old default)
TARGETBG_MONO_BB_GOOD   = 0.20     # mono broadband (L/LUMINANCE/CLEAR), well-calibrated deep stack
TARGETBG_MONO_BB_MEDIUM = 0.22     # mono broadband, medium stack
TARGETBG_MONO_BB_SHORT  = 0.25     # mono broadband, short stack / no calibration masters
TARGETBG_OSC_BB_GOOD    = 0.17     # OSC broadband (CLEAR/no filter/UHC/CLS), well-calibrated deep stack (was 0.20)
TARGETBG_OSC_BB_MEDIUM  = 0.20     # OSC broadband, medium stack (was 0.23)
TARGETBG_OSC_BB_SHORT   = 0.25     # OSC broadband, short stack / no calibration masters (was 0.28)

# --- Narrowband denoise (background noise) ---
# Previously narrowband was categorically excluded from denoise to protect
# faint filaments. DA3D at LOW mod is signal-preserving enough to knock down
# background shot noise without smearing filament structure — the risk was
# specifically from using broadband-strength mod (0.7-0.85) on narrowband.
# Keep this ceiling well below the broadband mod range; if filaments start
# looking soft/waxy on a real render, lower NARROWBAND_DENOISE_MOD further or
# set ENABLE_NARROWBAND_DENOISE = False to fully revert to the old behavior.
ENABLE_NARROWBAND_DENOISE = True
NARROWBAND_DENOISE_MOD = 0.2
SIRIL_VERSION_MAJOR = 1
SIRIL_VERSION_MINOR = 2 # Set to 4 when upgrading to enable cc -nostellar


def debug(message: str):
    if VERBOSE:
        print(f"[DEBUG] {message}", flush=True)

def emit(status: str, step: str, params: dict = None):
    """Emit a JSON message to stderr for IPC with Symfony API."""
    time = datetime.now().strftime('%H:%M:%S')
    payload = {"status": status, "step": step, "time": time}

    if params: payload["params"] = params
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

# --------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# --------------------------------------------------------------------------
def get_fits_header(fits_path: Path) -> dict:
    """Cache FITS header reads to avoid redundant file operations."""
    try:
        with fits.open(fits_path, mode='readonly', ignore_missing_end=True) as hdul:
            return dict(hdul[0].header)
    except Exception as e:
        debug(f"Failed to read FITS header from {fits_path}: {e}")
        return {}

def is_color_camera(fits_path: Path) -> bool:
    """Analyze FITS header to determine if sensor is color (OSC)."""
    emit("progress", "is_color_camera_started", params={"file": fits_path.name})

    if fits_path.suffix.lower() in {'.nef', '.cr2', '.cr3', '.arw', '.raw', '.dng'}:
        debug(f"RAW file detected ({fits_path.suffix}) → color camera assumed")
        emit("progress", "is_color_camera_done", params={"result": True, "reason": "raw_extension"})
        return True

    header = get_fits_header(fits_path)
    if not header:
        debug(f"⚠️ No FITS headers, default MONO")
        emit("progress", "is_color_camera_done", params={"result": False, "reason": "no_header"})
        return False

    debug(f"=== Sensor Detection ===")
    debug(f"  INSTRUME: {header.get('INSTRUME', 'N/A')}")
    debug(f"  BAYERPAT: {header.get('BAYERPAT', 'N/A')}")
    debug(f"  XBAYROFF: {header.get('XBAYROFF', 'N/A')}")
    debug(f"  CFAHEADER: {header.get('CFAHEADER', 'N/A')}")
    debug(f"  COLOR: {header.get('COLOR', 'N/A')}")

    if 'BAYERPAT' in header or 'XBAYROFF' in header or 'CFAHEADER' in header:
        debug(f"  → Result: COLOR (Bayer detected)")
        emit("progress", "is_color_camera_done", params={"result": True, "reason": "bayer_pattern"})
        return True

    instrument = header.get('INSTRUME', '').upper()
    if 'MC' in instrument and 'MM' not in instrument:
        debug(f"  → Result: COLOR (instrument MC)")
        emit("progress", "is_color_camera_done", params={"result": True, "reason": "instrument_mc"})
        return True

    if header.get('COLOR', '') == 'YES':
        debug(f"  → Result: COLOR (keyword COLOR=YES)")
        emit("progress", "is_color_camera_done", params={"result": True, "reason": "color_keyword"})
        return True

    emit("progress", "is_color_camera_done", params={"result": False, "reason": "no_color_indicator"})
    return False

def get_fits_bitdepth(fits_path: Path) -> int:
    """
    Detects bit depth via BITPIX from FITS header.
    BITPIX=16 → 16-bit signed integer
    BITPIX=-32 → float32, BITPIX=-64 → float64 → 32-bit
    """
    emit("progress", "get_fits_bitdepth_started", params={"file": fits_path.name})

    if fits_path.suffix.lower() in {'.nef', '.cr2', '.cr3', '.arw', '.raw', '.dng'}:
        debug(f"RAW file → assuming 16-bit output after Siril conversion")
        emit("progress", "get_fits_bitdepth_done", params={"bitdepth": 16, "reason": "raw_file"})
        return 16

    try:
        with fits.open(fits_path, mode='readonly', ignore_missing_end=True) as hdul:
            bitpix   = hdul[0].header.get('BITPIX', -32)
            detected = 16 if bitpix == 16 else 32
            debug(f"BITPIX={bitpix} → {detected}-bit")
            emit("progress", "get_fits_bitdepth_done", params={"bitdepth": detected, "bitpix": bitpix})
            return detected
    except Exception as e:
        debug(f"Unable to read bit depth from {fits_path}: {e}")
        emit("progress", "get_fits_bitdepth_done", params={"bitdepth": 32, "reason": "read_error"})
        return 32

# --------------------------------------------------------------------------
# DOF (DARKS, FLATS, BIAS)
# --------------------------------------------------------------------------
def ensure_2d_master(master_path: Path) -> Path | None:
    """Ensure the master is in the correct 2D FITS geometric format for Siril CLI."""
    emit("progress",  "ensure_2d_master_started", params={"file": master_path.name})

    if not master_path.exists():
        emit("progress", "ensure_2d_master_done",params={"result": None, "reason": "file_not_found"})
        return None

    if master_path.suffix.lower() not in ('.fit', '.fits', '.fts'):
        debug(f"Non-FITS master, skipping 2D check: {master_path.name}")
        emit("progress", "ensure_2d_master_done", params={"result": str(master_path.name), "reason": "non_fits"})
        return master_path

    output_path = master_path.parent / f"{master_path.stem}_2d.fit"

    if output_path.exists():
        try:
            output_path.unlink()
        except Exception as e:
            debug(f"Unable to delete {output_path} : {e}")
            emit("progress", "ensure_2d_master_done", params={"result": None, "reason": "delete_failed"})
            return None

    try:
        with fits.open(master_path) as hdul:
            header        = hdul[0].header.copy()
            data          = hdul[0].data
            if data is None:
                debug(f"Empty data in {master_path.name}, skipping 2D normalization")
                emit("progress", "ensure_2d_master_done", params={"result": None, "reason": "empty_data"})
                return None
            original_dtype = data.dtype

        # If image was accidentally read or saved as RGB (3D)
        if data.ndim == 3:
            data = np.mean(data, axis=-1).astype(original_dtype)  # Clean merge to pure intensity
            header.add_comment('Master normalized to 2D structure')
        elif data.ndim != 2:
            debug(f"Invalid image structure for calibration: {data.ndim} dimensions")
            emit("progress", "ensure_2d_master_done", params={"result": None, "reason": f"invalid_ndim_{data.ndim}"})
            return None

        header['NAXIS'] = 2
        if 'NAXIS3' in header:
            del header['NAXIS3']

        fits.writeto(output_path, data, header, overwrite=True)
        emit("progress", "ensure_2d_master_done", params={"result": output_path.name})
        return output_path
    except Exception as e:
        debug(f"Failed to normalize FITS 2D for {master_path.name} : {e}")
        emit("progress", "ensure_2d_master_done", params={"result": None, "reason": f"exception: {e}"})
        return None

def _find_master_dof(
    dof_dir: Path,
    generic_name: str,
    filter_name: str | None = None
) -> str | None:
    """
    Generic DOF master finder (dark, flat, bias).
    Priority order:
      1. Filter-specific master (e.g. masterDark_FILTER-HA.fit)
      2. Generic master variants (master_dark, masterDark, MasterDark...)
      3. On-the-fly stacking of individual frames if no master found
    Excludes _2d files (already processed intermediates).
    Matches case variants: HA / Ha / ha / Ha (capitalize).
    Returns the most recently modified match.
    """
    emit("progress", "find_master_dof_started", params={
        "type":   generic_name,
        "filter": filter_name or "generic",
        "dir":    dof_dir.name,
    })

    if not dof_dir.is_dir():
        emit("progress", "find_master_dof_done",params={"type": generic_name, "found": False})
        return None

    def resolve_master(path: Path) -> str:
        m2d = ensure_2d_master(path)
        if m2d:
            return str(m2d.resolve())
        debug(f"ensure_2d_master failed for {path.name}, using original")
        return str(path.resolve())

    def find_by_pattern(pattern: str) -> Path | None:
        matches = sorted(
            (
                p for p in dof_dir.glob(pattern)
                if not p.stem.endswith("_2d")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return matches[0] if matches else None

    # 1. Filter-specific (4 case variants)
    if filter_name:
        variants = {filter_name, filter_name.upper(), filter_name.lower(), filter_name.capitalize()}
        for variant in variants:
            for ext in VALID_EXTENSIONS:
                match = find_by_pattern(f"*FILTER-{variant}*{ext}")
                if match:
                    debug(f"Filter-specific {generic_name} found: {match.name}")
                    emit("progress", "find_master_dof_done", params={"type": generic_name, "found": True, "file": match.name, "method": "filter_specific"})
                    return resolve_master(match)

    # 2. Generic fallback : case variants of generic_name
    # Supports: master_dark, master-dark, masterDark, MasterDark, masterdark...
    base = generic_name.replace("master_", "")  # "dark", "flat", "bias"
    generic_variants = [
        f"master_{base}",               # master_dark   (snake_case)
        f"master-{base}",               # master-dark   (kebab-case)
        f"Master_{base}",               # Master_dark
        f"master{base}",                # masterdark    (lowercase)
        f"master{base.capitalize()}",   # masterDark    (camelCase)
        f"Master{base.capitalize()}",   # MasterDark
    ]

    for ext in VALID_EXTENSIONS:
        for variant in generic_variants:
            # Exact match first
            exact = dof_dir / f"{variant}{ext}"
            if exact.exists() and not exact.stem.endswith("_2d"):
                debug(f"Generic {generic_name} found (exact): {exact.name}")
                emit("progress", "find_master_dof_done", params={"type": generic_name, "found": True, "file": exact.name, "method": "generic_exact"})
                return resolve_master(exact)

            # Glob with suffixes (e.g. masterDark_BIN-1_4656x3520_EXPOSURE-300.00s.fit)
            match = find_by_pattern(f"{variant}*{ext}")
            if match:
                debug(f"Generic {generic_name} found (glob): {match.name}")
                emit("progress", "find_master_dof_done", params={"type": generic_name, "found": True, "file": match.name, "method": "generic_glob"})
                return resolve_master(match)

    # 3. Fallback : individual frames → stack on-the-fly
    # Triggered when no pre-built master found but raw frames exist in the directory
    raw_frames = sorted([
        f for f in dof_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
        and not f.stem.endswith("_2d")
        and not f.name.lower().startswith("master")
    ])

    if not raw_frames:
        debug(f"No {generic_name} found in {dof_dir}")
        emit("progress", "find_master_dof_done", params={"type": generic_name, "found": False})
        return None

    dof_type = generic_name.replace("master_", "")   # "dark" | "flat" | "bias"
    debug(f"No pre-built {generic_name} — stacking {len(raw_frames)} individual frames on-the-fly")
    emit("progress", "find_master_dof_onthefly_started", params={
        
        "type":   dof_type,
        "frames": len(raw_frames),
    })

    # Create a temp work subdir to isolate symlinks from originals
    work_subdir = dof_dir / f"work_{dof_type}"
    work_subdir.mkdir(exist_ok=True)

    for i, frame in enumerate(raw_frames, start=1):
        dst = work_subdir / f"{dof_type}_{i:05d}{frame.suffix.lower()}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(frame.resolve())
        except OSError:
            shutil.copy(frame, dst)

    norm = {"dark": "-nonorm", "bias": "-nonorm", "flat": "-norm=mul"}.get(dof_type, "-nonorm")
    master_output = dof_dir / f"master_{dof_type}.fit"

    script = "\n".join([
        "requires 1.2.0",
        f'cd "{work_subdir.resolve().as_posix()}"',
        f'convert {dof_type}',
        f'stack {dof_type} rej winsorized 3 3 {norm} -out=../master_{dof_type}',
        "close",
        "exit"
    ])

    # run_siril_command needs a session_dir for the temp .ssf file
    session_dir = dof_dir.parent
    success = run_siril_command(
        session_dir,
        script,
        f"dof_{dof_type}.ssf",
        work_dir=work_subdir
    )

    if success and master_output.exists():
        debug(f"✅ master_{dof_type}.fit generated ({len(raw_frames)} frames stacked)")
        # Cleanup work subdir
        shutil.rmtree(work_subdir, ignore_errors=True)
        emit("progress", "find_master_dof_onthefly_done", params={"type": dof_type, "success": True})
        emit("progress", "find_master_dof_done", params={"type": generic_name, "found": True, "file": master_output.name, "method": "on_the_fly"})
        return resolve_master(master_output)

    debug(f"⚠️ On-the-fly {dof_type} stacking failed — proceeding without master")
    emit("progress", "find_master_dof_onthefly_done", params={"type": dof_type, "success": False})
    emit("progress", "find_master_dof_done", params={"type": generic_name, "found": False})
    return None

def get_master_dark_path(session_dir: Path, filter_name: str | None = None) -> str | None:
    """Look for a master dark. Filter-specific first, then generic master_dark."""
    return _find_master_dof(session_dir / "darks", "master_dark", filter_name)


def get_master_flat_path(session_dir: Path, filter_name: str | None = None) -> str | None:
    """Look for a master flat. Filter-specific first, then generic master_flat."""
    return _find_master_dof(session_dir / "flats", "master_flat", filter_name)


def get_master_bias_path(session_dir: Path, filter_name: str | None = None) -> str | None:
    """Look for a master bias. Filter-specific first, then generic master_bias.
    Note: bias are usually filter-independent, but filter-specific ones are supported."""
    return _find_master_dof(session_dir / "bias", "master_bias", filter_name)

# --------------------------------------------------------------------------
# SUBSKY - Gradient Optimization
# --------------------------------------------------------------------------
def get_frame_dimensions(light_files: list) -> tuple | None:
    """
    Read (NAXIS1, NAXIS2) — i.e. (width, height) — from the first light
    frame's FITS header. Returns None if it can't be determined (missing
    files, unreadable header, missing NAXIS keywords) so callers can
    degrade gracefully instead of crashing the whole stack script.
    """
    if not light_files:
        return None
    try:
        with fits.open(light_files[0], ignore_missing_end=True) as hdul:
            header = hdul[0].header
            width, height = header.get('NAXIS1'), header.get('NAXIS2')
            if width and height:
                return int(width), int(height)
            debug(f"Frame dimensions: NAXIS1/NAXIS2 missing in {light_files[0]}")
            return None
    except Exception as e:
        debug(f"Frame dimensions: could not read {light_files[0]} ({e})")
        return None


def get_early_crop_command(light_files: list, percent: float) -> str:
    """
    Returns a Siril 'crop x y width height' command sized to shave
    BORDER_CROP_PERCENT off each side of the stacked master, computed from
    the light frames' own resolution (read via FITS header — Siril script
    syntax has no variables/arithmetic, so this can't be computed inside
    the .ssf itself).

    Placed right after 'load r_{seq}_stacked.fit', BEFORE subsky, color
    calibration, denoise, and stretch — moved here (previously this crop
    only happened much later, in Python, on the final ImageMagick-composed
    image) after re-reading Siril's own scripting tutorial: cropping should
    happen before background/gradient extraction, since a registration
    border fringe still present in the sampled background contaminates the
    subsky model. This was a real, previously-unexamined gap: the post-stack
    subsky path (get_subsky_command) ran directly on 'r_{seq}_stacked.fit',
    which can still carry a partial-coverage edge from register/stack, before
    any crop ever touched it.

    Cropping once here, per channel, before recombination also closes a
    second, separately-flagged risk: R/G/B or narrowband channels can each
    drift by a slightly different amount during their own registration, so
    a single crop applied post-composition (the old approach) could leave
    an asymmetric residual fringe if one channel's misalignment exceeded the
    crop amount. Cropping each channel individually, at its own native
    resolution, avoids that.

    Returns "" if percent <= 0 or if the frame dimensions can't be read
    (crop is skipped rather than guessed — an unverified crop box risks
    cutting into real signal more than leaving a thin border alone would).
    """
    if percent <= 0:
        return ""

    dims = get_frame_dimensions(light_files)
    if dims is None:
        debug("Early crop: frame dimensions unavailable, skipping crop")
        return ""

    width, height = dims
    shave_x = max(1, round(width * percent / 100))
    shave_y = max(1, round(height * percent / 100))
    crop_w = width - 2 * shave_x
    crop_h = height - 2 * shave_y
    if crop_w <= 0 or crop_h <= 0:
        debug(f"Early crop: percent={percent} too large for {width}x{height}, skipping")
        return ""

    cmd = f"crop {shave_x} {shave_y} {crop_w} {crop_h}"
    emit("progress", "get_early_crop_command_done", params={
        "command": cmd, "source_dims": f"{width}x{height}", "percent": percent
    })
    return cmd


def check_flat_staleness(master_flat_path: str, light_files: list) -> None:
    """
    Warn (never block) if the master flat's DATE-OBS is far from the light
    frames' DATE-OBS. Doesn't fix anything by itself — it's a diagnostic
    signal for the person running the pipeline to go re-shoot a flat rather
    than spend time chasing a phantom subsky/code bug.

    Root case that motivated this: NGC 4565 final composite showed all four
    corners brighter than the frame center — the OPPOSITE of normal optical
    vignetting (which darkens corners). The master flat in use was dated
    2022-10-18, likely long before the actual light session. A flat that no
    longer matches the current optical train (back-focus drift since then,
    dust motes, filter or corrector removed/reinstalled) will under- or
    over-correct vignetting in a way that can look exactly like this —
    brighter corners if the flat's recorded vignetting is now weaker than
    the light frames' actual vignetting, so dividing by it doesn't fully
    flatten the field.
    """
    if not master_flat_path or not light_files:
        return
    try:
        with fits.open(master_flat_path, ignore_missing_end=True) as hdul:
            flat_date_str = hdul[0].header.get('DATE-OBS')
        with fits.open(light_files[0], ignore_missing_end=True) as hdul:
            light_date_str = hdul[0].header.get('DATE-OBS')
    except Exception as e:
        debug(f"Flat staleness check: could not read DATE-OBS ({e}), skipping")
        return

    if not flat_date_str or not light_date_str:
        debug("Flat staleness check: DATE-OBS missing on flat or light frame, skipping")
        return

    try:
        flat_date = datetime.fromisoformat(flat_date_str.replace('Z', ''))
        light_date = datetime.fromisoformat(light_date_str.replace('Z', ''))
    except ValueError as e:
        debug(f"Flat staleness check: could not parse DATE-OBS ({e}), skipping")
        return

    age_days = abs((light_date - flat_date).days)
    if age_days > FLAT_STALENESS_WARNING_DAYS:
        debug(
            f"WARNING: master flat is {age_days} days from the light session "
            f"(flat: {flat_date_str}, lights: {light_date_str}). A gap this "
            f"large means the flat may no longer represent the current "
            f"optical train — consider re-shooting it if vignetting/gradient "
            f"artifacts show up in the final composite."
        )
        emit("progress", "flat_staleness_warning", params={
            "flat_path": str(master_flat_path),
            "flat_date": flat_date_str,
            "light_date": light_date_str,
            "age_days": age_days,
        })


def get_subsky_command(
    filter_name: str,
    num_files: int,
    is_narrowband: bool,
    master_dark_path: str = None,
    master_flat_path: str = None,
    master_bias_path: str = None,
    force_skip: bool = False,
) -> str:
    """
    Returns a Siril 1.2 subsky command calibrated to the imaging context.

    SubSky risks on deep or low-surface-brightness data:
      - Deep narrowband (SII, faint OIII): signal fills the frame → background
        samples are contaminated → RBF creates a negative bowl around the nebula.
      - IFN (Integrated Flux Nebulae): the "background" is the signal itself —
        subsky will destroy the very data we are trying to preserve.
      - Flat-corrected data: the flat already corrects optical vignetting;
        aggressive subsky then over-subtracts and creates halos.

    Safe rule: if a flat is present AND the stack is deep narrowband,
    skip subsky entirely (flat handles vignetting, stack handles noise).
    For all other cases, apply RBF/polynomial adapted to available calibration.
    """
    emit("progress", "get_subsky_command_started", params={
        
        "filter": filter_name,
        "frames": num_files,
    })

    if force_skip:
        debug("Subsky: skipped (force_skip=True)")
        emit("progress", "get_subsky_command_done", params={"filter": filter_name, "command": "", "reason": "force_skip"})
        return ""

    missing = []
    if not master_dark_path: missing.append("dark")
    if not master_flat_path: missing.append("flat")
    if not master_bias_path: missing.append("bias")

    debug(f"=== Subsky Masters Check ===")
    debug(f"  Filter: {filter_name} | NB: {is_narrowband} | Frames: {num_files}")
    debug(f"  Master Dark:  {'SET' if master_dark_path else 'MISSING'}")
    debug(f"  Master Flat:  {'SET' if master_flat_path else 'MISSING'}")
    debug(f"  Master Bias:  {'SET' if master_bias_path else 'MISSING'}")
    debug(f"  Missing: {missing}")

    has_flat   = "flat" not in missing
    nb_missing = len(missing)

    # ------------------------------------------------------------------
    # CASE 1: Flat available
    # The flat already corrects optical vignetting. SubSky should only
    # handle residual sky gradient (light pollution, atmospheric gradients).
    # ------------------------------------------------------------------
    if has_flat:
        # Deep narrowband with flat: flat corrects everything optical.
        # Subsky on a filled-frame SII/OIII risks clipping the outer signal.
        if is_narrowband and num_files >= 30:
            debug("Subsky: disabled — deep narrowband with flat (flat handles vignetting)")
            emit("progress", "get_subsky_command_done", params={"filter": filter_name, "command": "", "reason": "deep_nb_with_flat"})
            return ""

        # Flat available, not deep narrowband: gentle RBF to handle sky gradient.
        # High smooth prevents over-fitting around bright objects.
        # samples raised 50->80: a corner-brightening artifact seen on NGC 4565
        # is more consistent with the RBF undershooting the true background
        # near corners (sparse sample coverage there) than with the flat
        # itself (which measured with normal, correctly-shaped vignetting).
        # More sample points should reduce that risk, but this is a
        # reasoned adjustment, not a verified fix — the flat's age (see
        # check_flat_staleness) is at least as likely a contributor and
        # should be ruled out first on the next real test.
        tolerance = 1.2 if nb_missing == 0 else 1.4
        smooth    = 0.6 if nb_missing == 0 else 0.7
        cmd = f'subsky -rbf -tolerance={tolerance} -smooth={smooth} -samples=80'
        emit("progress", "get_subsky_command_done", params={"filter": filter_name, "command": cmd})
        return cmd

    # ------------------------------------------------------------------
    # CASE 2: No flat
    # Without flat, optical vignetting persists after calibration.
    # SubSky must correct both vignetting and sky gradient.
    # Use RBF (better for radial vignetting) with smooth tuned to stack depth.
    # ------------------------------------------------------------------
    if nb_missing == 1:
        # Dark only — decent calibration, moderate correction
        cmd = 'subsky 3 -tolerance=1.4 -samples=60'
    elif nb_missing == 2:
        # Missing flat + one other: RBF to handle radial patterns
        cmd = 'subsky -rbf -tolerance=1.5 -smooth=0.7 -samples=60'
    else:
        # No DOF at all: high smooth to prevent dark halo at center
        cmd = 'subsky -rbf -tolerance=1.3 -smooth=0.9 -samples=70'

    emit("progress",  "get_subsky_command_done", params={"filter": filter_name, "command": cmd})
    return cmd


def get_color_calibration_command(
    is_color: bool,
    filter_name: str,
    light_files: list,
) -> str:
    """
    Returns a Siril color calibration command for OSC broadband data only.

    Siril 1.2: only `pcc` is scriptable, requires NOMAD catalog (network or local).
    Siril 1.4: `cc -nostellar` is available without network or catalog.

    NOTE on -noflip: Siril's own scripting tutorial recommends leaving PCC's
    auto-flip ("Retourner l'image si nécessaire") ENABLED, since it uses
    astrometry (a more reliable source of truth than a fixed heuristic).
    -noflip is used here deliberately, NOT an oversight: this pipeline
    already applies an unconditional flip downstream, in
    correct_image_orientation() (see that function's docstring — Siril
    always stores/processes bottom-up regardless of ROWORDER). If PCC also
    flipped, the two flips would cancel out and undo the correction. These
    two are coupled: don't change one without checking the other.

    Never apply on:
      - Mono camera (no color channels to balance)
      - Narrowband filters (monochromatic signal — calibration corrupts ratios)
      - L/LUMINANCE on OSC (same as CLEAR — excluded by is_color=True check
        being sufficient; the explicit LUMINANCE exclusion was incorrect and
        prevented calibration for OSC+L sessions)
    """
    emit("progress", "get_color_calibration_command_started", params={
        
        "filter":   filter_name,
        "is_color": is_color,
    })

    if not is_color:
        emit("progress", "get_color_calibration_command_done", params={"command": "", "reason": "mono_camera"})
        return ""

    if filter_name.upper() in NARROWBAND_FILTERS:
        emit("progress", "get_color_calibration_command_done", params={"command": "", "reason": "narrowband_filter"})
        return ""

    # Siril 1.4+: cc -nostellar works without network or catalog
    if SIRIL_VERSION_MAJOR >= 1 and SIRIL_VERSION_MINOR >= 4:
        emit("progress", "get_color_calibration_command_done", params={"command": "cc -nostellar", "reason": "siril_1.4"})
        return "cc -nostellar"

    # Siril 1.2: pcc requires NOMAD catalog
    if USE_PCC and light_files:
        try:
            with fits.open(
                light_files[0],
                mode='readonly',
                ignore_missing_end=True
            ) as hdul:
                h          = hdul[0].header
                focal      = h.get('FOCALLEN')
                pixel_size = h.get('XPIXSZ') or h.get('PIXSIZE')

                if focal and pixel_size:
                    debug(f"PCC: focal={int(focal)}mm pixel={float(pixel_size)}µm")
                    cmd = (
                        f"pcc -focal={int(focal)}"
                        f" -pixelsize={float(pixel_size)}"
                        f" -noflip -downscale"
                    )
                    emit("progress", "get_color_calibration_command_done", params={"command": cmd, "reason": "pcc_with_metadata"})
                    return cmd
        except Exception as e:
            debug(f"PCC metadata read failed ({e}), skipping color calibration")

        emit("progress", "get_color_calibration_command_done", params={"command": "pcc -noflip -downscale", "reason": "pcc_wcs_fallback"})
        return "pcc -noflip -downscale"  # fallback: rely on existing WCS

    # Siril 1.2 without USE_PCC: no scriptable color calibration available
    debug("Color calibration: skipped (Siril 1.2, USE_PCC=False)")
    emit("progress", "get_color_calibration_command_done", params={"command": "", "reason": "siril_1.2_no_pcc"})
    return ""


# --------------------------------------------------------------------------
# SEQSUBSKY — per-frame background extraction before stacking
# --------------------------------------------------------------------------
# Configuration constants (add near top of file with other constants)
SEQSUBSKY_ENABLED      = True   # Set to False to revert to post-stack subsky only
SEQSUBSKY_SAMPLES_NB   = 30     # Fewer samples: narrowband frames are noisy
SEQSUBSKY_SAMPLES_BB   = 20     # Broadband frames: background is more uniform
SEQSUBSKY_TOLERANCE_NB = 2.0    # High tolerance: signal fills the frame on deep NB
SEQSUBSKY_TOLERANCE_BB = 1.5    # Standard broadband tolerance
SEQSUBSKY_SMOOTH       = 0.6    # Smooth RBF to avoid over-subtracting noise patterns


def get_seqsubsky_command(
    seq_name: str,
    filter_name: str,
    is_narrowband: bool,
    num_files: int,
    master_flat_path: str = None,
) -> tuple[str, str]:
    """
    Returns (siril_command, output_seq_name) for per-frame background extraction.

    seqsubsky creates a new sequence prefixed with 'bkg_':
        pp_light    → bkg_pp_light
        pp_pp_light → bkg_pp_pp_light

    Operating on individual frames (vs post-stack subsky) requires adjusted
    parameters: individual frames are noisier, so sample counts are lower
    and tolerances are higher than on the stacked image.

    Narrowband: signal fills the frame → very high tolerance to prevent
    the RBF model from fitting to the nebula instead of the background.
    Broadband: reliable background samples available in corners.

    With flat: the flat already corrects optical vignetting → tighter
    tolerance is safe. Without flat: tolerance must be higher to handle
    the combined vignetting + sky gradient without over-subtracting.
    """

    emit("progress", "get_seqsubsky_command_started", params={
        
        "filter":      filter_name,
        "sequence":    seq_name,
        "narrowband":  is_narrowband,
        "frames":      num_files,
    })

    if not SEQSUBSKY_ENABLED:
        emit("progress", "get_seqsubsky_command_done", params={"command": "", "output_seq": seq_name, "reason": "disabled"})
        return "", seq_name

    has_flat = bool(master_flat_path)

    if is_narrowband:
        # Deep narrowband: high tolerance protects signal-rich regions.
        # Extra smooth prevents the RBF from fitting extended emission.
        tolerance = SEQSUBSKY_TOLERANCE_NB
        samples   = SEQSUBSKY_SAMPLES_NB
        smooth    = max(SEQSUBSKY_SMOOTH, 0.75)
    else:
        # Broadband: standard parameters, slightly tighter with flat available
        tolerance = SEQSUBSKY_TOLERANCE_BB if not has_flat else 1.2
        samples   = SEQSUBSKY_SAMPLES_BB
        smooth    = SEQSUBSKY_SMOOTH

    cmd = (
        f"seqsubsky {seq_name}"
        f" -rbf"
        f" -tolerance={tolerance}"
        f" -smooth={smooth}"
        f" -samples={samples}"
    )
    output_seq = f"bkg_{seq_name}"

    debug(
        f"seqsubsky [{filter_name}]: {seq_name} → {output_seq} "
        f"(tolerance={tolerance}, smooth={smooth}, samples={samples})"
    )
    emit("progress", "get_seqsubsky_command_done", params={
        
        "filter":     filter_name,
        "command":    cmd,
        "output_seq": output_seq,
        "tolerance":  tolerance,
        "smooth":     smooth,
        "samples":    samples,
    })
    return cmd, output_seq


def should_apply_denoise(
    filter_name: str,
    num_files: int,
    is_color: bool,
    all_detected_filters: list[str],
    master_dark_path: str = None,
    master_flat_path: str = None,
    master_bias_path: str = None,
) -> tuple[bool, float]:
    """
    Automatically determines if DA3D denoise should be applied and with what mod.
    Returns (apply: bool, mod: float).

    Exclusion rules (priority order):
    1. Color OSC camera → never (inter-channel imbalance guaranteed)
    2. Broadband multi-filter palette (e.g. RGB) → never — each channel can
       have a different frame count, so the broadband activation rule below
       would pick a different mod per channel, risking a color cast.
    3. Broadband, single filter → only on very short stacks (frame-count
       dependent mod, see below).

    Narrowband gets a fixed, gentle mod (NARROWBAND_DENOISE_MOD) whenever
    ENABLE_NARROWBAND_DENOISE is set, regardless of how many narrowband
    filters are active in the session — since the mod is a fixed constant
    (not derived from frame count), every narrowband channel gets exactly
    the same treatment, so the "different mod per filter" color-cast risk
    that excludes broadband multi-filter palettes doesn't apply here.
    """

    # --- PRIORITY EXCLUSIONS ---

    emit("progress", "should_apply_denoise_started", params={
        
        "filter": filter_name,
        "frames": num_files,
    })

    # Color OSC: never
    if is_color:
        debug(f"Denoise [{filter_name}]: OFF — color OSC")
        emit("progress", "should_apply_denoise_done", params={"filter": filter_name, "apply": False, "reason": "color_osc"})
        return False, 0.0

    # Narrowband (single or multi-filter): gentle, fixed-mod denoise instead
    # of a blanket exclusion. DA3D at broadband-strength mod (0.7-0.85) would
    # smooth out faint filaments, but at a low, fixed mod it mainly knocks
    # down background shot noise. Fixed mod means every narrowband filter in
    # the session gets identical treatment — no differential-mod color cast.
    # Toggle off via ENABLE_NARROWBAND_DENOISE if filaments start looking
    # soft on a real render.
    is_narrowband = filter_name.upper() in NARROWBAND_FILTERS
    if is_narrowband:
        if ENABLE_NARROWBAND_DENOISE:
            debug(f"Denoise [{filter_name}]: ON — narrowband, gentle mod={NARROWBAND_DENOISE_MOD}")
            emit("progress", "should_apply_denoise_done", params={"filter": filter_name, "apply": True, "mod": NARROWBAND_DENOISE_MOD, "reason": "narrowband_gentle"})
            return True, NARROWBAND_DENOISE_MOD
        debug(f"Denoise [{filter_name}]: OFF — narrowband (ENABLE_NARROWBAND_DENOISE=False)")
        emit("progress", "should_apply_denoise_done", params={"filter": filter_name, "apply": False, "reason": "narrowband_disabled"})
        return False, 0.0

    # Broadband multi-filter palette (e.g. RGB): never — different frame
    # counts per channel would pick a different mod per channel below.
    active_narrowband = [f for f in all_detected_filters if f in NARROWBAND_FILTERS]
    active_broadband = [f for f in all_detected_filters if f not in NARROWBAND_FILTERS]
    if len(active_broadband) > 1:
        debug(f"Denoise [{filter_name}]: OFF — broadband multi-filter palette")
        emit("progress", "should_apply_denoise_done", params={"filter": filter_name, "apply": False, "reason": "multi_filter_palette"})
        return False, 0.0

    # Broadband mono: ONLY on very short stacks where noise dominates
    # Normal stacks (>= threshold frames) average noise naturally — DA3D not needed
    if num_files >= DENOISE_SHORT_STACK_THRESHOLD:
        debug(f"Denoise [{filter_name}]: OFF — broadband, enough frames to average noise")
        emit("progress", "should_apply_denoise_done", params={"filter": filter_name, "apply": False, "reason": "enough_frames"})
        return False, 0.0

    # Very short broadband stack (< threshold): light denoise
    has_dark  = bool(master_dark_path)
    has_flat  = bool(master_flat_path)
    has_bias  = bool(master_bias_path)
    dof_count = sum([has_dark, has_flat, has_bias])

    mod = 0.85 if dof_count == 0 else 0.7
    debug(f"Denoise [{filter_name}]: ON (very short broadband, frames={num_files}, mod={mod})")
    emit("progress", "should_apply_denoise_done", params={"filter": filter_name, "apply": True, "mod": mod})
    return True, mod


def get_rmgreen_command(is_color: bool) -> str:
    cmd = "rmgreen 0.3" if is_color else ""
    emit("progress", "get_rmgreen_command_done", params={"command": cmd})
    return cmd


def get_stretch_command(
    filter_name: str,
    is_color: bool,
    num_files: int,
    has_masters: bool
) -> list[str]:
    """
    Returns autostretch commands for Siril 1.2.
    ght and linstretch are Siril 1.4+ only — not used here.

    Core principle: targetbg scales inversely with signal quality.
    Better SNR (more frames + calibration masters) → lower targetbg
    (darker background, more dynamic range).
    Weaker SNR → higher targetbg to avoid clipping faint signal.

    shadowsclip controls how aggressively dark noise is clipped before stretch:
      -2.8 → standard, preserves faint diffuse nebulosity
      -3.0 → moderate, good balance on well-calibrated data
      -3.5 → aggressive, clips more noise but risks losing faint structures

    Parameter matrix (defaults — see TARGETBG_* / STACK_*_MIN_FRAMES constants):
    ┌─────────────────────────┬─────────────┬──────────┐
    │ Case                    │ shadowsclip │ targetbg │
    ├─────────────────────────┼─────────────┼──────────┤
    │ OSC broadband, many     │ -3.0        │ 0.17     │
    │ OSC broadband, medium   │ -2.8        │ 0.20     │
    │ OSC broadband, short    │ -2.8        │ 0.25     │
    │ OSC narrowband, good    │ -2.8        │ 0.12     │
    │ OSC narrowband, medium  │ -2.8        │ 0.17     │
    │ OSC narrowband, short   │ -2.8        │ 0.26     │
    │ Mono RGB channel        │ -2.8        │ 0.18-0.25│
    │ Mono broadband (CLEAR/L)│ -2.8/-3.0   │ 0.20-0.25│
    │ Mono NB, well calibrated│ -2.8        │ 0.12     │
    │ Mono NB, medium         │ -2.8        │ 0.17     │
    │ Mono NB, short/no DOF   │ -2.8        │ 0.26     │
    └─────────────────────────┴─────────────┴──────────┘
    """
    emit("progress", "get_stretch_command_started", params={
        
        "filter":      filter_name,
        "is_color":    is_color,
        "frames":      num_files,
        "has_masters": has_masters,
    })

    is_narrowband  = filter_name.upper() in NARROWBAND_FILTERS
    is_rgb_channel = filter_name.upper() in {"RED", "GREEN", "BLUE"}
    is_broadband   = filter_name.upper() in BROADBAND_FILTERS
    is_luminance   = filter_name.upper() in ("L", "LUMINANCE")

    # Signal quality score — drives targetbg selection
    good_stack  = num_files >= STACK_GOOD_MIN_FRAMES   and has_masters
    medium_stack = num_files >= STACK_MEDIUM_MIN_FRAMES and has_masters
    short_stack  = num_files < 6   or not has_masters

    def _done(cmds):
        emit("progress", "get_stretch_command_done", params={"filter": filter_name, "commands": cmds})
        return cmds

    # ------------------------------------------------------------------
    # OSC COLOR CAMERA
    # -linked is mandatory to preserve color balance.
    # shadowsclip is relaxed on short stacks to avoid clipping diffuse
    # nebulosity that would otherwise be indistinguishable from noise.
    # ------------------------------------------------------------------
    if is_color:
        if is_narrowband:
            # OSC + narrowband clip (e.g. dual-band filter on color camera).
            # Signal is narrow-band but sensor is color — treat like mono NB,
            # including scaling targetbg with stack quality (previously this
            # branch ignored num_files/has_masters entirely and always used
            # the "good" value, even on short/uncalibrated stacks).
            if good_stack:
                return _done([f"autostretch -linked -2.8 {TARGETBG_NB_GOOD}"])
            elif medium_stack:
                return _done([f"autostretch -linked -2.8 {TARGETBG_NB_MEDIUM}"])
            else:
                return _done([f"autostretch -linked -2.8 {TARGETBG_NB_SHORT}"])

        # OSC broadband (CLEAR, no filter, UHC, CLS...)
        # Previously 0.20/0.23/0.28, unchanged since before any of this
        # session's fixes — the RGB-channel and mono-CLEAR fixes only
        # touched the mono-camera branches above, not this OSC one, so an
        # OSC DSLR session (e.g. Nikon Z6II) shooting CLEAR never saw any
        # of that work. Nudged darker here too now, and fixed a missing
        # _done() call on the medium_stack case (progress event was silently
        # skipped, no effect on the actual stretch value).
        if good_stack:
            # Enough frames to trust SNR — aggressive clip, dark background
            return _done([f"autostretch -linked -3.0 {TARGETBG_OSC_BB_GOOD}"])
        elif medium_stack:
            # Moderate stack — balance between noise rejection and signal preservation
            return _done([f"autostretch -linked -2.8 {TARGETBG_OSC_BB_MEDIUM}"])
        else:
            # Short stack or no calibration — softer clip, brighter targetbg
            # Avoids clipping diffuse nebulosity (M16, M42, M31 outer halos)
            return _done([f"autostretch -linked -2.8 {TARGETBG_OSC_BB_SHORT}"])

    # ------------------------------------------------------------------
    # MONO CAMERA — RGB individual channels
    # Must use identical parameters across R/G/B to preserve color balance
    # when channels are later combined by compose_rgb_image. Previously this
    # was a flat 0.25 regardless of frame count, so a well-calibrated 75+
    # frame RGB session (e.g. 75/77/76 R/G/B) got the same conservative
    # background depth as a 5-frame one. Now scales with stack quality like
    # every other branch — R/G/B channels from the same session normally
    # have near-identical frame counts, so in practice they land in the same
    # bucket together; if a session ever splits across buckets (e.g. R=21,
    # B=19 frames, straddling STACK_GOOD_MIN_FRAMES), channels would get
    # different targetbg and could show a slight tint mismatch — worth a
    # cross-channel min() if that's ever observed in practice.
    # ------------------------------------------------------------------
    if is_rgb_channel:
        if good_stack:
            return _done([f"autostretch -linked -2.8 {TARGETBG_RGB_GOOD}"])
        elif medium_stack:
            return _done([f"autostretch -linked -2.8 {TARGETBG_RGB_MEDIUM}"])
        else:
            return _done([f"autostretch -linked -2.8 {TARGETBG_RGB_SHORT}"])

    # ------------------------------------------------------------------
    # MONO CAMERA — broadband (CLEAR, L, LUMINANCE)
    # CLEAR previously had a flat 0.25 targetbg regardless of frame count —
    # same bug as the old RGB-channel branch. A 40-frame (M16) or 240-frame
    # (M42) CLEAR stack is comfortably "good_stack" quality and was getting
    # the same conservative background depth as a 5-frame stack. Now shares
    # the same quality-scaled values as L/LUMINANCE (they're the same kind
    # of panchromatic mono signal).
    # ------------------------------------------------------------------
    if is_luminance:
        if good_stack:
            return _done([f"autostretch -linked -3.0 {TARGETBG_MONO_BB_GOOD}"])
        elif medium_stack:
            return _done([f"autostretch -linked -2.8 {TARGETBG_MONO_BB_MEDIUM}"])
        else:
            return _done([f"autostretch -linked -2.8 {TARGETBG_MONO_BB_SHORT}"])

    if is_broadband:
        if good_stack:
            return _done([f"autostretch -linked -2.8 {TARGETBG_MONO_BB_GOOD}"])
        elif medium_stack:
            return _done([f"autostretch -linked -2.8 {TARGETBG_MONO_BB_MEDIUM}"])
        else:
            return _done([f"autostretch -linked -2.8 {TARGETBG_MONO_BB_SHORT}"])

    # ------------------------------------------------------------------
    # UNRECOGNIZED FILTER SAFEGUARD (e.g. UHC, CLS, DUAL_NARROWBAND,
    # SOLAR, IR_CUT — anything in VALID_FILTERS but absent from both
    # NARROWBAND_FILTERS and BROADBAND_FILTERS).
    # Every other function in the pipeline (should_apply_denoise,
    # rejection/weighting in generate_siril_stack_script) only special-cases
    # NARROWBAND_FILTERS and treats everything else as broadband-like.
    # Falling through to the narrowband branch below for these filters would
    # apply a darker targetbg than the rest of the pipeline assumes,
    # producing an inconsistently crushed stretch. Mirror that convention
    # here instead of silently reusing the narrowband defaults.
    # ------------------------------------------------------------------
    if not is_narrowband:
        debug(f"Stretch: unrecognized filter '{filter_name}', using broadband defaults")
        return _done(["autostretch -linked -2.8 0.25"])

    # ------------------------------------------------------------------
    # MONO CAMERA — narrowband (HA, OIII, SII, H_BETA)
    # targetbg adapts to stack quality: darker bg when SNR is reliable,
    # brighter when signal may be weak or noise is not well-calibrated.
    # ------------------------------------------------------------------
    if good_stack:
        # Well-calibrated deep stack — push background dark to reveal faint filaments
        return _done([f"autostretch -linked -2.8 {TARGETBG_NB_GOOD}"])
    elif medium_stack:
        # Decent stack — moderate background
        return _done([f"autostretch -linked -2.8 {TARGETBG_NB_MEDIUM}"])
    else:
        # Short stack or missing masters — bright targetbg to avoid clipping signal
        return _done([f"autostretch -linked -2.8 {TARGETBG_NB_SHORT}"])


# --------------------------------------------------------------------------
# GENERATE NATIVE SIRIL SCRIPTS (.SSF)
# --------------------------------------------------------------------------
def generate_siril_stack_script(
    filter_work_dir: Path,
    filter_name: str,
    num_files: int,
    is_color: bool,
    all_detected_filters: list[str],
    master_dark_path: str = None,
    master_flat_path: str = None,
    master_bias_path: str = None,
    output_bits: int = 32,
    light_files: list = None,
    stretch_reference_files: int = None,
) -> str:
    """
    Generates the .ssf script for Siril.
    num_files: used to validate that we have at least 2 images for the stack

    stretch_reference_files: if provided, used INSTEAD of num_files only for
    the targetbg quality bucket (get_stretch_command) — not for rejection,
    weighting, or denoise, which must reflect this filter's actual frame
    count. Pass the minimum frame count across sibling channels that must
    end up with identical stretch parameters (RGB: RED/GREEN/BLUE: narrowband:
    HA/OIII/SII/H_BETA) so a session where channels straddle a quality
    threshold (e.g. R=21, B=19 frames around STACK_GOOD_MIN_FRAMES=20)
    doesn't get a different targetbg per channel and a visible tint mismatch
    in the composite. Defaults to num_files when not provided.
    """

    emit("progress", "generate_siril_stack_script_started", params={
        
        "filter":     filter_name,
        "frames":     num_files,
        "is_color":   is_color,
        "has_dark":   bool(master_dark_path),
        "has_flat":   bool(master_flat_path),
        "has_bias":   bool(master_bias_path),
        "output_bits": output_bits,
    })

    abs_work_dir = filter_work_dir.resolve().as_posix()
    is_narrowband = filter_name.upper() in NARROWBAND_FILTERS
    bit_cmd = "set32bits" if output_bits == 32 else "set16bits"

    check_flat_staleness(master_flat_path, light_files)

    lines = [
        "requires 1.2.0",
        f'cd "{abs_work_dir}"',
        'convert light -out=.',
    ]

    # 1. Calibration
    seq = "light"
    if any([master_dark_path, master_flat_path, master_bias_path]):
        cal_cmd = [f"calibrate {seq}"]
        if master_bias_path: cal_cmd.append(f"-bias={Path(master_bias_path).as_posix()}")
        if master_dark_path: cal_cmd.append(f"-dark={Path(master_dark_path).as_posix()}")
        if master_flat_path: cal_cmd.append(f"-flat={Path(master_flat_path).as_posix()}")
        if is_color: cal_cmd.extend(['-cfa', '-equalize_cfa'])
        lines.append(" ".join(cal_cmd))
        seq = "pp_light"

    # 2. Debayering (OSC only)
    if is_color:
        lines.append(f"preprocess {seq} -debayer")
        seq = f"pp_{seq}"

    # 3. Per-frame background extraction
    # seqsubsky removes the sky gradient from each individual frame before
    # stacking. This is more accurate than post-stack subsky because each
    # frame has its own sky level (changing conditions, atmospheric gradients).
    # Output sequence is prefixed with 'bkg_' by Siril convention.
    seqsubsky_cmd, seq = get_seqsubsky_command(
        seq,
        filter_name,
        is_narrowband,
        num_files,
        master_flat_path,
    )
    if seqsubsky_cmd:
        lines.append(seqsubsky_cmd)

    # 4. Registration
    lines.append(f"register {seq}")

    # -------------------------------------------------------------------------
    # REJECTION METHOD
    # ≤ 3 frames : no rejection — Sigma/Winsorized are statistically undefined
    #              with so few data points and risk leaving 0 usable frames.
    #              Simple mean (no rej keyword) is the only safe option.
    # 4–5 frames : Sigma with wide bounds (4σ) — light outlier rejection only
    # 6–100 frames: Winsorized — robust balance for typical EAA stacks
    # >100 frames: Linear Fit Clipping — best for large, varied-quality stacks
    #
    # sigma_high tightened (3.0->2.0 winsorized, 3.0->2.2 linear) after
    # confirming on real data (M20 LRGB: L=100 frames hit winsorized, R/G/B
    # at 118-126 frames hit linear) that a satellite trail survived rejection
    # under BOTH algorithms at the old bounds — this isn't a "wrong algorithm
    # for this frame count" problem, both let a bright transient through.
    # sigma_low is left wider: trails are bright outliers (need a tighter
    # HIGH bound to catch), tightening low too would mainly cost extra
    # rejection of dark noise/vignetting without helping against trails.
    # Unverified against a next real run — tune SIGMA_HIGH_WINSORIZED /
    # SIGMA_HIGH_LINEAR further if a trail still survives, or loosen if
    # legitimate faint signal starts getting clipped.
    # -------------------------------------------------------------------------
    SIGMA_LOW_WINSORIZED  = 3.0
    SIGMA_HIGH_WINSORIZED = 3.0
    SIGMA_LOW_LINEAR      = 4.0
    SIGMA_HIGH_LINEAR     = 3.0

    if num_files <= 3:
        rejection     = None      # no rejection at all
        sigmas        = None
    elif num_files <= 5:
        rejection     = "sigma"
        sigmas        = "4 4"     # wide bounds — avoid over-clipping small stacks
    elif num_files <= 100:
        rejection     = "winsorized"
        sigmas        = f"{SIGMA_LOW_WINSORIZED} {SIGMA_HIGH_WINSORIZED}"
    else:
        rejection     = "linear"
        sigmas        = f"{SIGMA_LOW_LINEAR} {SIGMA_HIGH_LINEAR}"

    # -------------------------------------------------------------------------
    # WEIGHTING
    # -weight_from_wfwhm   : requires many stars detected by register
    #                        → reliable on OSC broadband / CLEAR with > 15 frames
    #                        → unstable on narrowband (few stars, especially SII)
    # -weight_from_noise   : based on measured background noise
    #                        → reliable on all filters with clean calibration
    #                        → can be skewed if few frames and no dark
    # ""                   : no weighting
    #                        → short narrowband stack without reliable masters
    # -----------------------------------------------------------------------
    if is_color and not is_narrowband and num_files >= 10:
        # OSC broadband (CLEAR, L) with enough frames: wFWHM reliable
        weight = "-weight_from_wfwhm"
    elif any([master_dark_path, master_flat_path, master_bias_path]) and num_files >= 6:
        # Mono or narrowband calibrated: background noise reliable
        weight = "-weight_from_noise"
    else:
        # Very short stack or without calibration: no weighting
        weight = ""

    # -------------------------------------------------------------------------
    # FRAME FILTERING
    # We combine two complementary criteria:
    #
    # -filter-bkg      : excludes frames with sky background too high
    #                    (cloudy passages, strong gradient) — does not require stars
    #                    → applicable to all filters including SII
    #
    # -filter-wfwhm    : excludes frames with bad seeing (stretched PSF)
    #                    → requires stars → only broadband / OSC
    #
    # -filter-quality  : overall quality score calculated by register
    #                    → good complement to wfwhm for OSC
    #
    # Thresholds in %: we keep the best N% of frames.
    # Too restrictive (80%) on a small stack → we adapt to the number of frames.
    # -------------------------------------------------------------------------
    filters = []

    # Sky background filter: universal, safe even on SII
    if num_files >= 6:
        bkg_threshold = "85%" if num_files >= 20 else "90%"
        filters.append(f"-filter-bkg={bkg_threshold}")

    # wFWHM filter: only if enough stars expected
    if not is_narrowband and num_files >= 10:
        wfwhm_threshold = "80%" if num_files >= 30 else "85%"
        filters.append(f"-filter-wfwhm={wfwhm_threshold}")
    elif is_narrowband and not (filter_name.upper() == "SII") and num_files >= 10:
        # HA and OIII still have some stars → more permissive filter
        filters.append("-filter-wfwhm=90%")
    filter_str = " ".join(filters)

    if rejection is None:
        stack_cmd = f"stack r_{seq} -norm=addscale"
    else:
        stack_cmd = f"stack r_{seq} rej {rejection} {sigmas} -norm=addscale"

    stack_parts = [
        stack_cmd,
        weight,
        filter_str
    ]

    lines.extend([
#        f'register {seq}',
        " ".join(p for p in stack_parts if p),
        f"load r_{seq}_stacked.fit",
    ])

    # Early crop — see get_early_crop_command() docstring for why this moved
    # here instead of staying a post-composition Python-side step.
    early_crop_cmd = get_early_crop_command(light_files or [], BORDER_CROP_PERCENT)
    if early_crop_cmd:
        lines.append(early_crop_cmd)

    # 8. Post-stack processing
    apply_denoise, denoise_mod = should_apply_denoise(
        filter_name, num_files, is_color,
        all_detected_filters,
        master_dark_path, master_flat_path, master_bias_path
    )

    # 5. Autostretch
    # Uses stretch_reference_files (group-min across sibling channels) when
    # provided, so RGB/narrowband channels from the same session land in the
    # same good/medium/short bucket together — see docstring above.
    stretch_cmds = get_stretch_command(
        filter_name, is_color,
        stretch_reference_files if stretch_reference_files is not None else num_files,
        any([master_dark_path, master_flat_path, master_bias_path])
    )

    # Post-stack subsky: disabled when seqsubsky ran (per-frame extraction
    # already handled the gradient; double-subsky creates halos).
    # Still active when seqsubsky is disabled (SEQSUBSKY_ENABLED=False).
    post_stack_subsky = "" if seqsubsky_cmd else get_subsky_command(
        filter_name, num_files, is_narrowband,
        master_dark_path, master_flat_path, master_bias_path
    )

    lines.extend([
#        get_subsky_command(
#            filter_name,
#            num_files,
#            is_narrowband,
#            master_dark_path,
#            master_flat_path,
#            master_bias_path
#        ),
        post_stack_subsky,
        get_color_calibration_command(is_color, filter_name, light_files or []),
        get_rmgreen_command(is_color),
        f"denoise -da3d -mod={denoise_mod}" if apply_denoise else "",
        *stretch_cmds,
        bit_cmd,
        "setext fit",
        f'save "../stacked_{filter_name}.fit"',
        "close",
        "exit"
    ])

    script = "\n".join(line for line in lines if line)
    emit("progress", "generate_siril_stack_script_done", params={"filter": filter_name, "lines": len(lines)})
    return script


def generate_siril_script(
    session_dir: Path,
    filter_name: str,
    file_prefix: str,
    fit_path: Path = None
) -> str:
    """
    Generate the intermediate FIT to TIFF conversion script.
    Uses the native 'savetif' command from Siril 1.2 to avoid erroneous hybrid files like '.tif.fit'.
    """

    emit("progress", "generate_siril_script_started", params={
        
        "filter": filter_name,
    })

    if fit_path is None:
        fit_path = session_dir / f"{file_prefix}_{filter_name}.fit"
    tif_path = (session_dir / f"{file_prefix}_{filter_name}").as_posix()

    script = "\n".join([
        "requires 1.2.0",
        f'load "{fit_path.as_posix()}"',
        f'savetif "{tif_path}"',
        "close",
        "exit"
    ])

    emit("progress", "generate_siril_script_done", params={"filter": filter_name, "output_tif": f"{file_prefix}_{filter_name}.tif"})
    return script


# --------------------------------------------------------------------------
# CORE ENGINE EXECUTION (SIRIL-CLI)
# --------------------------------------------------------------------------
def run_siril_command(session_dir: Path, script_content: str, script_name: str, work_dir: Path = None) -> bool:
    """Execute a Siril script with siril-cli."""
    emit("progress", "run_siril_command_started", params={"script": script_name})

    script_path = session_dir / script_name
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    cmd = ["siril-cli", "-s", str(script_path)]
    effective_cwd = str(work_dir) if work_dir else str(session_dir)
    debug(f"Check effective_cwd: {effective_cwd}")

    debug(f"=== Script Siril ({script_name}) Start ===")
    debug(script_content)
    debug(f"=== Script end ===")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=effective_cwd,
            text=True
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                debug(f"[Siril LOG] {line.strip()}")

        process.wait()
        success = process.returncode == 0
        emit("progress", "run_siril_command_done", params={"script": script_name, "success": success, "returncode": process.returncode})
        return success
    except Exception as e:
        debug(f"Fatal error running siril-cli: {e}")
        emit("progress", "run_siril_command_done", params={"script": script_name, "success": False, "error": str(e)})
        return False
    finally:
        if script_path.exists():
            script_path.unlink()


# --------------------------------------------------------------------------
# CHROMINANCE & COMPOSITION VIA IMAGEMAGICK
# --------------------------------------------------------------------------
def crop_registration_border(image_path: Path, percent: float) -> bool:
    """
    UNUSED as of the early-crop change — kept for reference/rollback, not
    called anywhere in the current pipeline.

    Originally shaved a percentage of the border off each side of the
    FINAL composed image (post-ImageMagick composition) to remove
    registration artifacts: thin fringes of black, noisy, or mis-colored
    pixels along one or more edges where not every aligned frame
    contributed data (visible e.g. as a stray-hued line along the top edge
    on multi-channel composites).

    Superseded by get_early_crop_command(), which crops each channel
    individually INSIDE Siril, right after stacking and before subsky —
    closer to the source of the artifact, and avoids the subsky
    contamination and per-channel asymmetric-fringe risks that this
    post-composition approach couldn't address. See that function's
    docstring for the full reasoning. If early-crop is ever disabled and
    this needs reviving, call it from compose_rgb_image again where the
    two "Border crop now happens earlier..." comments currently sit.

    Ported from ImageMagick (-shave) to plain PIL crop() — pixel-exact,
    no behavior change from the original.

    Percentage is computed independently per axis from the image's own
    dimensions so it scales correctly regardless of output resolution.
    No-op if percent <= 0.
    """
    if percent <= 0:
        return True

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        debug(f"Border crop: could not read dimensions of {image_path.name} ({e}), skipping")
        return False

    shave_x = max(1, round(width * percent / 100))
    shave_y = max(1, round(height * percent / 100))

    try:
        with Image.open(image_path) as img:
            cropped = img.crop((shave_x, shave_y, width - shave_x, height - shave_y))
            save_kwargs = {"quality": 95} if image_path.suffix.lower() in (".webp", ".jpg", ".jpeg") else {}
            cropped.save(image_path, **save_kwargs)
        debug(f"Border crop: shaved {shave_x}x{shave_y}px ({percent}%) from {image_path.name}")
        emit("progress", "crop_registration_border_done", params={
            "file": image_path.name, "percent": percent, "shave_x": shave_x, "shave_y": shave_y
        })
        return True
    except Exception as e:
        debug(f"Border crop failed on {image_path.name}: {e}")
        emit("progress", "crop_registration_border_done", params={"file": image_path.name, "success": False, "error": str(e)})
        return False


def reduce_star_halos(image_path: Path) -> bool:
    """
    Shrink bloated bright-star halos on the final composed image, without a
    full star-removal/synthesis pipeline (no StarNet++ dependency, which may
    not be installed on this machine).

    Technique (pure PIL/numpy/scipy — ported from a 4-temp-file ImageMagick
    chain, see below):
      1. mask   = bright-pixel mask from the image's own luma, thresholded at
                  STAR_HALO_MASK_THRESHOLD%, edges feathered with a Gaussian
                  blur so the treated region doesn't end in a hard ring.
      2. eroded = a copy run through a min-filter (scipy.ndimage.minimum_filter)
                  of size STAR_HALO_ERODE_RADIUS, applied per RGB channel —
                  small round bright blobs (stars) shrink noticeably; large
                  diffuse structures (nebula) barely move, since the filter
                  only pulls each pixel down to the darkest value in its
                  small neighborhood.
      3. result = original blended toward eroded, weighted by
                  (mask * STAR_HALO_BLEND_PERCENT) — algebraically the same
                  end result as the original ImageMagick chain (Blend +
                  CopyOpacity + Over collapse into one linear interpolation:
                  result = original*(1 - mask*P) + eroded*(mask*P)), just
                  computed directly instead of through 3 intermediate
                  composited files.

    Approximation notes (functionally equivalent, not guaranteed bit-exact
    vs. the original ImageMagick version):
      - Grayscale luma (PIL .convert("L"), ITU-R 601-2 weights) may differ
        very slightly from ImageMagick's default -colorspace Gray (Rec 709
        in recent IM versions) — negligible for a bright-star threshold mask.
      - Gaussian blur (PIL ImageFilter.GaussianBlur) is a different
        implementation than IM's -blur, both true Gaussians but not
        guaranteed pixel-identical at the edges.

    UNVERIFIED ON REAL DATA (true before this port too — the ImageMagick
    version was never validated against a real render either, so this port
    doesn't regress an already-confirmed baseline). If stars look
    artificially chopped/square, lower STAR_HALO_ERODE_RADIUS or
    STAR_HALO_BLEND_PERCENT first (in that order) rather than raising
    STAR_HALO_MASK_THRESHOLD — a too-high threshold just makes the effect
    patchy, it doesn't make it gentler.
    """
    if not ENABLE_STAR_HALO_REDUCTION:
        return True

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img).astype(np.float32)  # (H, W, 3), 0-255

            # 1. Feathered bright-pixel mask
            gray = np.asarray(img.convert("L")).astype(np.float32)
            thresh_val = STAR_HALO_MASK_THRESHOLD / 100.0 * 255.0
            mask = np.clip((gray - thresh_val) / max(255.0 - thresh_val, 1e-6), 0.0, 1.0)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=STAR_HALO_MASK_FEATHER))
            mask = np.asarray(mask_img).astype(np.float32) / 255.0  # (H, W), 0-1

            # 2. Eroded (min-filtered) copy, independently per channel —
            # matches ImageMagick -statistic Minimum's default per-channel behavior.
            kernel = STAR_HALO_ERODE_RADIUS * 2 + 1
            eroded = np.stack([
                ndimage.minimum_filter(arr[..., c], size=kernel)
                for c in range(3)
            ], axis=-1)

            # 3. Weighted blend toward eroded, restricted to the mask
            alpha = (mask * (STAR_HALO_BLEND_PERCENT / 100.0))[..., None]  # (H, W, 1), broadcasts
            result = arr * (1 - alpha) + eroded * alpha
            result = np.clip(result, 0, 255).astype(np.uint8)

            out_img = Image.fromarray(result, mode="RGB")
            save_kwargs = {"quality": 95} if image_path.suffix.lower() in (".webp", ".jpg", ".jpeg") else {}
            out_img.save(image_path, **save_kwargs)

        emit("progress", "reduce_star_halos_done", params={"file": image_path.name, "success": True})
        return True
    except Exception as e:
        debug(f"Star halo reduction failed on {image_path.name}: {e}")
        emit("progress", "reduce_star_halos_done", params={"file": image_path.name, "success": False, "error": str(e)})
        return False


LOSSY_OUTPUT_FORMATS = {"webp", "jpg", "jpeg"}


def get_composition_paths(session_dir: Path, file_prefix: str, output_format: str) -> tuple:
    """
    Returns (work_path, final_path) for the composition pipeline.

    work_path is where compose_rgb_image (ImageMagick) and reduce_star_halos
    (PIL) both read/write, and is ALWAYS a lossless format (.tif) when the
    user requested a lossy final format (webp/jpg). final_path is the actual
    requested output (f"{file_prefix}_full.{output_format}").

    Why: these two steps used to write directly to the final webp/jpg, and
    correct_image_orientation (running after them) opened and re-saved that
    same lossy file again — meaning a webp/jpg output went through 2-3
    separate lossy re-encodes (compose -> halo reduction -> orientation
    flip), each pass quietly degrading fine detail a bit more. Confirmed on
    a real M16 render: the webp output had ~10% lower background noise
    std-dev than an equivalent TIFF of the same underlying data — consistent
    with repeated lossy compression smoothing out grain, not a processing
    difference. Now only correct_image_orientation (the last step in the
    pipeline) does the one-time lossy encode, into final_path.

    For an already-lossless target (png/tiff), work_path == final_path —
    nothing changes for those, no extra conversion step needed.
    """
    final_path = session_dir / f"{file_prefix}_full.{output_format}"
    if output_format.lower() in LOSSY_OUTPUT_FORMATS:
        work_path = session_dir / f"{file_prefix}_full.tif"
    else:
        work_path = final_path
    return work_path, final_path


def compose_rgb_image(
    session_dir: Path,
    tif_files: dict,
    output_format: str,
    file_prefix: str
) -> bool:
    """
    Combine normalized TIFF files into a final RGB image using ImageMagick.

    Supported palettes (detected automatically from available channels):
      - CLEAR / L  : single broadband mono channel with full post-processing
      - Mono        : single channel (any other filter) — passthrough
      - RGB         : RED + GREEN + BLUE
      - HOO         : HA→R, blend(HA 30% + OIII 70%)→G, OIII→B
      - SHO         : SII→R, HA→G, OIII→B  (classic Hubble palette)
      - SHO+RGB     : SHO with color stars blended from RGB (80% NB / 20% RGB)
      - SHA         : SII→R, HA→G, BLUE→B  (or darkened HA fallback if no BLUE)
      - HaRGB       : blend(HA 60% + RED 40%)→R, GREEN→G, BLUE→B

    Each palette applies a tailored post-processing chain (level, tone curve,
    saturation, sharpening) to account for the very different signal
    characteristics of narrowband vs broadband data.

    CLEAR/L/LUMINANCE are treated as a named palette rather than a generic
    mono passthrough so that broadband OSC images receive proper post-processing
    (sigmoidal contrast, sharpening) instead of being exported flat.
    """
    emit("progress", "compose_rgb_image_started", params={
               
        "channels":     list(tif_files.keys()),
        "output_format": output_format,
    })

    output_file, _final_output_file = get_composition_paths(session_dir, file_prefix, output_format)

    # Resolve reference geometry for synthetic black channels
    ref_path = next(iter(tif_files.values()))
    width, height = get_image_dimensions(ref_path)
    with Image.open(ref_path) as _ref_img:
        ref_mode = _ref_img.mode  # matched exactly when generating synthetic black channels below

    # ------------------------------------------------------------------
    # CHANNEL DETECTION
    # ------------------------------------------------------------------
    has_ha        = "HA"    in tif_files
    has_oiii      = "OIII"  in tif_files
    has_sii       = "SII"   in tif_files
    has_rgb       = all(k in tif_files for k in ("RED", "GREEN", "BLUE"))
    has_clear     = any(k in tif_files for k in ("CLEAR", "L", "LUMINANCE"))

    # Resolve the actual key used for the broadband mono channel
    clear_key     = next((k for k in ("CLEAR", "L", "LUMINANCE") if k in tif_files), None)

    _synthetic_black_path = {"path": None}

    def chan(key: str) -> str:
        """
        Return the channel TIFF path, or a synthetic black image of the correct size.

        Generated via PIL (Image.new), matched to the reference channel's
        exact mode/bit depth (ref_mode, resolved above) so it combines
        cleanly with the real channels in the ImageMagick -combine step
        downstream. Previously used ImageMagick's 'xc:black' pseudo-format —
        note the OLD 'xc:black[WxH]' syntax used before that was invalid
        (the [WxH] suffix is a read-region modifier, not a canvas-size
        directive, and silently produced a 1x1px image). No ImageMagick
        fallback here on purpose: a failure is surfaced as an exception
        rather than silently reintroducing that same wrong-size bug.
        """
        if key in tif_files:
            return str(tif_files[key])

        if _synthetic_black_path["path"] is None:
            black_path = session_dir / f".synthetic_black_{width}x{height}.tif"
            Image.new(ref_mode, (width, height), color=0).save(black_path)
            _synthetic_black_path["path"] = str(black_path)
            debug(f"Synthetic black channel generated: {black_path.name} ({width}x{height}, mode={ref_mode})")
        return _synthetic_black_path["path"]

    def blend_cmd(img_a: str, img_b: str, pct_a: int) -> list:
        """
        Blend two images using ImageMagick -compose Blend.
        Result = (pct_a% × img_a) + ((100 - pct_a)% × img_b)
        Must be used inside a -combine pipeline: each blend produces one channel.
        """
        return [
            "(", img_a, img_b,
            "-define", f"compose:args={pct_a},{100 - pct_a}",
            "-compose", "Blend",
            "-composite",
            ")",
        ]

    # ------------------------------------------------------------------
    # MONO PASSTHROUGH — single channel, not broadband
    # Applies the same finishing chain as the CLEAR palette (level clip,
    # sigmoidal contrast, light sharpening) so a single-filter session
    # (e.g. HA only) doesn't ship a flat, unprocessed export while every
    # other palette gets full post-processing.
    # CLEAR/L/LUMINANCE are handled below as a named palette instead.
    # ------------------------------------------------------------------
    if len(tif_files) == 1 and not has_clear:
        single_channel = list(tif_files.values())[0]
        cmd = [
            "convert", str(single_channel),
            "-level", "1%,99%",
            "-sigmoidal-contrast", "3x48%",
            "-unsharp", "0x1.0+0.5+0.02",
        ]
        # No -quality flag here: output_file is always the lossless work_path
        # now (see get_composition_paths) — the one-time lossy encode, if
        # the user requested webp/jpg, happens later in
        # correct_image_orientation, the pipeline's last step.
        cmd.append(str(output_file))
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            success = result.returncode == 0
            if not success:
                debug(f"ImageMagick STDERR (MONO): {result.stderr}")
            if success:
                # Border crop now happens earlier, per-channel, inside the
                # Siril .ssf script (see get_early_crop_command()) — no
                # longer needed here. Calling crop_registration_border again
                # on this already-cropped image would double-crop it.
                reduce_star_halos(output_file)
            emit("progress", "compose_rgb_image_done", params={"palette": "MONO", "success": success})
            return success
        except Exception as e:
            debug(f"ImageMagick Single Channel Failure: {e}")
            emit("progress", "compose_rgb_image_done", params={"palette": "MONO", "success": False, "error": str(e)})
            return False

    # ------------------------------------------------------------------
    # PALETTE SELECTION — priority order matters:
    # SHO+RGB before SHO (superset), HOO before RGB (narrowband priority),
    # CLEAR before RGB fallback (explicit broadband mono handling).
    # ------------------------------------------------------------------

    # SHO + RGB : full narrowband palette with color star blending.
    # Each channel = 80% narrowband + 20% broadband RGB.
    # Preserves natural star colors while keeping NB nebulosity dominant.
    if has_sii and has_ha and has_oiii and has_rgb:
        debug("Palette: SHO+RGB (color stars blend)")
        cmd = ["convert"]
        for nb_key, rgb_key in [("SII", "RED"), ("HA", "GREEN"), ("OIII", "BLUE")]:
            cmd.extend(blend_cmd(chan(nb_key), chan(rgb_key), 80))
        palette_label = "SHO+RGB"

    # SHO : classic Hubble palette — SII→R, HA→G, OIII→B
    elif has_sii and has_ha and has_oiii:
        debug("Palette: SHO (Hubble)")
        cmd = ["convert", chan("SII"), chan("HA"), chan("OIII")]
        palette_label = "SHO"

    # HOO : HA→R, blend(HA+OIII)→G, OIII→B
    # The blended green channel (30% HA / 70% OIII) creates smooth color
    # transitions between HA-dominant (red/orange) and OIII-dominant (blue/cyan)
    # regions, avoiding the flat blue-gray look of a pure OIII green.
    elif has_ha and has_oiii and not has_sii:
        debug("Palette: HOO")
        cmd = ["convert",
            chan("HA"),                                      # R = HA
            *blend_cmd(chan("HA"), chan("OIII"), 30),        # G = 30% HA + 70% OIII
            chan("OIII"),                                    # B = OIII
        ]
        palette_label = "HOO"

    # SHA : SII→R, HA→G, BLUE→B (or darkened HA fallback)
    # Used when OIII is unavailable. If a broadband BLUE channel exists it
    # provides a real blue reference; otherwise HA is gamma-darkened to
    # simulate a cooler blue-shifted channel without a pure black gap.
    elif has_sii and has_ha and not has_oiii:
        debug("Palette: SHA")
        if has_rgb:
            # Ideal: use the real broadband BLUE channel for the blue slot
            debug("  SHA: using broadband BLUE channel")
            cmd = ["convert", chan("SII"), chan("HA"), chan("BLUE")]
        else:
            # Fallback: darken HA via gamma boost to simulate a blue-shifted channel.
            # -level 0%,100%,1.5 raises midtones; -modulate 60 reduces overall brightness.
            debug("  SHA fallback: gamma-darkened HA as synthetic blue channel")
            cmd = [
                "convert",
                chan("SII"),
                chan("HA"),
                "(", chan("HA"),
                    "-level", "0%,100%,1.5",
                    "-modulate", "60",
                ")",
            ]
        palette_label = "SHA"

    # HaRGB : HA-enriched red channel blended with broadband RGB.
    # Boosts ionized hydrogen regions (HA) in the red channel while keeping
    # natural star colors from the broadband GREEN and BLUE channels.
    elif has_ha and has_rgb:
        debug("Palette: HaRGB")
        cmd = ["convert",
            *blend_cmd(chan("HA"), chan("RED"), 60),         # R = 60% HA + 40% RED
            chan("GREEN"),                                   # G = broadband GREEN
            chan("BLUE"),                                    # B = broadband BLUE
        ]
        palette_label = "HaRGB"

    elif has_clear and has_rgb:
        debug(f"Palette: LRGB (key={clear_key})")
        cmd = ["convert",
            # Extract Hue channel from the RGB base
            "(",
                "(", chan("RED"), chan("GREEN"), chan("BLUE"),
                    "-combine", "-colorspace", "sRGB",
                ")",
                "-colorspace", "HSL",
                "-channel", "R", "-separate", "+channel",
            ")",
            # Extract Saturation channel from the RGB base
            "(",
                "(", chan("RED"), chan("GREEN"), chan("BLUE"),
                    "-combine", "-colorspace", "sRGB",
                ")",
                "-colorspace", "HSL",
                "-channel", "G", "-separate", "+channel",
            ")",
            # Dedicated Luminance channel
            str(tif_files[clear_key]),
            # CRITICAL: -set colorspace HSL DECLARES the colorspace without converting.
            # Using -colorspace HSL here would CONVERT [H,S,L] data as if it were sRGB
            # → completely wrong color mapping → the red/noisy catastrophe observed.
            "-combine",
            "-set", "colorspace", "HSL",    # ← déclaration, PAS conversion
            "-colorspace", "sRGB",           # ← conversion HSL→sRGB
        ]
        palette_label = "LRGB"

    # CLEAR / L / LUMINANCE : broadband mono channel.
    # Treated as a named palette to apply a proper post-processing chain.
    # A generic passthrough would export a flat, low-contrast image on
    # diffuse nebulae (M16, M42) where the dynamic range is compressed
    # after Siril's autostretch.
    elif has_clear:
        debug(f"Palette: CLEAR (key={clear_key})")
        cmd = ["convert", str(tif_files[clear_key])]
        palette_label = "CLEAR"

    # Classic RGB — standard broadband color image.
    else:
        debug("Palette: RGB")
        missing_channels = [k for k in ("RED", "GREEN", "BLUE") if k not in tif_files]
        if missing_channels:
            debug(f"WARNING: missing RGB channels {missing_channels}, filling with black.")
        cmd = ["convert", chan("RED"), chan("GREEN"), chan("BLUE")]
        palette_label = "RGB"

    emit("progress", "combine", params={"palette": palette_label, "message": f"Selected palette: {palette_label}"})

    # ------------------------------------------------------------------
    # FINALIZATION — combine channels then apply palette-specific
    # post-processing: level clip, tone curve, saturation, sharpening.
    #
    # Design rationale per palette:
    #
    #   SHO / SHO+RGB
    #     Hubble palette has very narrow tonal range per channel.
    #     Gentle level clip (0.5%) preserves faint emission structures.
    #     Moderate sigmoidal (3x45%) avoids crushing the dim filaments.
    #     Saturation boost (135%) compensates for the inherently muted
    #     SII/HA/OIII color separation after combination.
    #     Green cast correction (-level 0%,95% on Green channel) removes
    #     the yellow-green tint inherent to the HA→G mapping.
    #
    #   HOO
    #     Similar narrowband constraints. Slightly lower saturation (145%)
    #     and stronger sigmoidal (4x45%) because the blended green channel
    #     adds chromatic variety but OIII-dominant targets need more punch.
    #
    #   SHA
    #     Bi-filter palette; less chromatic range than tri-filter.
    #     Lower saturation boost (115%) to avoid over-saturating the
    #     SII red channel which tends to dominate.
    #
    #   HaRGB
    #     Mixed narrowband/broadband. Moderate sigmoidal + light saturation
    #     boost (110%) to enhance HA regions without skewing star colors.
    #
    #   RGB
    #     Standard broadband. Stronger sigmoidal (4x50%) for punchier
    #     contrast. Larger unsharp radius for star/detail crispness.
    #
    #   CLEAR / L / LUMINANCE
    #     Broadband mono. No -combine step (single channel).
    #     Sigmoidal (4x50%) matches RGB broadband treatment.
    #     Lighter sharpening than RGB to avoid amplifying mono noise.
    # ------------------------------------------------------------------

    # Multi-channel palettes need -combine to merge R/G/B into a single image.
    # CLEAR and Mono passthrough are already single images — skip -combine.
    if palette_label not in ("CLEAR", "LRGB"):
        cmd.extend([
            "-combine",
            "-colorspace", "sRGB",
        ])

    if palette_label in ("SHO", "SHO+RGB"):
        cmd.extend([
            "-level", "0.5%,99.5%",
            "-sigmoidal-contrast", "3x45%",
            "-modulate", "100,135",
            "-unsharp", "0.5x1.0+0.5+0.01",
            # Subtle green cast reduction: HA mapped to Green introduces a
            # yellow-green tint on stars; pulling the Green ceiling to 95%
            # neutralizes it without affecting nebula hue significantly.
            "-channel", "Green", "-level", "0%,95%", "+channel",
        ])

    elif palette_label == "HOO":
        cmd.extend([
            "-level", "0.5%,99.5%",
            "-sigmoidal-contrast", "4x45%",
            "-modulate", "100,145",
            "-unsharp", "0.5x1.0+0.5+0.01",
        ])

    elif palette_label == "SHA":
        cmd.extend([
            "-level", "0.5%,99.5%",
            "-sigmoidal-contrast", "3x45%",
            "-modulate", "100,115",
            "-unsharp", "0.5x1.0+0.5+0.01",
        ])

    elif palette_label == "HaRGB":
        cmd.extend([
            "-level", "1%,99%",
            "-sigmoidal-contrast", "3x45%",
            "-modulate", "100,110",
            "-unsharp", "0x1.0+0.6+0.02",
        ])

    elif palette_label == "LRGB":
        cmd.extend([
            "-level", "1%,99%",
            "-sigmoidal-contrast", "3x50%",
            "-modulate", "100,110",           # slight chrominance saturation boost
            "-unsharp", "0x1.0+0.6+0.02",
        ])

    elif palette_label == "RGB":
        cmd.extend([
            "-level", "1%,98%",
            "-sigmoidal-contrast", "4x50%",
            "-unsharp", "0x1.2+0.8+0.05",
            # OSC background neutralization (approximate cc -nostellar equivalent).
            # Sample the corners (sky background) and normalize each channel
            # to correct the RGGB Bayer red-channel dominance on emission nebulae.
            # This is a best-effort correction for Siril 1.2 where cc is unavailable.
            "-region", "100x100+0+0",
            "-auto-level",
            "+region",
        ])

    elif palette_label == "CLEAR":
        cmd.extend([
            "-level", "1%,99%",
            "-sigmoidal-contrast", "3x48%",
            "-unsharp", "0x1.0+0.5+0.02",
        ])

    else:
        # Safety fallback — should never be reached with the palette logic above
        cmd.extend(["-level", "2%,98%"])

    # No -quality flag here: output_file is always the lossless work_path now
    # (see get_composition_paths) — the one-time lossy encode, if the user
    # requested webp/jpg, happens later in correct_image_orientation, the
    # pipeline's last step.

    cmd.append(str(output_file))

    debug(f"ImageMagick [{palette_label}]: {' '.join(str(c) for c in cmd)}")

    emit("progress", "imagemagick_started", params={"palette": palette_label})

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            debug(f"ImageMagick STDERR: {result.stderr}")
        success = result.returncode == 0
        emit("progress", "imagemagick_done", params={"palette": palette_label, "success": success})
        if success:
            # Border crop now happens earlier, per-channel, inside the
            # Siril .ssf script (see get_early_crop_command()) — no longer
            # needed here. Calling crop_registration_border again on this
            # already-cropped image would double-crop it.
            reduce_star_halos(output_file)
        emit("progress", "compose_rgb_image_done", params={"palette": palette_label, "success": success})
        return success
    except Exception as e:
        debug(f"ImageMagick composite assembly failure: {e}")
        emit("progress", "imagemagick_done", params={      "palette": palette_label, "success": False, "error": str(e)})
        emit("progress", "compose_rgb_image_done", params={"palette": palette_label, "success": False, "error": str(e)})
        return False
    finally:
        if _synthetic_black_path["path"] and _synthetic_black_path["path"] != "xc:black":
            try:
                Path(_synthetic_black_path["path"]).unlink(missing_ok=True)
            except Exception as e:
                debug(f"Failed to remove synthetic black channel file: {e}")


def get_image_dimensions(ref_path: Path) -> tuple:
    """Get real dimensions of the master FITS for ImageMagick."""
    try:
        with Image.open(ref_path) as img:
            return img.size # Returns (width, height)
    except Exception:
        # Fallback if PIL fails
        debug(f"Warning: Could not read dimensions of {ref_path}, fallback 2048x2048")
        return (2048, 2048)


def correct_image_orientation(image_path: Path, fits_source_path: Path = None, final_path: Path = None) -> None:
    """
    Flip the image vertically to correct FITS bottom-up storage convention.

    FITS standard stores image rows starting from the bottom. Siril always
    reads/displays/processes images in bottom-up order internally, REGARDLESS
    of any ROWORDER header on the source file — per Siril's own docs, ROWORDER
    only tells Siril which Bayer demosaic pattern to use on OSC data, it does
    NOT change how Siril orients the image for display or for its own FITS
    output. So for a Siril-driven pipeline, checking ROWORDER to decide
    whether to flip is meaningless: the flip is always needed to go from
    Siril's bottom-up convention to the conventional top-left screen origin.

    (A previous version of this function conditionally skipped the flip based
    on ROWORDER, assuming it was a display-orientation hint like it is in some
    other tools — e.g. SharpCap/INDI writers. That assumption doesn't hold for
    Siril and produced upside-down composites. Do not reintroduce that check
    without first confirming Siril's ROWORDER semantics have changed.)

    final_path: if given and different from image_path, this call also
    performs the pipeline's ONE-AND-ONLY lossy encode (webp/jpg, quality=95)
    here, writing the flipped result to final_path and deleting the
    lossless intermediate (image_path) afterward. This function runs last
    in the pipeline (called from run(), after compose_rgb_image and
    reduce_star_halos have both already finished), which is why it's the
    right place to do this — see get_composition_paths()'s docstring for
    the full reasoning on why the earlier steps deliberately avoid writing
    lossy output directly. If final_path is None or equal to image_path,
    behavior is unchanged: flip in place.
    """
    emit("progress", "correct_image_orientation_started", params={"file": image_path.name})

    target = final_path if final_path is not None else image_path
    try:
        with Image.open(image_path) as img:
            flipped = ImageOps.flip(img)  # vertical flip (top-bottom), matches ImageMagick -flip
            save_kwargs = {"quality": 95} if target.suffix.lower() in (".webp", ".jpg", ".jpeg") else {}
            flipped.save(target, **save_kwargs)
        if final_path is not None and final_path != image_path:
            try:
                image_path.unlink(missing_ok=True)
            except Exception as e:
                debug(f"Could not remove lossless intermediate {image_path.name}: {e}")
        emit("progress", "correct_image_orientation_done", params={"file": target.name, "flipped": True})
    except Exception as e:
        debug(f"Orientation flip failed: {e}")
        emit("progress", "correct_image_orientation_done", params={"file": image_path.name, "flipped": False, "error": str(e)})


# --------------------------------------------------------------------------
# IMAGE ALIGNMENT WITH SIRIL-CLI
# --------------------------------------------------------------------------
def align_channels(session_dir: Path, images_to_align: list[Path], ref_image: Path = None) -> bool:
    """
    Align each master FITS to a common reference using Siril's register command.

    Siril 1.2 does not support coregister (single-image alignment). The workaround
    is to build an artificial 2-frame sequence [reference, target] per pair, run
    register on it, then extract r_align_00002.fit as the aligned result.

    Root cause of the previous failure: calling `convert align -out=.` when
    the source files (align_00001.fit, align_00002.fit) are IN the same directory
    as the output causes Siril to overwrite the input files during conversion.
    cfitsio then fails to open the corrupted/empty file on the subsequent `register`.

    Fix: source files are placed in a `src/` subdirectory, while the `convert`
    output is directed to the parent work directory via an absolute -out= path.
    This completely eliminates the read/write filename collision.

    Directory layout per channel pair:
        _align_work_{stem}/
            src/
                align_00001.fit  ← reference copy (input only)
                align_00002.fit  ← target copy    (input only)
            align_00001.fit      ← written by `convert align -out=work_dir`
            align_00002.fit      ← written by `convert align -out=work_dir`
            align_.seq           ← written by `convert`
            r_align_00002.fit    ← written by `register` (aligned result)

    Args:
        session_dir:      Root session directory, used to write the .ssf script.
        images_to_align:  List of master FITS files to align onto the reference.
        ref_image:        Reference FITS frame (typically HA or the first stacked channel).

    Returns:
        True if all alignments succeeded, False if any channel failed.
        On failure, the original (unaligned) files remain in master_files_map
        and the pipeline continues without inter-filter alignment.
    """
    emit("progress", "align_channels_started", params={
            
        "reference": ref_image.name if ref_image else None,
        "to_align":  [img.name for img in images_to_align if img != ref_image],
        "count":     len([img for img in images_to_align if img != ref_image]),
    })

    if not images_to_align or not ref_image:
        emit("progress", "align_channels_done", params={"success": False, "reason": "no_images_or_reference"})
        return False

    all_ok = True

    for img in images_to_align:
        if img == ref_image:
            continue

        emit("progress", "channel_alignment_started", params={
                
            "channel":   img.stem,
            "reference": ref_image.stem,
        })

        work_dir = session_dir / f"_align_work_{img.stem}"
        src_dir  = work_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy both frames into src/ subdirectory.
            # convert will read from src/ and write to work_dir/ → no filename collision.
            for dst, src in [
                (src_dir / "align_00001.fit", ref_image),   # frame 0 = reference
                (src_dir / "align_00002.fit", img),          # frame 1 = target
            ]:
                if dst.exists():
                    dst.unlink()
                shutil.copy2(src, dst)

            output_aligned = img.parent / f"{img.stem}_aligned.fit"

            # Script flow:
            #   1. cd src/        → Siril working dir points to the source files
            #   2. convert align  → reads src/align_0000*.fit,
            #                       writes converted FITS + align_.seq to work_dir/
            #                       (absolute -out avoids any relative-path ambiguity)
            #   3. cd work_dir/   → switch to the directory holding the converted files
            #   4. register align → reads align_.seq, aligns frame 1 onto frame 0,
            #                       writes r_align_00001.fit and r_align_00002.fit
            #   5. load/save      → extract the aligned target to the session directory
            script = "\n".join([
                "requires 1.2.0",
                f'cd "{src_dir.as_posix()}"',
                "convert align -out=..",
                f'cd "{work_dir.as_posix()}"',
                "register align",
                'load "r_align_00002.fit"',
                f'save "{output_aligned.as_posix()}"',
                "close",
                "exit",
            ])

            success = run_siril_command(
                session_dir,
                script,
                f"align_{img.stem}.ssf"
            )

            if not success or not output_aligned.exists():
                debug(f"⚠️ Alignment failed for {img.name}")
                emit("progress", "channel_alignment_done", params={"channel": img.stem, "success": False})
                all_ok = False
            else:
                debug(f"✅ Aligned: {img.name} → {output_aligned.name}")
                emit("progress", "channel_alignment_done", params={"channel": img.stem, "success": True, "output": output_aligned.name})

        except Exception as e:
            debug(f"❌ Unexpected error aligning {img.name}: {e}")
            emit("progress", "channel_alignment_done", params={"channel": img.stem, "success": False, "error": str(e)})
            all_ok = False

        finally:
            # Always remove the work directory, even on failure
            shutil.rmtree(work_dir, ignore_errors=True)

    emit("progress", "align_channels_done", params={"success": all_ok})
    return all_ok


# --------------------------------------------------------------------------
# IMAGE ALIGNMENT WITH ASTROALIGN (NOT CALLED)
# --------------------------------------------------------------------------
def align_images_with_astroalign(session_dir, filter_name, reference_path):
    """Align all images of a filter to a reference with Astroalign.
    """
    emit("progress", "align_images_with_astroalign_started", params={"filter": filter_name})
    try:
        import astroalign as aa
        from astropy.nddata import CCDData
    except ImportError:
        debug("Astroalign not installed. Use 'pip install astroalign' to enable it.")
        emit("progress", "align_images_with_astroalign_done", params={"success": False, "reason": "not_installed"})
        return False

    reference = CCDData.read(reference_path, unit='adu')
    for img_path in sorted(session_dir.glob(f"*{filter_name}*.fit")):
        if img_path == reference_path:
            continue
        image = CCDData.read(img_path, unit='adu')
        try:
            aligned, _ = aa.register(image, reference)
            aligned.write(img_path.with_name(f"{img_path.stem}_aligned{img_path.suffix}"), overwrite=True)
        except Exception as e:
            debug(f"Astroalign alignment failure for {img_path}: {e}")
            emit("progress", "align_images_with_astroalign_done", params={"success": False, "error": str(e)})
            return False

    debug(f"Astroalign alignment completed for {filter_name}")
    emit("progress", "align_images_with_astroalign_done", params={"success": True, "filter": filter_name})
    return True


# --------------------------------------------------------------------------
# COORDINATION AND MAIN RUNNER
# --------------------------------------------------------------------------
def cleanup_session(session_dir: Path):
    """Clean all temporary files, including converted masters and gradient residuals."""

    emit("progress", "cleanup_session_started", params={"session": session_dir.name})

    temp_patterns = [
        "*.tmp", "*.log", "*.pid", "*.lock", "*.txt",
        "work_*/",           # Working directories
#        "stacked_*.fit",     # Stacked masters
        "stacked_*.tif",     # Masters converted to TIFF
        "*_aligned.fit",
        "r_pp_*.fit",        # Calibrated sequences
        "pp_*.fit",          # Pre-calibration
        "r_light_*.fit",     # Aligned lights
        "light_*.fit",       # Converted lights
        "master*_2d.fit",    # ✅ Residual gradient models generated by subsky
        "*.ssf"              # Temporary Siril scripts
    ]

    # Clean in main directory and DOF subdirectories
    base_dirs = [
        session_dir,
        session_dir / "darks",
        session_dir / "bias",
        session_dir / "flats"
    ]

    deleted_count = 0
    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue
        for pattern in temp_patterns:
            matches = list(base_dir.glob(pattern))
            for match in matches:
                try:
                    if match.is_file():
                        match.unlink()
                        deleted_count += 1
                        debug(f"Temporary file deleted: {match}")
                    elif match.is_dir():
                        shutil.rmtree(match)
                        deleted_count += 1
                        debug(f"Temporary directory deleted: {match}")
                except Exception as e:
                    debug(f"Failed to delete {match}: {e}")

    emit("progress", "cleanup_session_done", params={"deleted": deleted_count})


def run(args) -> bool:
    start = datetime.now()
    session_uuid = args.uuid
    local_sessions_root = BASE_DIR / "sessions"
    current_session_dir = local_sessions_root / session_uuid

    lights_dir = current_session_dir / "lights"
    format_requested = args.format.lower()
    dso_name = re.sub(r'[^a-zA-Z0-9_-]', '', DSO_NAME.lower().replace(" ", ""))

    first_light = None
    for f in sorted(lights_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            first_light = f
            break

    output_bits = 32
    if first_light:
        output_bits = get_fits_bitdepth(first_light)
        debug(f"Detected Bit depth from {first_light.name}: {output_bits}-bit")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_prefix = f"{dso_name}_{timestamp}"

    if not lights_dir.is_dir():
        emit("error",  "start", params={"detail": "Lights directory not found"})
        return False

    cleanup_session(current_session_dir)

    # 1. Sort and index files by detected filter
    emit("progress", "start", params={"target": dso_name, "message": f"Processing started for {dso_name}"})
    emit("progress", "file_analysis", params={"target": dso_name})
    debug(f"Analyzing raws for {dso_name.upper()} | Session: {current_session_dir.name}")

    files_by_filter = {}
    for f in lights_dir.iterdir():
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            matched_filter = None

            # Step 1: Try to read filter from FITS header
            try:
                with fits.open(f, mode='readonly', ignore_missing_end=True) as hdul:
                    header = hdul[0].header
                    filter_keyword = header.get('FILTER', '').strip()

                    if not filter_keyword:
                        filter_keyword = header.get('FILTERS', '').strip()
                    if not filter_keyword:
                        filter_keyword = header.get('IMAGETYP', '').strip()
                    if not filter_keyword:
                        filter_keyword = header.get('HIERARCH ESO INS FILT1 NAME', '').strip()

                    filter_map = {
                        'Red': 'RED', 'RED': 'RED', 'R': 'RED',
                        'Green': 'GREEN', 'GREEN': 'GREEN', 'G': 'GREEN',
                        'Blue': 'BLUE', 'BLUE': 'BLUE', 'B': 'BLUE',
                        'Ha': 'HA', 'H-alpha': 'HA', 'H-ALPHA': 'HA', 'H-A': 'HA', 'Hydrogen Alpha': 'HA', 'H': 'HA',
                        'Oiii': 'OIII', 'O-III': 'OIII', 'OXYGEN III': 'OIII', 'O': 'OIII',
                        'Sii': 'SII', 'S-II': 'SII', 'SULPHUR II': 'SII', 'S': 'SII',
                        'Luminance': 'LUMINANCE', 'LUM': 'LUMINANCE', 'L': 'LUMINANCE',
                        'Clear': 'CLEAR', 'BROADBAND': 'BROADBAND',
                        'IR-Cut': 'IR_CUT', 'UV-IR Cut': 'UV_IR_CUT',
                        'UHC': 'UHC', 'CLS': 'CLS', 'DUAL': 'DUAL_NARROWBAND',
                        'LRGB': 'LRGB', 'RGB': 'RGB', 'H-Beta': 'H_BETA', 'SOLAR': 'SOLAR'
                    }

                    filter_keyword = filter_keyword.replace(' ', '').replace('-', '_')

                    filter_map_normalized = {
                        k.replace(' ', '').replace('-', '_'): v
                        for k, v in filter_map.items()
                    }
                    matched_filter = filter_map_normalized.get(filter_keyword)
            except Exception as e:
                debug(f"Failed to read FITS header: {f.name} ({type(e).__name__}: {e})")

            # Step 2: If step 1 failed or no valid filter found, try with filename
            if matched_filter is None:
                filename_upper = f.name.upper()
                if re.search(r'[_\s]L[_\s\.]', filename_upper):
                    matched_filter = "LUMINANCE"
                else:
                    for v_filter in VALID_FILTERS:
                        if v_filter in filename_upper:
                            matched_filter = v_filter
                            break

            # Step 3: If both previous steps failed, default to CLEAR
            if matched_filter is None:
                matched_filter = "CLEAR"
                debug(f"Not identified, classified as CLEAR: {f.name}")

            if matched_filter not in files_by_filter:
                files_by_filter[matched_filter] = []
            files_by_filter[matched_filter].append(f)

    detected_filters = list(files_by_filter.keys())
    if not detected_filters:
        emit("error", "start", params={"detail": "No valid FITS files found"})
        return False

    PRIORITY = {
        'HA': 0, 'OIII': 1, 'SII': 2,
        'RED': 3, 'GREEN': 4, 'BLUE': 5,
        'LUMINANCE': 6, 'CLEAR': 7
    }
    detected_filters = sorted(detected_filters, key=lambda f: PRIORITY.get(f, 99))
    emit("progress", "filters_detected", params={"filters": detected_filters})

    # Cross-channel stretch consistency: RGB channels (and, separately,
    # narrowband channels) must share the same targetbg bucket or the
    # composite can show a slight tint mismatch between channels that
    # straddle a quality threshold (e.g. R=21 frames "good", B=19 "medium").
    # Compute the group minimum once here and pass it through to
    # generate_siril_stack_script as stretch_reference_files.
    _rgb_group_filters = [f for f in detected_filters if f in {"RED", "GREEN", "BLUE"}]
    _nb_group_filters  = [f for f in detected_filters if f in NARROWBAND_FILTERS]
    rgb_group_min_files = min((len(files_by_filter[f]) for f in _rgb_group_filters), default=None)
    nb_group_min_files  = min((len(files_by_filter[f]) for f in _nb_group_filters), default=None)
    if _rgb_group_filters:
        debug(f"RGB group stretch reference: {rgb_group_min_files} frames (min of {[len(files_by_filter[f]) for f in _rgb_group_filters]})")
    if _nb_group_filters:
        debug(f"Narrowband group stretch reference: {nb_group_min_files} frames (min of {[len(files_by_filter[f]) for f in _nb_group_filters]})")

    total_files = sum(len(v) for v in files_by_filter.values())
    emit("progress", "filters_detected", params={
        "filters":     detected_filters,
        "count":       len(detected_filters),
        "total_files": total_files,
    })

    # 2. Sensor type diagnostic on the first available raw
    first_filter_key = detected_filters[0]
    first_fits_file = files_by_filter[first_filter_key][0]
#     camera_is_color = is_color_camera(first_fits_file)

    # Dictionary for storing generated master FITS for final cross-alignment
    master_files_map = {}

    # 3. Individual processing and stacking of channels in Siril
    for current_filter in detected_filters:
        emit("progress", "stacking_started", params={"filter": current_filter})

        master_dark_path = get_master_dark_path(current_session_dir, current_filter)
        master_flat_path = get_master_flat_path(current_session_dir, current_filter)
        master_bias_path = get_master_bias_path(current_session_dir, current_filter)

        emit("progress", "calibration_masters", params={
            "filter":  current_filter,
            "dark":    bool(master_dark_path),
            "flat":    bool(master_flat_path),
            "bias":    bool(master_bias_path),
        })

        first_file_for_filter = sorted(files_by_filter[current_filter])[0]
        filter_is_color = is_color_camera(first_file_for_filter)
        # Reclassify L/LUMINANCE as CLEAR for OSC cameras:
        # OSC cameras don't do dedicated LRGB luminance — 'L' filter means
        # "unfiltered" (= CLEAR). Only mono cameras produce true L channels.
        effective_filter = current_filter
        if filter_is_color and current_filter in ("LUMINANCE", "L"):
            effective_filter = "CLEAR"
            debug(f"Reclassified {current_filter} → CLEAR (OSC camera, unfiltered)")

        emit("progress", "sensor_type", params={
                       
            "filter":           current_filter,
            "effective_filter": effective_filter,
            "type":             "color" if filter_is_color else "mono",
        })


        filter_work_dir = current_session_dir / f"work_{current_filter}"
        filter_work_dir.mkdir(parents=True, exist_ok=True)

        num_files = 0
        for i, src_file in enumerate(sorted(files_by_filter[current_filter]), start=1):
            FITS_EXTENSIONS = {'.fits', '.fit', '.fts'}
            src_ext = src_file.suffix.lower()
            dst_ext = ".fit" if src_ext in FITS_EXTENSIONS else src_ext
            dst_name = f"light{i:05d}{dst_ext}"
            dst_file = filter_work_dir / dst_name
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            try:
                dst_file.symlink_to(src_file.resolve())
            except OSError:
                shutil.copy(src_file, dst_file)
            num_files += 1

        # Generate and execute stacking script
        emit("progress", "siril_script_started", params={"filter": current_filter, "frames": num_files})
        # Determine cross-channel stretch reference: RGB and narrowband
        # channels use their group minimum (computed above) so siblings land
        # in the same targetbg bucket; anything else (L/CLEAR/single-filter
        # narrowband) just uses its own frame count.
        if effective_filter in {"RED", "GREEN", "BLUE"} and rgb_group_min_files is not None:
            stretch_reference_files = rgb_group_min_files
        elif effective_filter in NARROWBAND_FILTERS and nb_group_min_files is not None:
            stretch_reference_files = nb_group_min_files
        else:
            stretch_reference_files = num_files

        stack_script = generate_siril_stack_script(
            filter_work_dir,
            effective_filter,
            num_files,
            filter_is_color,
            detected_filters,
            master_dark_path,
            master_flat_path,
            master_bias_path,
            output_bits=output_bits,
            light_files=sorted(files_by_filter[current_filter]),
            stretch_reference_files=stretch_reference_files,
        )
        success = run_siril_command(
            current_session_dir,
            stack_script,
            f"stack_{current_filter}.ssf",
            work_dir=filter_work_dir
        )

        emit("progress", "siril_script_done", params={"filter": current_filter, "success": success})

        fits_count = len(list(filter_work_dir.glob("*.fit")))
        seq_count  = len(list(filter_work_dir.glob("*.seq")))
        debug(f"=== work_{current_filter}: {fits_count} FITS, {seq_count} SEQ ===")

        stacked_file = current_session_dir / f"stacked_{effective_filter}.fit"
        if stacked_file.exists():
            size_kb = stacked_file.stat().st_size / 1024
            debug(f"✅ Stacked file found: {stacked_file.name} ({size_kb:.1f} KB)")
        else:
            debug(f"❌ Missed stacked file: {stacked_file.name}")
            success = False

        if not success:
            emit("error", "stacking_failed", params={"filter": current_filter})
            if filter_work_dir.is_dir():
                shutil.rmtree(filter_work_dir)
            continue

        emit("progress", "stacking_done", params={"filter": current_filter})

        siril_default_fit = current_session_dir / f"stacked_{effective_filter}.fit"
        custom_fit_name   = current_session_dir / f"{file_prefix}_{current_filter}.fit"

        if siril_default_fit.is_file():
            if custom_fit_name.exists():
                custom_fit_name.unlink()
            siril_default_fit.rename(custom_fit_name)
            master_files_map[current_filter] = custom_fit_name
        else:
            debug(f"⚠️ stacked_{effective_filter}.fit not found after stacking")

        if filter_work_dir.is_dir():
            shutil.rmtree(filter_work_dir)

    # 4. Crucial step: Global cross-filter alignment
    if len(master_files_map) > 1:
        emit("progress", "inter_filter_alignment_started", params={
              
            "total":   len(master_files_map),
            "message": "Global geometric alignment...",
        })
        ref_candidate = master_files_map.get('HA') or list(master_files_map.values())[0]

        # Define list of files to align
        files_to_align = [f for f in master_files_map.values() if f != ref_candidate]

        if align_channels(current_session_dir, files_to_align, ref_image=ref_candidate):
            for filter_k, fit_path in master_files_map.items():
                aligned = fit_path.parent / f"{fit_path.stem}_aligned.fit"
                if aligned.exists():
                    master_files_map[filter_k] = aligned
        else:
            emit("warning", "inter_filter_alignment_failed", params={"message": "Failed, using raws"})

     # 5. Final extraction of FITS containers to linear TIFF images for each validated channel
    for current_filter, final_fit_path in master_files_map.items():
        emit("progress", "tiff_extraction_started", params={"filter": current_filter})
        debug(f"Finalized linear master TIFF extraction : {current_filter}")
        # Temporarily go through the final tree to generate conversion .ssf
        conv_script = generate_siril_script(
            current_session_dir,
            current_filter,
            file_prefix,
            fit_path=final_fit_path
        )
        run_siril_command(current_session_dir, conv_script, f"conv_{current_filter}.ssf")
        emit("progress", "tiff_extraction_done", params={"filter": current_filter})

    # 6. Mapping of generated TIFF files
    tif_mapped_files = {}
    for current_filter in detected_filters:
        target_tiff = current_session_dir / f"{file_prefix}_{current_filter}.tif"
        if target_tiff.is_file():
            tif_mapped_files[current_filter] = target_tiff

    if not tif_mapped_files:
        debug("Critical failure collecting intermediate TIFF matrices.")
        emit("error", "tiff_mapping_failed", params={"detail": "No TIFF files found after extraction"})
        emit("done", params={"uuid": session_uuid, "output_format": format_requested})
        return False

    # 7. Final composition via ImageMagick
    emit("progress", "composition_started", params={"format": format_requested, "channels": list(tif_mapped_files.keys())})
    composite_success = compose_rgb_image(current_session_dir, tif_mapped_files, format_requested, file_prefix)

    emit("progress", "cleanup")
    try:
        cleanup_session(current_session_dir)
    except Exception as e:
        debug(f"Final clean-up failed : {e}")

    if composite_success:
        work_image, final_image = get_composition_paths(current_session_dir, file_prefix, format_requested)
        if work_image.is_file():
            emit("progress", "orientation_correction", params={"file": work_image.name})
            last_stacked = next(iter(master_files_map.values()), None)
            correct_image_orientation(work_image, fits_source_path=last_stacked, final_path=final_image)

            end = datetime.now()
            elapsed = end - start
            emit("done", "composition_done", params={
                "uuid":          session_uuid,
                "output_format": format_requested,
                "file_prefix":   file_prefix,
                "final_file":    final_image.name,
                "elapsed_time":  str(elapsed), # Format "H:MM:SS.ms"
                "total_seconds": elapsed.total_seconds(),
            })

            for tiff_path in tif_mapped_files.values():
                if tiff_path.is_file():
                    tiff_path.unlink()
    else:
        emit("error", "composition_failed", params={"detail": "Final composition failed"})

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking pipeline")
    parser.add_argument("uuid",      help="Unique session UUID/directory")
    parser.add_argument("--format",  default="png",     choices=["png", "jpg", "tiff", "webp"], help="Final file encoding format")
    parser.add_argument("--dso",     default="unknown",  help="Target celestial object name (e.g., ngc2359, m42)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debug log")

    args     = parser.parse_args()
    VERBOSE  = args.verbose
    DSO_NAME = args.dso

    try:
        success = run(args)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Processing runner failure : {e}")
        traceback.print_exc()
        sys.exit(1)
