#!/usr/bin/env python3
"""
Siril stacking script for EAA & SaaS.

Generates and runs Siril scripts for:
- Automatically detect sensor type (Mono vs Color) via FITS Header.
- Isolate and stack raws by filter (HA, SII, OIII, RED, CLEAR, etc.).
- Clean spaces in filenames to immunize Siril parser.
- Use native Siril 1.2+ commands ('cd', 'convert -out=.', 'stack r_light rej').
- Properly convert FIT containers to master TIFFs (via load/save).
- Merge channels into a chromatic composite (SHO, HOO, RGB, Mono) via ImageMagick.
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
from datetime import datetime
from astropy.io import fits
from PIL import Image
import numpy as np

# Root directory
BASE_DIR = Path(__file__).resolve().parent

# Valid filters accepted and recognized by the system
VALID_FILTERS = [
    'IR_CUT', 'UV_IR_CUT', 'UHC', 'CLS', 'BROADBAND', 'DUAL_NARROWBAND',
    'LRGB', 'LUMINANCE', 'RED', 'GREEN', 'BLUE', 'RGB', 'HA', 'H_BETA',
    'OIII', 'SII', 'SOLAR', 'CLEAR'
]

def debug(message: str):
    if VERBOSE:
        print(f"[DEBUG] {message}", flush=True)

def emit(status: str, data: dict = None, params: dict = None):
    """Emit a JSON message to stderr for IPC with Symfony API."""
    import json
    payload = {"status": status}
    if data: payload["data"] = data
    if params: payload["params"] = params
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

# --------------------------------------------------------------------------
# FONCTIONS DE PIPELINE (Renommées)
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
    header = get_fits_header(fits_path)
    if not header:
        debug(f"⚠️ No headers FITS, default MONO")
        return False

    debug(f"=== Détection Capteur ===")
    debug(f"  INSTRUME: {header.get('INSTRUME', 'N/A')}")
    debug(f"  BAYERPAT: {header.get('BAYERPAT', 'N/A')}")
    debug(f"  XBAYROFF: {header.get('XBAYROFF', 'N/A')}")
    debug(f"  CFAHEADER: {header.get('CFAHEADER', 'N/A')}")
    debug(f"  COLOR: {header.get('COLOR', 'N/A')}")

    if 'BAYERPAT' in header or 'XBAYROFF' in header or 'CFAHEADER' in header:
        debug(f"  → Result: COLOR (Bayer detected)")
        return True

    instrument = header.get('INSTRUME', '').upper()
    if 'MC' in instrument and 'MM' not in instrument:
        debug(f"  → Result: COLOR (instrument MC)")
        return True

    if header.get('COLOR', '') == 'YES':
        debug(f"  → Result: COLOR (keyword COLOR=YES)")
        return True

    return False

def get_fits_bitdepth(fits_path: Path) -> int:
    """
    Détecte la profondeur de bits via BITPIX du header FITS.
    BITPIX=16 → 16-bit entier signé
    BITPIX=-32 → float32, BITPIX=-64 → float64 → 32-bit
    """
    try:
        with fits.open(fits_path, mode='readonly', ignore_missing_end=True) as hdul:
            bitpix = hdul[0].header.get('BITPIX', -32)
            detected = 16 if bitpix == 16 else 32
            debug(f"BITPIX={bitpix} → {detected}-bit")
            return detected
    except Exception as e:
        debug(f"Impossible de lire le bit depth de {fits_path}: {e}")
        return 32

# --------------------------------------------------------------------------
# DOF (DARKS, FLATS, BIAS)
# --------------------------------------------------------------------------
def ensure_2d_master(master_path: Path) -> Path | None:
    """Ensure the master is in the correct 2D FITS geometric format (Mono or CFA) for Siril CLI."""
    if not master_path.exists():
        return None

    output_path = master_path.parent / f"{master_path.stem}_2d.fit"

    if output_path.exists():
        try:
            output_path.unlink()
        except Exception as e:
            debug(f"Unable to delete {output_path} : {e}")
            return None

    try:
        with fits.open(master_path) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data
            original_dtype = data.dtype
        # Si l'image a été lue ou sauvée par erreur en RGB (3D)
        if data.ndim == 3:
            data = np.mean(data, axis=-1).astype(original_dtype)  # Fusion propre en intensité pure
            header.add_comment('Master normalized to 2D structure')
        elif data.ndim != 2:
            debug(f"Invalid image structure for calibration: {data.ndim} dimensions")
            return None

        header['NAXIS'] = 2
        if 'NAXIS3' in header:
            del header['NAXIS3']

        fits.writeto(output_path, data, header, overwrite=True)
        return output_path
    except Exception as e:
        debug(f"Failed to normalize FITS 2D for {master_path.name} : {e}")
        return None

# def get_master_dark_path(session_dir: Path, filter_name: str = None) -> str | None:
#     """
#     Look for a master dark. Prioritize filter-specific master, then generic master_dark.
#     Supports naming patterns like: masterDark_*_FILTER-{filter}_*.fit(s)
#     """
#     darks_dir = session_dir / "darks"
#     if not darks_dir.is_dir():
#         return None
#
#     def resolve_master(path: Path) -> str | None:
#         """Apply ensure_2d_master and return resolved path string, or None."""
#         m2d = ensure_2d_master(path)
#         if m2d:
#             return str(m2d.resolve())
#         debug(f"ensure_2d_master failed for {path.name}, using original")
#         return str(path.resolve())
#
#     def find_by_pattern(pattern: str) -> Path | None:
#         """Return the most recently modified match for a glob pattern, or None."""
#         matches = sorted(
#             darks_dir.glob(pattern),
#             key=lambda p: p.stat().st_mtime,
#             reverse=True
#         )
#         return matches[0] if matches else None
#
#     # 1. Filter-specific master dark (case variants: HA, Ha, ha)
#     if filter_name:
#         variants = {filter_name, filter_name.upper(), filter_name.lower(), filter_name.capitalize()}
#         for variant in variants:
#             for ext in ('.fit', '.fits'):
#                 match = find_by_pattern(f"*FILTER-{variant}*{ext}")
#                 if match:
#                     debug(f"Filter-specific dark found: {match.name}")
#                     return resolve_master(match)
#
#     # 2. Generic master dark
#     for ext in ('.fit', '.fits'):
#         generic = darks_dir / f"master_dark{ext}"
#         if generic.exists():
#             debug(f"Generic dark found: {generic.name}")
#             return resolve_master(generic)
#
#     debug(f"No master dark found in {darks_dir}")
#     return None
#
# def get_master_flat_path(session_dir: Path, filter_name: str) -> str | None:
#     """
#     Look for a master flat. Prioritize a generic master_flat,
#     otherwise search for a file containing the filter name in its name.
#     Exclude temporary gradient files *_2d.fit.
#     """
#     flats_dir = session_dir / "flats"
#     if not flats_dir.is_dir():
#         return None
#
#     # 1. Search for a direct generic master flat
#     for ext in ['.fit', '.fits']:
#         master_file = flats_dir / f"master_flat{ext}"
#         if master_file.exists():
#             m2d = ensure_2d_master(master_file)
#             return str(m2d.resolve()) if m2d else str(master_file.resolve())
#
#     # 2. Search by filter name match
#     for ext in ['.fit', '.fits']:
#         filter_pattern = f"*{filter_name}*{ext}"
#         # Filter results to ignore residuals _2d.fit
#         matches = [f for f in flats_dir.glob(filter_pattern) if not f.name.endswith(f"_2d{ext}")]
#
#         ## Fallback in case filter is written differently (e.g., ha instead of HA)
#         if not matches:
#             matches = [f for f in flats_dir.glob(f"*{filter_name.lower()}*{ext}") if not f.name.endswith(f"_2d{ext}")]
#
#         if matches:
#             # If the found file is a single raw (doesn't contain 'master')
#             if "master" not in matches[0].name.lower():
#                 return str((flats_dir / f"master_flat_{filter_name}.fit").resolve())
#
#             m2d = ensure_2d_master(matches[0])
#             return str(m2d.resolve()) if m2d else str(matches[0].resolve())
#
#     return None
#
#
# def get_master_bias_path(session_dir: Path) -> str | None:
#     """
#     Look for a standard master bias (offset) in the dedicated subdirectory.
#     Exclude temporary gradient files *_2d.fit.
#     """
#     bias_dir = session_dir / "bias"
#     if not bias_dir.is_dir():
#         return None
#
#     for ext in ['.fit', '.fits']:
#         master_file = bias_dir / f"master_bias{ext}"
#         if master_file.exists():
#             m2d = ensure_2d_master(master_file)
#             return str(m2d.resolve()) if m2d else str(master_file.resolve())
#
#     # Fallback: if there are single FITS files but no 'master_bias.fit'
#     for ext in ['.fit', '.fits']:
#         all_fits = [f for f in bias_dir.glob(f"*{ext}") if not f.name.endswith(f"_2d{ext}")]
#         if all_fits:
#             # If the first file found doesn't have 'master' in its name, schedule its creation
#             if "master" not in all_fits[0].name.lower():
#                 return str((bias_dir / "master_bias.fit").resolve())
#
#             m2d = ensure_2d_master(all_fits[0])
#             return str(m2d.resolve()) if m2d else str(all_fits[0].resolve())
#
#     return None

def _find_master_dof(
    dof_dir: Path,
    generic_name: str,
    filter_name: str | None = None
) -> str | None:
    """
    Generic DOF master finder (dark, flat, bias).
    Prioritizes filter-specific master, then generic master_{type}.
    Excludes _2d files (already processed intermediates).
    Matches case variants: HA / Ha / ha / Ha (capitalize).
    Returns the most recently modified match.
    """
    if not dof_dir.is_dir():
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

    # 1. Filter-specific (4 variantes de casse)
    if filter_name:
        variants = {filter_name, filter_name.upper(), filter_name.lower(), filter_name.capitalize()}
        for variant in variants:
            for ext in ('.fit', '.fits'):
                match = find_by_pattern(f"*FILTER-{variant}*{ext}")
                if match:
                    debug(f"Filter-specific {generic_name} found: {match.name}")
                    return resolve_master(match)

    # 2. Generic fallback : master_dark.fit / master_flat.fit / master_bias.fit
    for ext in ('.fit', '.fits'):
        generic = dof_dir / f"{generic_name}{ext}"
        if generic.exists():
            debug(f"Generic {generic_name} found: {generic.name}")
            return resolve_master(generic)

    debug(f"No {generic_name} found in {dof_dir}")
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
def get_subsky_command(
    master_dark_path: str = None,
    master_flat_path: str = None,
    master_bias_path: str = None
) -> str:
    """
    Adjust subsky parameters according to the available calibration quality.
    Optimized to clean residual vignetting in corners in the absence of Flat,
    while preserving the center of the image (no dark halo).
    """

    missing = []
    if not master_dark_path: missing.append("dark")
    if not master_flat_path: missing.append("flat")
    if not master_bias_path: missing.append("bias")

    debug(f"=== Subsky Masters Check ===")
    debug(f"  Master Dark:  {'SET' if master_dark_path else 'MISSING'} ({master_dark_path})")
    debug(f"  Master Flat:  {'SET' if master_flat_path else 'MISSING'} ({master_flat_path})")
    debug(f"  Master Bias:  {'SET' if master_bias_path else 'MISSING'} ({master_bias_path})")
    debug(f"  Missing: {missing}")

    has_flat = "flat" not in missing
    nb_missing = len(missing)

    # Base parameters depending on flat availability
    if has_flat:
        base_cmd = 'subsky -rbf'
        common_params = '-smooth=0.4 -samples=50'
        tolerance = 1.2 if nb_missing == 0 else 1.4
        return f'{base_cmd} -tolerance={tolerance} {common_params}'
    else:
        base_cmd = 'subsky'
        degree = 3
        if nb_missing == 1:
            tolerance, smooth, samples = 1.4, 0.70, 60
        elif nb_missing == 2:
           tolerance, smooth, samples, degree = 1.8, 0.5, 65, 2
        else:
            tolerance, smooth, samples = 1.6, 0.85, 70

        emit("progress", data={"degree": degree,"tolerance": tolerance, "smooth": smooth, "samples": samples})
        return f'{base_cmd} {degree} -tolerance={tolerance} -smooth={smooth} -samples={samples}'

def get_color_calibration_command(is_color: bool) -> str:
    """
    Generate the best possible color calibration command.
    If PCC is possible and requested, use it. Otherwise, fallback to local 'cc'.
    """
    if not is_color:
        return ""

    # Mode SaaS Premium : Tentative de PCC si on a un objet et des fichiers valides
#     if light_files and dso_name.lower() not in ["unknown", ""]:
#         try:
#             with fits.open(light_files[0], mode='readonly', ignore_missing_end=True) as hdul:
#                 header = hdul[0].header
#                 focal = header.get('FOCALLEN')
#                 pixel_size = header.get('XPIXSZ') or header.get('PIXSIZE')
#
#                 if focal and pixel_size:
#                     # Étalonnage par Photométrie (Optionnel et conditionnel)
#                     return f"pcc -cc={dso_name.upper()} -focal={int(focal)} -pixel={float(pixel_size)} -server=simbad"
#         except Exception as e:
#             debug(f"Métadonnées incomplètes pour le PCC ({e}), bascule sur l'étalonnage local.")

    # FALLBACK LOCAL : Étalonnage des couleurs standard (Siril 1.2)
    # Détecte le fond du ciel et balance les blancs de manière itérative et locale
    return ""

def get_rmgreen_command(is_color: bool) -> str:
    """
    Generate the noise removal command for green (SCNR).
    Only relevant on color sensors.
    """
    return "rmgreen 0.3" if is_color else ""

def _seq_detect_block(seq_prefix: str) -> list:
    """Génère un bloc de détection de séquence Siril."""
    return [
        f'set seq_name "{seq_prefix}"',
        f'if [ -f "{seq_prefix}_.seq" ]; then',
        f'    set seq_name "{seq_prefix}_"',
        'endif',
        'load $seq_name'
    ]
# --------------------------------------------------------------------------
# GENERATE NATIVE SIRIL SCRIPTS (.SSF)
# --------------------------------------------------------------------------
def generate_siril_stack_script(
    filter_work_dir: Path,
    filter_name: str,
    num_files: int,
    is_color: bool,
    master_dark_path: str = None,
    master_flat_path: str = None,
    master_bias_path: str = None,
    output_bits: int = 32
) -> str:
    """
    Generates the .ssf script for Siril.
    num_files: used to validate that we have at least 2 images for the stack
    """
    abs_work_dir = filter_work_dir.resolve().as_posix()
    bit_cmd = "set32bits" if output_bits == 32 else "set16bits"

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

    # 2. Pré-traitement (Dématriçage)
    if is_color:
        lines.append(f"preprocess {seq} -debayer")
        seq = f"pp_{seq}"

    # 3. Registration & Stack
    lines.extend([
        f'register {seq}',
        f'stack r_{seq} rej winsorized 3 3 -norm=add -weight_from_noise',
        f'load r_{seq}_stacked.fit',
        get_subsky_command(master_dark_path, master_flat_path, master_bias_path),
    ])

    # 4. Post-processing (linear data, before stretch)
    lines.extend([
        get_rmgreen_command(is_color),
        "autostretch",
        bit_cmd,
        f'save "../stacked_{filter_name}.fit"',
        "close",
        "exit"
    ])

    return "\n".join(lines)

def generate_siril_script(session_dir: Path, filter_name: str, file_prefix: str) -> str:
    """
    Generate the intermediate FIT to TIFF conversion script.
    Uses the native 'savetif' command from Siril 1.2 to avoid erroneous hybrid files like '.tif.fit'.
    """
    fit_path = (session_dir / f"{file_prefix}_{filter_name}.fit").as_posix()
    tif_path = (session_dir / f"{file_prefix}_{filter_name}").as_posix()

    return "\n".join([
        "requires 1.2.0",
        f'load "{fit_path}"',
        f'savetif "{tif_path}"',
        "close",
        "exit"
    ])

# --------------------------------------------------------------------------
# CORE ENGINE EXECUTION (SIRIL-CLI)
# --------------------------------------------------------------------------
def run_siril_command(session_dir: Path, script_content: str, script_name: str, work_dir: Path = None) -> bool:
    """Execute a user Siril script with siril-cli."""
    script_path = session_dir / script_name
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    cmd = ["siril-cli", "-s", str(script_path)]
    effective_cwd = str(work_dir) if work_dir else str(session_dir)
    debug(f"Check effective_cwd: {effective_cwd}")

    debug(f"=== Script Siril ({script_name}) ===")
    debug(script_content)
    debug(f"=== Fin du Script ===")

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
        return process.returncode == 0
    except Exception as e:
        debug(f"Fatal error running siril-cli: {e}")
        return False
    finally:
        if script_path.exists():
            script_path.unlink()

# --------------------------------------------------------------------------
# CHROMINANCE & COMPOSITION VIA IMAGEMAGICK
# --------------------------------------------------------------------------
def compose_rgb_image(session_dir: Path, tif_files: dict, output_format: str, file_prefix: str) -> bool:
    """Combine normalized TIFF files and handle assembly palettes (LRVB / SHO / HOO / HOO+RGB)."""
    output_file = session_dir / f"{file_prefix}_full.{output_format}"

    # 1. Determine reference geometry for black areas (xc:black)
    ref_path = next(iter(tif_files.values()))
    width, height = get_image_dimensions(ref_path)

    # Single channel (Mono or simple raw extraction)
    if len(tif_files) == 1:
        single_channel = list(tif_files.values())[0]
        cmd = ["convert", str(single_channel)]
        if output_format in ["webp", "jpg"]:
            cmd.extend(["-quality", "95"])
        cmd.append(str(output_file))
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.returncode == 0
        except Exception as e:
            debug(f"ImageMagick Single Channel Failure: {e}")
            return False

    # --- STEP 1: DETERMINE ASSEMBLY MODE ---
    cmd = ["convert"]

    # Check presence of filter blocks
    has_ha = "HA" in tif_files
    has_oiii = "OIII" in tif_files
    has_sii = "SII" in tif_files
    has_rgb = "RED" in tif_files and "GREEN" in tif_files and "BLUE" in tif_files

    # SPECIAL CASE ONLY: SHO + RGB (Advanced mixing for colored stars)
    if has_sii and has_ha and has_oiii and has_rgb:
        mix_channels = [
            (tif_files["SII"], tif_files["RED"]),
            (tif_files["HA"], tif_files["GREEN"]),
            (tif_files["OIII"], tif_files["BLUE"])
        ]
        for nb_file, rgb_file in mix_channels:
            cmd.extend(["(", str(nb_file), str(rgb_file), "-blend", "80x20", ")"])

    # STANDARD CASES (SHO, HOO pure, or classic RGB)
    else:
        # Default assignment / classic RGB
        r_channel = tif_files.get("RED", tif_files.get("HA", "xc:black"))
        g_channel = tif_files.get("GREEN", tif_files.get("OIII", "xc:black"))
        b_channel = tif_files.get("BLUE", tif_files.get("SII", "xc:black"))

        # Narrowband palette mapping
        if has_sii and has_ha and has_oiii:
            r_channel, g_channel, b_channel = tif_files["SII"], tif_files["HA"], tif_files["OIII"]
        elif has_ha and has_oiii:
            r_channel, g_channel, b_channel = tif_files["HA"], tif_files["OIII"], tif_files["OIII"]

        # Add channels to command with dynamic size correction
        for channel in [r_channel, g_channel, b_channel]:
            if channel == "xc:black":
                cmd.extend(["-size", f"{width}x{height}", "xc:black"])
            else:
                cmd.append(str(channel))

    # --- STEP 2: FINALIZATION AND EXECUTION ---
    cmd.extend([
        "-despeckle",
        "-level", "2%,98%,1.1",
        "-combine",
        "-colorspace", "sRGB",
        "-noise", "1",
        "-unsharp", "0x1.5+1+0.002"
    ])
    if output_format in ["webp", "jpg"]:
        cmd.extend(["-quality", "95"])
    cmd.append(str(output_file))

    debug(f"Running ImageMagick chrominance synthesis: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            debug(f"ImageMagick STDERR Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        debug(f"ImageMagick composite assembly failure: {e}")
        return False

def get_image_dimensions(ref_path: Path) -> tuple:
    """Get real dimensions of the master FITS for ImageMagick."""
    try:
        with Image.open(ref_path) as img:
            return img.size # Retourne (width, height)
    except Exception:
        # Fallback de secours si PIL échoue
        debug(f"Warning: Could not read dimensions of {ref_path}, fallback 2048x2048")
        return (2048, 2048)

def correct_image_orientation(image_path: Path):
    """Vertically flip the final image to compensate for FITS coordinate system (origin at bottom-left)."""
    try:
        subprocess.run(["convert", str(image_path), "-flip", str(image_path)], check=True)
    except Exception as e:
        debug(f"Orientation correction failure: {e}")

# --------------------------------------------------------------------------
# IMAGE ALIGNMENT WITH SIRIL-CLI
# --------------------------------------------------------------------------
def align_channels(session_dir: Path, images_to_align: list[Path], ref_image: Path = None) -> bool:
    """
    Properly align images to a reference via a single .ssf script.
    """
    if not images_to_align or not ref_image:
        return False

    # Prepare the script file to be executed in a single Siril instance
    script_path = session_dir / "inter_filter_align.ssf"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(f'cd "{session_dir.as_posix()}"\n')
        for img in images_to_align:
            if img == ref_image:
                continue
            output_aligned = img.parent / f"{img.stem}_aligned.fit"
            f.write(f'load "{ref_image.as_posix()}"\n')
            f.write(f'register "{img.as_posix()}"\n')
            f.write(f'save "{output_aligned.as_posix()}"\n')
        f.write('close\n')

    # Exécution unique
    cmd = ["siril-cli", "-s", str(script_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if script_path.exists():
        script_path.unlink()

    return result.returncode == 0

# --------------------------------------------------------------------------
# IMAGE ALIGNMENT WITH ASTROALIGN (NOT CALLED)
# --------------------------------------------------------------------------
def align_images_with_astroalign(session_dir, filter_name, reference_path):
    """Align all images of a filter to a reference with Astroalign.
    """
    try:
        import astroalign as aa
        from astropy.nddata import CCDData
    except ImportError:
        debug("Astroalign not installed. Use 'pip install astroalign' to enable it.")
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
            return False
    debug(f"Astroalign alignment completed for {filter_name}")
    return True

# --------------------------------------------------------------------------
# COORDINATION AND MAIN RUNNER
# --------------------------------------------------------------------------
def cleanup_session(session_dir: Path):
    """Clean all temporary files, including converted masters and gradient residuals."""
    temp_patterns = [
        "*.tmp", "*.log", "*.pid", "*.lock", "*.txt",
        "work_*/",           # Répertoires de travail
#        "stacked_*.fit",     # Masters empilés
        "stacked_*.tif",     # Masters convertis en TIFF
        "r_pp_*.fit",        # Séquences calibrées
        "pp_*.fit",          # Pré-calibration
        "r_light_*.fit",     # Lights alignés
        "light_*.fit",       # Lights convertis
        "master_*_2d.fit",   # ✅ Modèles de gradient résiduels générés par subsky
        "*.ssf"              # Scripts Siril temporaires
    ]

    import glob
    import shutil

    # Nettoyer dans le répertoire principal et les sous-dossiers DOF
    base_dirs = [
        session_dir,
        session_dir / "darks",
        session_dir / "bias",
        session_dir / "flats"
    ]

    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue
        for pattern in temp_patterns:
            matches = list(base_dir.glob(pattern))
            for match in matches:
                try:
                    if match.is_file():
                        match.unlink()
                        debug(f"Temporary file deleted: {match}")
                    elif match.is_dir():
                        shutil.rmtree(match)
                        debug(f"Temporary directory deleted: {match}")
                except Exception as e:
                    debug(f"Failed to delete {match}: {e}")

def run(args) -> bool:
    session_uuid = args.uuid
    local_sessions_root = BASE_DIR / "sessions"
    current_session_dir = local_sessions_root / session_uuid

    lights_dir = current_session_dir / "lights"
    format_requested = args.format.lower()
    dso_name = re.sub(r'[^a-zA-Z0-9_-]', '', args.dso.lower().replace(" ", ""))

    first_light = None
    for f in lights_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.fits', '.fit', '.fts']:
            first_light = f
            break
    output_bits = 32
    if first_light:
        output_bits = get_fits_bitdepth(first_light)
        debug(f"Detected Bit depth from {first_light.name}: {output_bits}-bit")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_prefix = f"{dso_name}_{timestamp}"

    cleanup_session(current_session_dir)

    if not lights_dir.is_dir():
        emit("error", params={"detail": "Lights directory not found"})
        return False

    # 1. Sort and index files by detected filter
    emit("progress", data={"step": "start", "message": f"Processing started for {dso_name}"})
    debug(f"Analyzing raws for {dso_name.upper()} | Session: {current_session_dir.name}")

    files_by_filter = {}
    for f in lights_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.fits', '.fit', '.fts']:
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

                    if filter_keyword in filter_map:
                        matched_filter = filter_map[filter_keyword]
            except Exception as e:
                debug(f"Failed to read FITS header: {f.name} -> {matched_filter} (original: {header.get('FILTER', 'N/A')}) / {e}")

            # Step 2: If step 1 failed or no valid filter found, try with filename
            if matched_filter is None:
                filename_upper = f.name.upper()
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
        emit("error", params={"detail": "No valid FITS files found"})
        return False

    PRIORITY = {
        'HA': 0, 'OIII': 1, 'SII': 2,
        'RED': 3, 'GREEN': 4, 'BLUE': 5,
        'LUMINANCE': 6, 'CLEAR': 7
    }
    detected_filters = sorted(detected_filters, key=lambda f: PRIORITY.get(f, 99))
    emit("progress", data={"step": "filters_detected", "filters": detected_filters})

    # 2. Sensor type diagnostic on the first available raw
    first_filter_key = detected_filters[0]
    first_fits_file = files_by_filter[first_filter_key][0]
    camera_is_color = is_color_camera(first_fits_file)
    emit("progress", data={"step": "sensor_type", "type": "color" if camera_is_color else "mono"})

    # Dictionary for storing generated master FITS for final cross-alignment
    master_files_map = {}

    # 3. Individual processing and stacking of channels in Siril
    for current_filter in detected_filters:
        emit("progress", data={"step": "stacking_started", "filter": current_filter})

        master_dark_path = get_master_dark_path(current_session_dir, current_filter)
        master_flat_path = get_master_flat_path(current_session_dir, current_filter)
        master_bias_path = get_master_bias_path(current_session_dir, current_filter)

        emit("progress", data={
            "step": "calibration_status",
            "masters": {
                "dark": bool(master_dark_path),
                "flat": bool(master_flat_path),
                "bias": bool(master_bias_path)
             }
        })

        filter_work_dir = current_session_dir / f"work_{current_filter}"
        filter_work_dir.mkdir(parents=True, exist_ok=True)

        num_files = 0
        for i, src_file in enumerate(sorted(files_by_filter[current_filter]), start=1):
#             clean_name = src_file.name.replace(" ", "_")
            dst_name = f"light{i:05d}.fit"
            dst_file = filter_work_dir / dst_name
            if not dst_file.exists():
                try:
                    dst_file.symlink_to(src_file.resolve())
                except OSError:
                    shutil.copy(src_file, dst_file)
            num_files += 1

        # Generate and execute stacking script
        stack_script = generate_siril_stack_script(
            filter_work_dir,
            current_filter,
            num_files,
            camera_is_color,
            master_dark_path,
            master_flat_path,
            master_bias_path,
            output_bits=output_bits
        )
        success = run_siril_command(
            current_session_dir,
            stack_script,
            f"stack_{current_filter}.ssf",
            work_dir=filter_work_dir
        )

        debug(f"=== Files in {filter_work_dir} ===")
        for f in filter_work_dir.glob("*.fit"):
            debug(f"  FITS: {f.name}")
        for f in filter_work_dir.glob("*.seq"):
            debug(f"  SEQ: {f.name}")
        debug(f"============================")

        stacked_file = current_session_dir / f"stacked_{current_filter}.fit"
        if stacked_file.exists():
            debug(f"✅ Stacked file found: {stacked_file.name} ({stacked_file.stat().st_size} bytes)")
        else:
            debug(f"❌ Missed stacked file: {stacked_file.name}")
            success = False  # ✅ Forcer l'échec si pas de fichier

        if not success:
            emit("error", data={"step": "stacking_failed", "filter": current_filter})
            continue

        if filter_work_dir.is_dir():
            shutil.rmtree(filter_work_dir)

        if not success or not stacked_file.exists():
            emit("error", data={"step": "stacking_failed", "filter": current_filter})
            continue
        else:
            emit("progress", data={"step": "stacking_done", "filter": current_filter})

        siril_default_fit = current_session_dir / f"stacked_{current_filter}.fit"
        custom_fit_name = current_session_dir / f"{file_prefix}_{current_filter}.fit"
        
        if siril_default_fit.is_file():
            if custom_fit_name.exists():
                custom_fit_name.unlink()
            siril_default_fit.rename(custom_fit_name)
            master_files_map[current_filter] = custom_fit_name
        else:
            # Safety: If Siril named it differently (e.g., without prefix or already with custom name)
            fallback_fit = current_session_dir / f"{file_prefix}_{current_filter}.fit"
            if fallback_fit.is_file():
                master_files_map[current_filter] = fallback_fit
            else:
                # If the file remained in the work subdirectory
                work_fit = current_session_dir / f"work_{current_filter}" / f"stacked_{current_filter}.fit"
                if work_fit.is_file():
                    shutil.move(work_fit, custom_fit_name)
                    master_files_map[current_filter] = custom_fit_name

        if filter_work_dir.is_dir():
            shutil.rmtree(filter_work_dir)

    # 4. Crucial step: Global cross-filter alignment
    if len(master_files_map) > 1:
        emit("progress", data={"step": "inter_filter_alignment_started", "message": "Recalage géométrique global..."})
        ref_candidate = master_files_map.get('HA') or list(master_files_map.values())[0]

        # Define list of files to align
        files_to_align = [f for f in master_files_map.values() if f != ref_candidate]

        if align_channels(current_session_dir, files_to_align, ref_image=ref_candidate):
            for filter_k, fit_path in master_files_map.items():
                aligned = fit_path.parent / f"{fit_path.stem}_aligned.fit"
                if aligned.exists():
                    master_files_map[filter_k] = aligned
        else:
            emit("warning", data={"step": "inter_filter_alignment_failed", "message": "Failed, using raws"})

     # 5. Final extraction of FITS containers to linear TIFF images for each validated channel
    for current_filter, final_fit_path in master_files_map.items():
        debug(f"Finalized linear master TIFF extraction : {current_filter}")
        # Temporarily go through the final tree to generate conversion .ssf
        conv_script = generate_siril_script(current_session_dir, current_filter, file_prefix)
        run_siril_command(current_session_dir, conv_script, f"conv_{current_filter}.ssf")

    # 6. Mapping of generated TIFF files
    tif_mapped_files = {}
    for current_filter in detected_filters:
        target_tiff = current_session_dir / f"{file_prefix}_{current_filter}.tif"
        if target_tiff.is_file():
            tif_mapped_files[current_filter] = target_tiff

    if not tif_mapped_files:
        debug("Critical failure collecting intermediate TIFF matrices.")
        emit("done", data={"uuid": session_uuid, "output_format": format_requested})
        return False

    # 7. Final composition via ImageMagick
    emit("progress", data={"step": "composition_started", "format": format_requested})
    composite_success = compose_rgb_image(current_session_dir, tif_mapped_files, format_requested, file_prefix)
    
    try:
        cleanup_session(current_session_dir)
    except Exception as e:
        debug(f"Final clean-up failed : {e}")

    if composite_success:
        final_image = current_session_dir / f"{file_prefix}_full.{format_requested}"
        if final_image.is_file():
            correct_image_orientation(final_image)
            emit("done", data={
                "uuid": session_uuid,
                "output_format": format_requested,
                "file_prefix": file_prefix,
                "final_file": final_image.name
            })

            for tiff_path in tif_mapped_files.values():
                if tiff_path.is_file(): 
                    tiff_path.unlink()
    else:
        emit("error", data={"step": "composition_failed", "detail": "Final composition failed"})

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking pipeline")
    parser.add_argument("uuid", help="Unique session UUID/directory")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "tiff", "webp"], help="Final file encoding format")
    parser.add_argument("--dso", default="unknown", help="Target celestial object name (e.g., ngc2359, m42)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display debug log")

    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    try:
        success = run(args)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Rupture du runner de traitement : {e}")
        traceback.print_exc()
        sys.exit(1)
