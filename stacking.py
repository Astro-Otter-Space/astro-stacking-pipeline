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
        logger.error(f"Failed to read FITS header from {fits_path}: {e}")
        return {}

def is_color_camera(fits_path: Path) -> bool:
    """Analyze FITS header to determine if sensor is color (OSC)."""
    header = get_fits_header(fits_path)
    if not header:
        return False

    if 'BAYERPAT' in header or 'XBAYROFF' in header or 'CFAHEADER' in header:
        return True

    instrument = header.get('INSTRUME', '').upper()
    if 'MC' in instrument and 'MM' not in instrument:
        return True

    return False

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

        # Si l'image a été lue ou sauvée par erreur en RGB (3D)
        if data.ndim == 3:
            data = np.mean(data, axis=-1)  # Fusion propre en intensité pure
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

def get_master_dark_path(session_dir: Path, light_files: list[Path]) -> str | None:
    """
    Look for an appropriate master dark. Prioritize a generic master_dark,
    otherwise search by exposure time match (EXPTIME/EXPOSURE).
    """
    darks_dir = session_dir / "darks"
    if not darks_dir.is_dir():
        return None

    # 1. Recherche d'un master dark générique direct
    for ext in ['.fit', '.fits']:
        master_file = darks_dir / f"master_dark{ext}"
        if master_file.exists():
            m2d = ensure_2d_master(master_file)
            return str(m2d.resolve()) if m2d else str(master_file.resolve())

    # 2. Si pas de master générique, extraction du temps de pose de la première brute valide
    exposure = None
    for light in light_files:
        try:
            with fits.open(light, mode='readonly', ignore_missing_end=True) as hdul:
                header = hdul[0].header
                exposure = header.get('EXPTIME') or header.get('EXPOSURE')
                if exposure is not None:
                    exposure = float(exposure)
                    break
        except Exception as e:
            debug(f"Unable to read exposure from {light.name}: {e}")
            continue

    if exposure is None:
        return None

    # 3. Recherche par pattern de temps de pose (ex: EXPOSURE-30.00s ou 30s)
    for ext in ['.fit', '.fits']:
        patterns = [
            f"*EXPOSURE-{exposure:.2f}s*{ext}",
            f"*EXPOSURE-{int(exposure)}s*{ext}",
            f"*{int(exposure)}s*{ext}"
        ]
        for pattern in patterns:
            matches = list(darks_dir.glob(pattern))
            if matches:
                # Take the first match found
                m2d = ensure_2d_master(matches[0])
                return str(m2d.resolve()) if m2d else str(matches[0].resolve())

    return None


def get_master_flat_path(session_dir: Path, filter_name: str) -> str | None:
    """
    Look for a master flat. Prioritize a generic master_flat,
    otherwise search for a file containing the filter name in its name.
    Exclude temporary gradient files *_2d.fit.
    """
    flats_dir = session_dir / "flats"
    if not flats_dir.is_dir():
        return None

    # 1. Search for a direct generic master flat
    for ext in ['.fit', '.fits']:
        master_file = flats_dir / f"master_flat{ext}"
        if master_file.exists():
            m2d = ensure_2d_master(master_file)
            return str(m2d.resolve()) if m2d else str(master_file.resolve())

    # 2. Search by filter name match
    for ext in ['.fit', '.fits']:
        filter_pattern = f"*{filter_name}*{ext}"
        # Filter results to ignore residuals _2d.fit
        matches = [f for f in flats_dir.glob(filter_pattern) if not f.name.endswith(f"_2d{ext}")]

        ## Fallback in case filter is written differently (e.g., ha instead of HA)
        if not matches:
            matches = [f for f in flats_dir.glob(f"*{filter_name.lower()}*{ext}") if not f.name.endswith(f"_2d{ext}")]

        if matches:
            # If the found file is a single raw (doesn't contain 'master')
            if "master" not in matches[0].name.lower():
                return str((flats_dir / f"master_flat_{filter_name}.fit").resolve())

            m2d = ensure_2d_master(matches[0])
            return str(m2d.resolve()) if m2d else str(matches[0].resolve())

    return None


def get_master_bias_path(session_dir: Path) -> str | None:
    """
    Look for a standard master bias (offset) in the dedicated subdirectory.
    Exclude temporary gradient files *_2d.fit.
    """
    bias_dir = session_dir / "bias"
    if not bias_dir.is_dir():
        return None

    for ext in ['.fit', '.fits']:
        master_file = bias_dir / f"master_bias{ext}"
        if master_file.exists():
            m2d = ensure_2d_master(master_file)
            return str(m2d.resolve()) if m2d else str(master_file.resolve())

    # Fallback: if there are single FITS files but no 'master_bias.fit'
    for ext in ['.fit', '.fits']:
        all_fits = [f for f in bias_dir.glob(f"*{ext}") if not f.name.endswith(f"_2d{ext}")]
        if all_fits:
            # If the first file found doesn't have 'master' in its name, schedule its creation
            if "master" not in all_fits[0].name.lower():
                return str((bias_dir / "master_bias.fit").resolve())

            m2d = ensure_2d_master(all_fits[0])
            return str(m2d.resolve()) if m2d else str(all_fits[0].resolve())

    return None
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

    has_flat = "flat" not in missing
    nb_missing = len(missing)

    # Base parameters depending on flat availability
    if has_flat:
        base_cmd = 'subsky -rbf'
        common_params = '-smooth=0.5 -samples=25' if nb_missing == 0 else '-smooth=0.6 -samples=25'
        tolerance = 1.2 if nb_missing == 0 else 1.6
        return f'{base_cmd} -tolerance={tolerance} {common_params}'
    else:
        base_cmd = 'subsky'
        smooth = 0.75
        samples = 35
        if nb_missing == 1:
            degree = 2
            tolerance = 1.2
        elif nb_missing == 2:
            degree = 2
            tolerance = 1.3
            smooth = 0.70
        else:  # nb_missing == 3
            degree = 3
            tolerance = 1.8
            smooth = 0.80
            samples = 40
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
    return "rmgreen" if is_color else ""

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
    master_bias_path: str = None
) -> str:
    """Generate native Siril 1.2.0 stacking instructions (.ssf)."""
    absolute_work_dir = filter_work_dir.resolve()
    debug(f'absolute_work_dir: {absolute_work_dir}')
    lines = [
        "requires 1.2.0",
        f'cd "{absolute_work_dir.as_posix()}"',
        ""
    ]

    # =========================================================================
    # CASE A: SINGLE IMAGE (Single image processing with alignment)
    # =========================================================================
    if num_files == 1:
        debug(f"LINE 339: CASE A: SINGLE IMAGE (Single image processing with alignment)")
        lines.append('# Single image processing with potential alignment')
        lines.append('convert light')

        has_masters = any([master_dark_path, master_flat_path, master_bias_path])
        if has_masters:
            calibrate_parts = ["calibrate light"]
            if master_bias_path: calibrate_parts.append(f"-bias={Path(master_bias_path).as_posix()}")
            if master_dark_path:
                clean_dark = master_dark_path.replace('-dark=', '').replace('"', '')
                calibrate_parts.append(f'-dark={clean_dark}')
            if master_flat_path: calibrate_parts.append(f"-flat={Path(master_flat_path).as_posix()}")
            if is_color: calibrate_parts.append('-cfa')
            lines.append(" ".join(calibrate_parts))

        # Align single image to ensure geometric consistency
        current_file = "pp_light_00001.fit" if has_masters else "light_00001.fit"
        lines.append(f'register "{current_file}"')
        lines.append(f'load r_{current_file}')
        
        if is_color:
            lines.extend(['debayer', 'save pp_light_00001_debayer.fit', 'load pp_light_00001_debayer.fit'])

        subsky_cmd = get_subsky_command(master_dark_path, master_flat_path, master_bias_path)
        if subsky_cmd: lines.append(subsky_cmd)

        lines.extend(["autostretch", f'save "../stacked_{filter_name}.fit"', "close", "exit"])
        return "\n".join(lines)

    # =========================================================================
    # CASE B: STANDARD SEQUENCE (Multiple images)
    # =========================================================================
    # 1. Convert raw monochrome (CFA)
    debug(f"CAS B : SÉQUENCE STANDARD (Multi-images)")
    current_sequence = "light"

    lines.append(f'convert {current_sequence}')

    # 2. Calibration of the 'light' sequence
    debug(f"master_dark_path: {master_dark_path}")
    debug(f"master_flat_path: {master_flat_path}")
    debug(f"master_bias_path: {master_bias_path}")
    has_masters = any([master_dark_path, master_flat_path, master_bias_path])
    debug(f"Any: {has_masters}")
    if has_masters:
        calibrate_parts = ["calibrate light"]
        if master_bias_path:
            calibrate_parts.append(f"-bias={Path(master_bias_path).as_posix()}")
        if master_dark_path:
            path_str = Path(master_dark_path).as_posix()
            clean_dark = path_str.replace('"', '').replace("'", "")
            calibrate_parts.append(f"-dark={clean_dark}")
        if master_flat_path:
            calibrate_parts.append(f"-flat={Path(master_flat_path).as_posix()}")
        if is_color:
            calibrate_parts.extend(['-cfa', '-equalize_cfa'])
        lines.append(" ".join(calibrate_parts))
        current_sequence = "pp_light"

    # 3. De-mosaic the sequence via the preprocess command
    if is_color:
        lines.extend([f"preprocess {current_sequence} -debayer", ""])
        current_sequence = f"pp_{current_sequence}"

    # 4. Alignment (register) and Stacking (stack)
    clean_sequence = current_sequence.rstrip('_')
    lines.extend([
        f'register {clean_sequence}',
        f'stack r_{clean_sequence} rej winsorized 3 3 -norm=add -weight_from_noise',
        f'load r_{clean_sequence}_stacked.fit',
        "",
        get_subsky_command(master_dark_path, master_flat_path, master_bias_path),
        get_rmgreen_command(is_color),
        "autostretch",
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
def run_siril_command(session_dir: Path, script_content: str, script_name: str) -> bool:
    """Execute a user Siril script with siril-cli."""
    script_path = session_dir / script_name
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    cmd = ["siril-cli", "-s", str(script_path)]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(session_dir),
            text=True
        )

        for line in process.stdout:
            if any(k in line.lower() for k in ["error", "failed", "inconnue"]):
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
# def compose_rgb_image(session_dir: Path, tif_files: dict, output_format: str, file_prefix: str) -> bool:
#     """Combine normalized TIFF files and handle assembly palettes (LRVB / SHO / HOO / HOO+RGB)."""
#     output_file = session_dir / f"{file_prefix}_full.{output_format}"
#
#     # Canal unique (Mono ou extraction brute simple)
#     if len(tif_files) == 1:
#         single_channel = list(tif_files.values())[0]
#         cmd = ["convert", str(single_channel)]
#         if output_format in ["webp", "jpg"]:
#             cmd.extend(["-quality", "95"])
#         cmd.append(str(output_file))
#         try:
#             result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             return result.returncode == 0
#         except Exception as e:
#             debug(f"Échec ImageMagick Canal Unique : {e}")
#             return False
#
#     # --- ÉTAPE 1 : DÉTERMINATION DU MODE D'ASSEMBLAGE ---
#     cmd = ["convert"]
#
#     # Vérification de la présence des blocs de filtres
#     has_narrowband = "HA" in tif_files and "OIII" in tif_files
#     has_rgb = "RED" in tif_files and "GREEN" in tif_files and "BLUE" in tif_files
#
#     # CAS SPÉCIAL EXCLUSIF : HOO + RGB (Mixage avancé pour étoiles colorées)
#     if has_narrowband and has_rgb:
#         # On prépare les paires de mixage (Nébuleuse x Étoiles)
#         # R = 80% HA + 20% RED
#         # G = 80% OIII + 20% GREEN
#         # B = 80% OIII + 20% BLUE
#         mix_channels = [
#             (tif_files["HA"], tif_files["RED"]),
#             (tif_files["OIII"], tif_files["GREEN"]),
#             (tif_files["OIII"], tif_files["BLUE"])
#         ]
#
#         for nb_file, rgb_file in mix_channels:
#             # En syntaxe subprocess, les parenthèses d'ImageMagick doivent être des éléments isolés
#             cmd.extend(["(", str(nb_file), str(rgb_file), "-blend", "80x20", ")"])
#
#     # CAS STANDARDS (SHO, HOO pur, ou RVB classique)
#     else:
#         # Assignation par défaut / RVB classique
#         r_channel = tif_files.get("RED", tif_files.get("HA", "xc:black"))
#         g_channel = tif_files.get("GREEN", tif_files.get("OIII", "xc:black"))
#         b_channel = tif_files.get("BLUE", tif_files.get("SII", "xc:black"))
#
#         # Mapping des palettes bandes étroites
#         if "HA" in tif_files and "OIII" in tif_files and "SII" in tif_files:
#             # Palette SHO (Hubble) -> R=SII, G=Ha, B=OIII
#             r_channel = tif_files["SII"]
#             g_channel = tif_files["HA"]
#             b_channel = tif_files["OIII"]
#         elif "HA" in tif_files and "OIII" in tif_files and "SII" not in tif_files:
#             # Palette HOO -> R=Ha, G=OIII, B=OIII
#             r_channel = tif_files["HA"]
#             g_channel = tif_files["OIII"]
#             b_channel = tif_files["OIII"]
#
#         # Ajout des canaux à la commande
#         for channel in [r_channel, g_channel, b_channel]:
#             if channel == "xc:black":
#                 cmd.extend(["-size", f"{ref_path.width}x{ref_path.height}", "xc:black"])
#             else:
#                 cmd.append(str(channel))
#
#     # --- ÉTAPE 2 : FINALISATION ET EXÉCUTION ---
#     cmd.append("-combine")
#     if output_format in ["webp", "jpg"]:
#         cmd.extend(["-quality", "95"])
#     cmd.append(str(output_file))
#
#     debug(f"Exécution de la synthèse chromatique ImageMagick : {' '.join(cmd)}")
#     try:
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if result.returncode != 0:
#             debug(f"Erreur ImageMagick STDERR : {result.stderr}")
#         return result.returncode == 0
#     except Exception as e:
#         debug(f"Échec de l'assemblage composite ImageMagick : {e}")
#         return False

def compose_rgb_image(session_dir: Path, tif_files: dict, output_format: str, file_prefix: str) -> bool:
    """Combine normalized TIFF files and handle assembly palettes (LRVB / SHO / HOO / HOO+RGB)."""
    from PIL import Image
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
        "-median", "1",
        "-level", "2%,98%,1.0",
        "-combine",
        "-colorspace", "sRGB"
    ])
#     cmd.extend(["-level", "1%,100%,1.0", "-despeckle", "-combine", "-colorspace", "sRGB"])
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
        from PIL import Image
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

    # Génération du timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_prefix = f"{dso_name}_{timestamp}"

    cleanup_session(current_session_dir)

    if not lights_dir.is_dir():
        emit("error", params={"detail": "Lights directory not found"})
        return False

    # 1. Sort and index files by detected filter
    emit("progress", data={"step": "start", "message": f"Processing started for {dso_name}"})
    debug(f"Analyzing raws for {dso_name.upper()} | Session: {current_session_dir.name}")

    # 1. Sort and index files by detected filter
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
        emit("error", params={"detail": "Aucun fichier FITS valide trouvé"})
        return False

    PRIORITY = {
        'HA': 0, 'OIII': 1, 'SII': 2,
        'RED': 3, 'GREEN': 4, 'BLUE': 5,
        'LUMINANCE': 6, 'CLEAR': 7
    }
    detected_filters = sorted(detected_filters, key=lambda f: PRIORITY.get(f, 99))
    emit("progress", data={"step": "filters_detected", "filters": detected_filters})

    # 2. Diagnostic du type de capteur sur la première brute disponible
    first_filter_key = detected_filters[0]
    first_fits_file = files_by_filter[first_filter_key][0]
    camera_is_color = is_color_camera(first_fits_file)
    emit("progress", data={"step": "sensor_type", "type": "color" if camera_is_color else "mono"})

    # Dictionnaire de stockage des masters FITS générés pour l'alignement croisé final
    master_files_map = {}

    # 3. Traitement individualisé et empilement des canaux dans Siril
    for current_filter in detected_filters:
        emit("progress", data={"step": "stacking_started", "filter": current_filter})

        master_dark_path = get_master_dark_path(current_session_dir, files_by_filter[current_filter])
        master_flat_path = get_master_flat_path(current_session_dir, current_filter)
        master_bias_path = get_master_bias_path(current_session_dir)

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
        for src_file in files_by_filter[current_filter]:
            clean_name = src_file.name.replace(" ", "_")
            dst_file = filter_work_dir / clean_name
            if not dst_file.exists():
                try:
                    dst_file.symlink_to(src_file.resolve())
                except OSError:
                    shutil.copy(src_file, dst_file)
            num_files += 1

        # Génération et exécution du script de stacking
        stack_script = generate_siril_stack_script(
            filter_work_dir,
            current_filter,
            num_files,
            camera_is_color,
            master_dark_path,
            master_flat_path,
            master_bias_path
        )
        success = run_siril_command(current_session_dir, stack_script, f"stack_{current_filter}.ssf")

        if filter_work_dir.is_dir():
            shutil.rmtree(filter_work_dir)

        if not success:
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
            # Sécurité : Si Siril l'a nommé différemment (ex: sans préfixe ou déjà avec le nom custom)
            fallback_fit = current_session_dir / f"{file_prefix}_{current_filter}.fit"
            if fallback_fit.is_file():
                master_files_map[current_filter] = fallback_fit
            else:
                # Si le fichier est resté dans le sous-dossier de travail
                work_fit = current_session_dir / f"work_{current_filter}" / f"stacked_{current_filter}.fit"
                if work_fit.is_file():
                    shutil.move(work_fit, custom_fit_name)
                    master_files_map[current_filter] = custom_fit_name

        if filter_work_dir.is_dir():
            shutil.rmtree(filter_work_dir)

    # 4. Étape cruciale : Alignement croisé global
    if len(master_files_map) > 1:
        emit("progress", data={"step": "inter_filter_alignment_started", "message": "Recalage géométrique global..."})
        ref_candidate = master_files_map.get('HA') or list(master_files_map.values())[0]

        # On définit une liste des fichiers à aligner
        files_to_align = [f for f in master_files_map.values() if f != ref_candidate]

        if align_channels(current_session_dir, files_to_align, ref_image=ref_candidate):
            for filter_k, fit_path in master_files_map.items():
                aligned = fit_path.parent / f"{fit_path.stem}_aligned.fit"
                if aligned.exists():
                    master_files_map[filter_k] = aligned
        else:
            emit("warning", data={"step": "inter_filter_alignment_failed", "message": "Échec, utilisation des brutes"})

    # 5. Extraction finale des conteneurs FITS en images TIFF linéaires pour chaque canal validé
    for current_filter, final_fit_path in master_files_map.items():
        debug(f"Extraction TIFF du master linéaire finalisé : {current_filter}")
        # On passe temporairement par l'arborescence finale pour générer le .ssf de conversion
        conv_script = generate_siril_script(current_session_dir, current_filter, file_prefix)
        run_siril_command(current_session_dir, conv_script, f"conv_{current_filter}.ssf")

    # 6. Cartographie des fichiers TIFF générés
    tif_mapped_files = {}
    for current_filter in detected_filters:
        target_tiff = current_session_dir / f"{file_prefix}_{current_filter}.tif"
        if target_tiff.is_file():
            tif_mapped_files[current_filter] = target_tiff

    if not tif_mapped_files:
        debug("Échec critique de la collecte des matrices intermédiaires TIFF.")
        emit("done", data={"uuid": session_uuid, "output_format": format_requested})
        return False

    # 7. Composition finale via ImageMagick
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

    emit("done", data={"uuid": session_uuid, "output_format": format_requested, "file_prefix": file_prefix})
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking pipeline")
    parser.add_argument("uuid", help="Unique session UUID/directory")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "tiff", "webp"], help="Format d'encodage du fichier final généré")
    parser.add_argument("--dso", default="unknown", help="Nom de l'objet céleste ciblé (ex: ngc2359)")
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
