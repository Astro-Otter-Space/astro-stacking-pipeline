# Astro-Otter: Automated Astrophotography Stacking Script

A Python script designed for automated deep-sky image stacking and processing. It orchestrates Siril for calibration and stacking, then composes and finishes the final image in pure Python (numpy, scipy, tifffile, Pillow), creating a streamlined workflow for both EAA (Electronically Assisted Astronomy) and SaaS (Software as a Service) high-availability scenarios.

## Information
This script is currently in development and is not yet ready for production use. If you see any issues or have suggestions, please report them on the [GitHub repository](https://github.com/Astro-Otter-Space/astro-stacking-pipeline/issues).

## Features

*   **Automatic Sensor Detection**: Identifies if your camera is Mono or OSC (One-Shot-Color) by reading FITS header keywords (e.g., `BAYERPAT`, `INSTRUME`).
*   **Smart Filter Sorting**: Automatically categorizes light frames into filters (HA, OIII, SII, RED, GREEN, BLUE, LUMINANCE, etc.) using FITS header data (`FILTER`, `IMAGETYP`) or filename patterns.
*   **Master Calibration**: Automatically locates and applies master darks, flats, and bias frames for calibration, and warns when a master flat is much older than the lights (`flat_staleness_warning`).
*   **Robust File Handling**: Normalizes master calibration files (ensures 2D FITS format) and cleans spaces from filenames to prevent Siril parsing errors.
*   **Adaptive Processing**: Adjusts Siril's `subsky` parameters based on the availability of calibration frames (e.g., uses higher tolerance if no flat is available).
*   **Flexible Output**: Supports multiple output formats (PNG, JPG, TIFF, WEBP). Lossy formats are encoded only once, at the very end.
*   **Cross-Filter Alignment**: Performs a final geometric alignment between stacked filter channels for precise registration.
*   **Advanced Compositing**: Creates final color images using palettes detected automatically from the available channels:
    *   **SHO (Hubble)**: SII (Red), Ha (Green), OIII (Blue).
    *   **SHO+RGB**: SHO blended (80/20) with broadband RGB for natural star colors.
    *   **HOO**: Ha (Red), 30 % Ha + 70 % OIII (Green), OIII (Blue).
    *   **SHA**: SII (Red), Ha (Green), broadband Blue (or a synthetic blue derived from Ha when no RGB is available).
    *   **HaRGB**: Ha-enriched red (60 % Ha + 40 % Red) with broadband Green and Blue.
    *   **LRGB**: hue and saturation from RGB, lightness from the luminance channel.
    *   **RGB**: Standard color imaging (mono camera with filters, or OSC).
    *   **CLEAR / L**: single broadband channel (typically an OSC camera), with the full finishing chain.
*   **Pure-Python Finishing Chain**: per-palette levels, sigmoidal contrast, saturation and unsharp mask, followed by automatic corrections applied when relevant:
    *   background flattening (removes a smooth residual gradient or edge glow measured on clean sky),
    *   sky neutralization (per-channel offset so the background is neutral grey),
    *   highlight roll-off on large bright cores (galaxy nuclei, bright nebula centers) without touching saturated stars,
    *   star-only green reduction on SHO palettes,
    *   noise-cored detail boost on luminance, and star halo reduction.
    The tuning constants live at the top of `stacking.py`; there are no extra command-line parameters.
*   **Headless Operation**: Designed to run entirely from the command line, perfect for automated pipelines. Progress is reported as one JSON event per line.

## Prerequisites

Before using this script, ensure the following software is installed on your system:

1.  **Python 3.9 or higher**: The script is written in Python.
2.  **Siril 1.2.0 or higher**: The core calibration and stacking engine. The `siril-cli` command must be available in your system's PATH.
    *   Installation (Ubuntu/Debian): `sudo apt install siril`
    *   Installation (macOS with Homebrew): `brew install siril`
    *   Windows: Download from the [official Siril website](https://www.siril.org/).
3.  **Python Packages**: `numpy`, `scipy`, `tifffile`, `Pillow` and `astropy`. Install them with pip:
    ```bash
    pip install -r requirements.txt
    ```

No other system dependency is required: image composition and finishing no longer rely on ImageMagick.

## Installation

1.  Clone this repository or download the script:
    ```bash
    git clone https://github.com/Astro-Otter-Space/astro-stacking-pipeline.git
    cd astro-stacking-pipeline
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure `siril-cli` is accessible from your command line.


## Usage

The script is executed from the command line.
```bash
python3 stacking.py <uuid> [OPTIONS]
```
### Arguments

*   `<uuid>` (Required): A unique identifier for the processing session. This is used to organize input and output files.

### Options

*   `--format <format>`: The output image format. Default is `png`. Choices are `png`, `jpg`, `tiff`, `webp`.
*   `--dso <name>`: The name of the Deep-Sky Object (DSO) being processed (e.g., `M42`, `ngc2359`). This is used in the output filename. Default is `unknown`.
*   `--verbose`, `-v`: Enable debug logging to the console.

### Directory Structure

The script expects a specific directory layout relative to its location:
```
venv/                    # Python libraries
sessions/
    └── <uuid>/
        ├── lights/          # Raw light frames (.fit, .fits, .fts, or camera RAW)
        ├── darks/           # Master darks and dark frames (optional)
        ├── flats/           # Master flats and flat frames (optional)
        └── bias/            # Master bias/offset frames (optional)
```
### Example Commands
```bash
# Basic usage, outputs PNG
python3 stacking.py my_session_001 --dso=M42

# Output as high-quality WebP
python3 stacking.py ngc2359_ha_run --format=webp --dso=ngc2359

# Verbose mode for debugging
python3 stacking.py debug_session --dso=m42 --verbose
```
## Output

The deliverable is a single file in the `sessions/<uuid>/` directory:
*   `<dso>_<timestamp>_full.<format>`: The final composed image.

Intermediate files (per-filter stacked and aligned FITS, linear TIFFs, temporary Siril scripts and working directories) are deleted once the composition is done.

While it runs, the script prints progress events as JSON lines (`start`, `stacking_started`, `combine`, `finishing_started`, `finishing_done`, `compose_color_stats`, `composition_done`, ...). `compose_color_stats` reports the sky and mean colors and the corrections applied, which helps diagnosing a color cast straight from the log.

## How It Works

1.  **Initialization**: The script reads command-line arguments and sets up logging.
2.  **File Discovery & Sorting**: It scans the `lights/` directory, reads FITS headers to determine the filter and sensor type, and sorts frames accordingly.
3.  **Master Frame Lookup**: For each filter, it searches the `darks/`, `flats/`, and `bias/` directories for appropriate master calibration frames.
4.  **Siril Processing**: For each filter, it generates a Siril script (.ssf) to:
    *   Convert raw files.
    *   Apply calibration (if masters are found).
    *   Register (align) and stack the frames.
    *   Perform background extraction (`subsky`).
    *   Stretch the result and save it as a FITS file.
5.  **Cross-Filter Alignment**: If multiple filters are stacked, the script aligns the master frames against a reference (e.g., Ha) to ensure perfect registration.
6.  **TIFF Conversion**: The final stacked FITS files are converted to linear TIFFs using Siril.
7.  **Final Composition**: The TIFFs are merged into the final color image with the appropriate palette and finished (contrast, saturation, sharpening, background and sky corrections), all in Python with numpy, scipy and tifffile.
8.  **Orientation & Encoding**: The image is flipped to the conventional top-left origin and written once in the requested format.
9.  **Cleanup**: Temporary files, intermediate FITS and TIFFs, and working directories are removed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
