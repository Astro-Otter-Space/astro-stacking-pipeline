# Astro-Otter: Automated Astrophotography Stacking Script

A Python script designed for automated deep-sky image stacking and processing. It orchestrates Siril for calibration and stacking, then uses ImageMagick for final composition, creating a streamlined workflow for both EAA (Electronically Assisted Astronomy) and SaaS (Service as a Service) high-availability scenarios.

## Features

*   **Automatic Sensor Detection**: Identifies if your camera is Mono or OSC (One-Shot-Color) by reading FITS header keywords (e.g., `BAYERPAT`, `INSTRUME`).
*   **Smart Filter Sorting**: Automatically categorizes light frames into filters (HA, OIII, SII, RED, GREEN, BLUE, etc.) using FITS header data (`FILTER`, `IMAGETYP`) or filename patterns.
*   **Master Calibration**: Automatically locates and applies master darks, flats, and bias frames for calibration.
*   **Robust File Handling**: Normalizes master calibration files (ensures 2D FITS format) and cleans spaces from filenames to prevent Siril parsing errors.
*   **Adaptive Processing**: Adjusts Siril's `subsky` parameters based on the availability of calibration frames (e.g., uses higher tolerance if no flat is available).
*   **Flexible Output**: Supports multiple output formats (PNG, JPG, TIFF, WEBP).
*   **Cross-Filter Alignment**: Performs a final geometric alignment between stacked filter channels for precise registration.
*   **Advanced Compositing**: Creates final color images using various palettes:
    *   **Hubble Palette (SHO)**: SII (Red), Ha (Green), OIII (Blue).
    *   **HOO Palette**: Ha (Red), OIII (Green), OIII (Blue).
    *   **RGB**: Standard color imaging.
    *   **HOO+RGB Hybrid**: Blends narrowband data (Ha, OIII) with RGB for vibrant stars and nebulosity.
*   **Headless Operation**: Designed to run entirely from the command line, perfect for automated pipelines.

## Prerequisites

Before using this script, ensure the following software is installed on your system:

1.  **Python 3.8 or higher**: The script is written in Python.
2.  **Siril 1.2.0 or higher**: The core image processing engine. The `siril-cli` command must be available in your system's PATH.
    *   Installation (Ubuntu/Debian): `sudo apt install siril`
    *   Installation (macOS with Homebrew): `brew install siril`
    *   Windows: Download from the [official Siril website](https://www.siril.org/).
3.  **ImageMagick**: Used for the final image composition and format conversion.
    *   Installation (Ubuntu/Debian): `sudo apt install imagemagick`
    *   Installation (macOS with Homebrew): `brew install imagemagick`
    *   Windows: Download from the [ImageMagick website](https://imagemagick.org/).
4.  **Python Packages**: Install the required Python dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

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
4.  Ensure `siril-cli` and `convert` (from ImageMagick) are accessible from your command line.


## Usage

The script is executed from the command line.
```
bash
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
        ├── lights/          # Raw light frames (.fit, .fits, .fts)
        ├── darks/           # Master darks and dark frames (optional)
        ├── flats/           # Master flats and flat frames (optional)
        └── bias/            # Master bias/offset frames (optional)
```
### Example Commands
```
bash
# Basic usage, outputs PNG
python3 stacking.py my_session_001 --dso=M42

# Output as high-quality WebP
python3 stacking.py ngc2359_ha_run --format=webp --dso=ngc2359

# Verbose mode for debugging
python3 stacking.py debug_session --dso=m42 --verbose
```
## Output

The script will create the following files in the `sessions/<uuid>/` directory:
*   `stacked_<FILTER>.fit`: Intermediate stacked FITS files for each filter.
*   `stacked_<FILTER>.tif`: Linear TIFF files converted from the stacked FITS.
*   `<dso>_<timestamp>_full.<format>`: The final composed output image.

## How It Works

1.  **Initialization**: The script reads command-line arguments and sets up logging.
2.  **File Discovery & Sorting**: It scans the `lights/` directory, reads FITS headers to determine the filter and sensor type, and sorts frames accordingly.
3.  **Master Frame Lookup**: For each filter, it searches the `darks/`, `flats/`, and `bias/` directories for appropriate master calibration frames.
4.  **Siril Processing**: For each filter, it generates a Siril script (.ssf) to:
    *   Convert raw files.
    *   Apply calibration (if masters are found).
    *   Register (align) and stack the frames.
    *   Perform background extraction (`subsky`).
    *   Save the result as a FITS file.
5.  **Cross-Filter Alignment**: If multiple filters are stacked, the script aligns the master frames against a reference (e.g., Ha) to ensure perfect registration.
6.  **TIFF Conversion**: The final stacked FITS files are converted to linear TIFFs using Siril.
7.  **Final Composition**: ImageMagick combines the TIFFs into the final color image using the appropriate palette.
8.  **Cleanup**: Temporary files and working directories are removed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.