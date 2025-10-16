# Path: spec2class/src/spec2class/data_processor.py
# Helper functions for parsing MGF/MSP files

"""Data format parsers for Spec2Class"""

import numpy as np
import pandas as pd


def parse_mgf_file(mgf_path: str) -> pd.DataFrame:
    """
    Parse MGF file format to dataframe

    Args:
        mgf_path: Path to MGF file

    Returns:
        DataFrame with columns for Spec2Class (mz, Intensity, DB., ExactMass)
    """
    spectra = []
    current_spectrum = {}
    mz_list = []
    intensity_list = []

    with open(mgf_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "BEGIN IONS":
                current_spectrum = {}
                mz_list = []
                intensity_list = []

            elif line == "END IONS":
                if mz_list:
                    current_spectrum["mz"] = np.array(mz_list)
                    current_spectrum["Intensity"] = np.array(intensity_list)
                    if "DB." not in current_spectrum:
                        current_spectrum["DB."] = f"spectrum_{len(spectra)}"
                    spectra.append(current_spectrum)

            elif "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().upper()
                value = value.strip()
                
                if key == "TITLE":
                    current_spectrum["DB."] = value
                elif key == "PEPMASS":
                    try:
                        current_spectrum["ExactMass"] = float(value.split()[0])
                    except (ValueError, IndexError):
                        pass

            elif line and not line.startswith("#"):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz_list.append(float(parts[0]))
                        intensity_list.append(float(parts[1]))
                except ValueError:
                    continue

    df = pd.DataFrame(spectra)
    print(f"Parsed {len(df)} spectra from MGF file")
    return df


def parse_msp_file(msp_path: str) -> pd.DataFrame:
    """
    Parse MSP file format to dataframe

    Args:
        msp_path: Path to MSP file

    Returns:
        DataFrame with columns for Spec2Class (mz, Intensity, DB., ExactMass)
    """
    spectra = []
    current_spectrum = {}
    mz_list = []
    intensity_list = []
    reading_peaks = False

    with open(msp_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_spectrum and mz_list:
                    current_spectrum["mz"] = np.array(mz_list)
                    current_spectrum["Intensity"] = np.array(intensity_list)
                    if "DB." not in current_spectrum:
                        current_spectrum["DB."] = f"spectrum_{len(spectra)}"
                    spectra.append(current_spectrum)
                current_spectrum = {}
                mz_list = []
                intensity_list = []
                reading_peaks = False
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key.lower() in ["name", "title"]:
                    current_spectrum["DB."] = value
                elif key.lower() in ["precursormz", "precursor_mz"]:
                    try:
                        current_spectrum["ExactMass"] = float(value)
                    except ValueError:
                        pass
                elif "num" in key.lower() and "peak" in key.lower():
                    reading_peaks = True

            elif reading_peaks:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz_list.append(float(parts[0]))
                        intensity_list.append(float(parts[1]))
                except ValueError:
                    continue

    # Add last spectrum if exists
    if current_spectrum and mz_list:
        current_spectrum["mz"] = np.array(mz_list)
        current_spectrum["Intensity"] = np.array(intensity_list)
        if "DB." not in current_spectrum:
            current_spectrum["DB."] = f"spectrum_{len(spectra)}"
        spectra.append(current_spectrum)

    df = pd.DataFrame(spectra)
    print(f"Parsed {len(df)} spectra from MSP file")
    return df