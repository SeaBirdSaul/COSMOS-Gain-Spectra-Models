import numpy as np
import pandas as pd
import os
import json
import re

def spectra_dict_to_array(spectra_dict, num_channels=95, default_val=-60.0):
    """
    Converts a spectrum dictionary {'1': -20, '3': -20} to a dense numpy array of shape (95,).
    Missing channels are filled with default_val (noise floor).
    """
    arr = np.full(num_channels, default_val, dtype=np.float32)
    
    for ch_str, power in spectra_dict.items():
        try:
            ch_idx = int(ch_str) - 1 # 0-based index
            if 0 <= ch_idx < num_channels:
                arr[ch_idx] = float(power)
        except ValueError:
            continue
    return arr

def active_idx_to_mask(active_idx_list, num_channels=95):
    """
    Converts list of active channel indices [1, 5, 10] to boolean mask.
    """
    mask = np.zeros(num_channels, dtype=np.int8)
    for ch in active_idx_list:
        try:
            ch_int = int(ch)
            if 1 <= ch_int <= num_channels:
                mask[ch_int-1] = 1
        except ValueError:
            pass
    return mask

def get_path_metadata(json_path):
    """
    Extracts metadata from folder structure:
    .../dataset/booster/18dB/fix/rdm1-co1/file.json
    Returns: edfa_type, gain_folder, channel_type, roadm
    """
    try:
        parts = os.path.normpath(json_path).split(os.sep)
        source = os.path.basename(json_path)

        # Actual layout: .../dataset/<edfa_type>/<gain_folder>/<channel_type>/<filename>.json
        # ROADM name is in the filename, not a folder level.
        roadm_match = re.search(r'edfa_meas_([^.]+)', source)
        roadm = roadm_match.group(1) if roadm_match else 'unknown'
        
        # Adjust indices based on your actual depth. These assume the structure above.
        # If running flat or different depth, this might need adjustment.
        # Safe fallback defaults if structure is shallow:
        if len(parts) >= 5:
            edfa_type = parts[-4]
            gain_folder = parts[-3]
            channel_type = parts[-2]
            return edfa_type, gain_folder, channel_type, roadm
    except Exception:
        pass
    return 'unknown', 'unknown', 'unknown', 'unknown'



# --- Optics Helpers ---
C_NM_GHZ = 299792.458

def freq_ghz_to_wavelength_nm(freq_ghz):
    """Converts Frequency (GHz) to Wavelength (nm)."""
    return C_NM_GHZ / np.asarray(freq_ghz, dtype=np.float64)


def extract_samples_from_json(json_path, num_channels=95, calculate_ripple=False):
    try:
        with open(json_path, "r") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

    rows = []
    # 1. Metadata
    edfa_type, gain_folder, channel_type, roadm = get_path_metadata(json_path)
    
    # 2. Global Setup (Frequency Grid)
    setup = obj.get("measurement_setup", {})
    freq_grid_ghz = setup.get("roadm_wss_channel_freq_center_list", None)
    wl_nm = None
    if freq_grid_ghz:
        wl_nm = freq_ghz_to_wavelength_nm(freq_grid_ghz)

    measurements = obj.get("measurement_data", [])
    
    for m in measurements:
        sin_dict  = m.get("roadm_dut_wss_output_power_spectra")
        active    = m.get("roadm_dut_wss_active_channel_index")
        
        # Check both Booster and Preamp keys
        sout_dict = m.get("roadm_dut_booster_output")
        if sout_dict is None:
            sout_dict = m.get("roadm_dut_preamp_output")

        if sin_dict is None or sout_dict is None or active is None:
            continue

        # Arrays
        sin  = spectra_dict_to_array(sin_dict, num_channels=num_channels, default_val=-60.0)
        sout = spectra_dict_to_array(sout_dict, num_channels=num_channels, default_val=-60.0)

        # Mask
        mask = active_idx_to_mask(active, num_channels=num_channels)
        num_active = int(mask.sum())

        # Gain
        gain = np.full(num_channels, np.nan, dtype=np.float32)
        valid_indices = (mask == 1)
        active_gain = sout[valid_indices] - sin[valid_indices]
        gain[valid_indices] = active_gain

        ripple = None
        if calculate_ripple:
            ripple = np.full(num_channels, np.nan, dtype=np.float32)
            ripple[valid_indices] = active_gain - np.nanmean(active_gain)

        # Meta
        edfa_info = m.get("roadm_dut_edfa_info", {}) or {}
        
        row = {
            "edfa_type": edfa_type,
            "gain_folder": gain_folder,
            "channel_type": channel_type,
            "open_channel_type": m.get("open_channel_type", "unknown"),
            "roadm": roadm,
            "source_file": os.path.basename(json_path),
            "sequence_id": m.get("sequence_id", -1),
            "pin_total": edfa_info.get("input_power", np.nan),
            "pout_total": edfa_info.get("output_power", np.nan),
            "target_gain": edfa_info.get("target_gain", np.nan),
            "input_spectra": sin,
            "output_spectra": sout,
            "ripple_spectra": ripple,
            "gain_spectra": gain,
            "mask": mask,
            "num_active_channels": num_active,
            "wavelength_nm": wl_nm
        }
        rows.append(row)

    return rows

def extract_samples_from_multispan_json(json_path, num_channels=95, calculate_ripple=False):
    try:
        with open(json_path, "r") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []
    
    if not isinstance(obj, list):
        return []
    
    rows = []
    source = os.path.basename(json_path)

    for elem in obj:
        active_list = elem.get("open_channel_list", [])
        if not active_list:
            continue
        mask = active_idx_to_mask(active_list, num_channels=num_channels)
        num_active = int(mask.sum())
        in_keys = [k for k in elem if k.endswith("In") and isinstance(elem[k], dict)]

        for in_key in in_keys:
            prefix = in_key[:-2]
            out_key = prefix + "Out"
            edfa_key = prefix + "EDFA"
            sin_dict = elem.get(in_key)
            sout_dict = elem.get(out_key)
            if sin_dict is None or sout_dict is None:
                continue
            sin = spectra_dict_to_array(sin_dict, num_channels=num_channels, default_val=-60.0)
            sout = spectra_dict_to_array(sout_dict, num_channels=num_channels, default_val=-60.0)

            gain = np.full(num_channels, np.nan, dtype=np.float32)
            valid = (mask == 1)
            active_gain = sout[valid] - sin[valid]
            gain[valid] = active_gain

            ripple = None
            if calculate_ripple:
                ripple = np.full(num_channels, np.nan, dtype=np.float32)
                ripple[valid] = active_gain - np.nanmean(active_gain)

            edfa_info = elem.get(edfa_key, {}) or {}

            rows.append({
                "source_file": source,
                "pin_total": edfa_info.get("input_power", np.nan),
                "input_spectra": sin,
                "output_spectra": sout,
                "gain_spectra": gain,
                "ripple_spectra": ripple,
                "mask": mask,
                "num_active_channels": num_active,
                "roadm": prefix,
                "target_gain": edfa_info.get("target_gain", np.nan),
                "target_power": edfa_info.get("target_power", np.nan),
                "output_power": edfa_info.get("output_power", np.nan),
                "voa_input_power": edfa_info.get("voa_input_power", np.nan),
                "voa_output_power": edfa_info.get("voa_output_power", np.nan),
                "voa_attenuation": edfa_info.get("voa_attenuation", np.nan),
                "open_channel_type": "multispan",
                "gain_folder": "multispan"
            })
    return rows