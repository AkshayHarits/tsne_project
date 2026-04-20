import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from tqdm import tqdm

RAW_DIR = "data/raw_spectra/"
CSV_PATH = "data/golden_sample.csv"
OUTPUT_MATRIX = "data/clean_fluxes.npy"
OUTPUT_IDS = "data/clean_ids.npy"

C_KM_S = 299792.458  

def preprocess_spectra_standard():
    df = pd.read_csv(CSV_PATH)
    velocities = dict(zip(df['APOGEE_ID'].astype(str), df['VHELIO_AVG']))
    
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.fits')]
    
    # 1. Establish Master Grid
    with fits.open(os.path.join(RAW_DIR, files[0])) as sample_hdul:
        log_wave = sample_hdul[1].header['CRVAL1'] + sample_hdul[1].header['CDELT1'] * np.arange(sample_hdul[1].header['NAXIS1'])
        master_wavelengths = 10 ** log_wave

    clean_matrix = []
    valid_ids = []

    print("Running APOGEE-Standard Preprocessing (Error Arrays & Gaps)...")
    for f in tqdm(files):
        apogee_id = f.split("dr17-")[1].replace(".fits", "")
        if apogee_id not in velocities or np.isnan(velocities[apogee_id]):
            continue
            
        v_helio = velocities[apogee_id]

        try:
            with fits.open(os.path.join(RAW_DIR, f)) as hdul:
                flux = hdul[1].data
                err = hdul[2].data
                
                # --- THE TRUE ASPCAP FILTER ---
                # The ASPCAP pipeline marks bad pixels and camera gaps by 
                # driving the error extremely high or setting it to NaN.
                # A 20% error (0.2) on normalized flux is our safe cutoff.
                bad_pixels = (err > 0.2) | np.isnan(err) | np.isnan(flux) | (flux <= 0)
                
                # The APOGEE camera gaps make up ~5% of the 8575 pixels. 
                # If more than 20% of the star is broken, we reject it entirely.
                if np.sum(bad_pixels) > (len(flux) * 0.2):
                    continue
                
                flux_clean = np.copy(flux)
                flux_clean[bad_pixels] = np.nan
                
                # De-redshifting
                doppler_factor = 1.0 + (v_helio / C_KM_S)
                obs_wavelengths = master_wavelengths * doppler_factor
                
                valid_mask = ~np.isnan(flux_clean)
                
                # Interpolate over the bad pixels and camera gaps
                f_interp = interp1d(
                    obs_wavelengths[valid_mask], 
                    flux_clean[valid_mask], 
                    bounds_error=False, 
                    fill_value=1.0 
                )
                
                rest_flux = f_interp(master_wavelengths)
                rest_flux = np.nan_to_num(rest_flux, nan=1.0)
                
                # --- NEW: Per-Spectrum Normalization ---
                median_flux = np.median(rest_flux)
                if median_flux > 0:
                    rest_flux = rest_flux / median_flux
                
                clean_matrix.append(rest_flux)
                valid_ids.append(apogee_id)
                
        except Exception as e:
            pass

    clean_matrix = np.array(clean_matrix)
    valid_ids = np.array(valid_ids)
    
    np.save(OUTPUT_MATRIX, clean_matrix)
    np.save(OUTPUT_IDS, valid_ids)
    print(f"\nSaved strict-processed matrix: {clean_matrix.shape}")

if __name__ == "__main__":
    preprocess_spectra_standard()