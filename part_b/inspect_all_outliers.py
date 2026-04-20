import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

OUTLIERS_CSV = "data/dbscan_outliers.csv"
ALL_STARS_CSV = "data/golden_sample.csv"
RAW_DIR = "data/raw_spectra/"
OUTPUT_DIR = "data/anomaly_plots/"

def get_wavelengths(header):
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    naxis1 = header['NAXIS1']
    return 10 ** (crval1 + cdelt1 * np.arange(naxis1))

def plot_all_anomalies():
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(OUTLIERS_CSV):
        print("Outliers CSV not found!")
        return

    outliers_df = pd.read_csv(OUTLIERS_CSV)
    all_stars_df = pd.read_csv(ALL_STARS_CSV)
    
    # Grab one normal star to use as the baseline for all plots
    normal_df = all_stars_df[~all_stars_df['APOGEE_ID'].isin(outliers_df['APOGEE_ID'])]
    normal_star_id = str(normal_df.iloc[0]['APOGEE_ID'])
    normal_file = os.path.join(RAW_DIR, f"aspcapStar-dr17-{normal_star_id}.fits")
    
    with fits.open(normal_file) as hdul_normal:
        normal_flux = hdul_normal[1].data
        wavelengths = get_wavelengths(hdul_normal[1].header)

    print(f"Generating comparison plots for {len(outliers_df)} anomalies...")
    
    for i, row in outliers_df.iterrows():
        weird_star_id = str(row['APOGEE_ID'])
        weird_file = os.path.join(RAW_DIR, f"aspcapStar-dr17-{weird_star_id}.fits")
        
        try:
            with fits.open(weird_file) as hdul_weird:
                weird_flux = hdul_weird[1].data
        except FileNotFoundError:
            print(f"Missing file for {weird_star_id}, skipping...")
            continue
            
        plt.figure(figsize=(14, 6))
        
        # Plot Normal Star
        plt.plot(wavelengths, normal_flux, color='cornflowerblue', alpha=0.6, linewidth=0.8, label="Normal Star")
        
        # Plot Anomaly
        plt.plot(wavelengths, weird_flux, color='red', alpha=0.9, linewidth=1.0, label=f"Anomaly: {weird_star_id}")

        plt.title(f"DBSCAN Anomaly #{i+1} | APOGEE ID: {weird_star_id}")
        plt.xlabel("Wavelength (Angstroms)")
        plt.ylabel("Normalized Flux")
        plt.xlim(16000, 16400) 
        plt.ylim(0.2, 1.3)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot and close it to free up memory
        save_path = os.path.join(OUTPUT_DIR, f"anomaly_{i+1:02d}_{weird_star_id}.png")
        plt.savefig(save_path)
        plt.close() 
        
    print(f"Done! All {len(outliers_df)} plots have been saved to the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    plot_all_anomalies()