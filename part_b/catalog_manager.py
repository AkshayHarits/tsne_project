import os
import pandas as pd
from astropy.io import fits
import numpy as np

CATALOG_PATH = "data/allStar-dr17-synspec_rev1.fits"
OUTPUT_CSV = "data/golden_sample.csv"

def filter_heavy_catalog(sample_size=1000):
    if not os.path.exists('data'): os.makedirs('data')
    
    print(f"Opening {CATALOG_PATH}...")
    with fits.open(CATALOG_PATH, memmap=True) as hdul:
        data = hdul[1].data
        
        print("Extracting and separating strings from numbers...")
        
        # 1. NUMBERS ONLY: We flip the bytes for Windows compatibility
        def fix_num(arr):
            return arr.byteswap().view(arr.dtype.newbyteorder('='))

        # 2. STRINGS ONLY: We do NOT flip the bytes, just cast to string
        def fix_str(arr):
            return arr.astype(str)

        subset = {
            'APOGEE_ID': fix_str(data['APOGEE_ID']),
            'TELESCOPE': fix_str(data['TELESCOPE']),
            'FIELD': fix_str(data['FIELD']),
            'SNR': fix_num(data['SNR']),
            'STARFLAG': fix_num(data['STARFLAG']),
            'ASPCAPFLAG': fix_num(data['ASPCAPFLAG']),
            'VHELIO_AVG': fix_num(data['VHELIO_AVG']),
            'VSCATTER': fix_num(data['VSCATTER']), # ADDED THIS
            'TEFF': fix_num(data['TEFF']),
            'LOGG': fix_num(data['LOGG']),
            'FE_H': fix_num(data['FE_H'])
        }
        
        df = pd.DataFrame(subset)

    print("Applying quality filters...")
    mask = (
        (df['SNR'] > 100) & 
        (df['STARFLAG'] == 0) & 
        (df['ASPCAPFLAG'] == 0) &
        (df['TEFF'] < 8000) & 
        (df['VSCATTER'] < 1.0)
    )
    clean_df = df[mask].copy()

    # Clean up strings (removes the b'...' formatting if present and strips spaces)
    for col in ['APOGEE_ID', 'TELESCOPE', 'FIELD']:
        clean_df[col] = clean_df[col].str.replace(r"^b'|'$", "", regex=True).str.strip()

    print(f"Found {len(clean_df)} perfect stars. Sampling {sample_size}...")
    
    final_sample = clean_df.sample(n=min(sample_size, len(clean_df)), random_state=42)
    final_sample.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Your IDs are readable again.")

if __name__ == "__main__":
    filter_heavy_catalog(sample_size=30000)#change here if you want to increase number of samples