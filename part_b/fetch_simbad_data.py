import pandas as pd
import numpy as np
from astroquery.simbad import Simbad
import os

# Paths
OUTLIERS_CSV = "data/dbscan_outliers.csv"
FINAL_TABLE_PATH = "data/anomaly_summary_table.csv"

def get_simbad_classifications():
    if not os.path.exists(OUTLIERS_CSV):
        print(f"Error: {OUTLIERS_CSV} not found.")
        return

    # 1. Load your detected outliers
    outliers_df = pd.read_csv(OUTLIERS_CSV)
    ids = outliers_df['APOGEE_ID'].astype(str).tolist()
    
    print(f"Preparing to query SIMBAD for {len(ids)} anomalies...")

    # 2. Convert APOGEE IDs (2M...) to SIMBAD format (2MASS J...)
    formatted_ids = [f"2MASS J{id[2:]}" if id.startswith("2M") else id for id in ids]

    # 3. Configure SIMBAD
    custom_simbad = Simbad()
    custom_simbad.reset_votable_fields()
    # We only add the fields that aren't provided by default
    custom_simbad.add_votable_fields('otype', 'sp_type', 'velocity')

    # 4. Execute the Query
    print("Connecting to SIMBAD servers (Strasbourg, France)...")
    try:
        result_table = custom_simbad.query_objects(formatted_ids)
        
        if result_table is None:
            print("No results found. Check your internet connection or ID format.")
            return

        # 5. Convert to Pandas
        simbad_df = result_table.to_pandas()
        
        # Clean up byte strings (very common in astroquery)
        for col in simbad_df.columns:
            if simbad_df[col].dtype == object:
                simbad_df[col] = simbad_df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        # 6. Create the final summary
        # SIMBAD usually returns uppercase column names. We'll use .get() to be safe.
        summary_table = pd.DataFrame({
            'APOGEE_ID': ids,
            'MATCHED_ID': simbad_df.get('TYPED_ID', simbad_df.get('typed_id', 'N/A')),
            'Object_Type': simbad_df.get('OTYPE', simbad_df.get('otype', 'Unknown')),
            'Spectral_Type': simbad_df.get('SP_TYPE', simbad_df.get('sp_type', 'N/A')),
            'Radial_Vel': simbad_df.get('VELOCITY', simbad_df.get('velocity', 'N/A'))
        })

        # 7. Save to CSV
        summary_table.to_csv(FINAL_TABLE_PATH, index=False)
        print(f"\nSuccess! Summary table saved to {FINAL_TABLE_PATH}")
        print("\nTop 10 identified anomalies:")
        print(summary_table.head(10))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_simbad_classifications()