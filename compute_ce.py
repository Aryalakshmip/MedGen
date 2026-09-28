from pprint import pprint
import pandas as pd
from modules.metrics import compute_mlc

def main():
    # Update paths for IU X-ray dataset
    res_path = "results/iu_xray/res_labeled.csv"
    gts_path = "results/iu_xray/gts_labeled.csv"
    
    # Read and preprocess data
    res_data = pd.read_csv(res_path)
    gts_data = pd.read_csv(gts_path)
    
    # Fill NA values with 0
    res_data = res_data.fillna(0)
    gts_data = gts_data.fillna(0)

    # Get label set and convert to numpy arrays
    label_set = res_data.columns[1:].tolist()
    res_data = res_data.iloc[:, 1:].to_numpy()
    gts_data = gts_data.iloc[:, 1:].to_numpy()
    
    # Convert -1 values to 0
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    # Compute metrics
    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)

if __name__ == '__main__':
    main()
