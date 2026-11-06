import kagglehub
import os
import pandas as pd

try:
    print("Downloading nitirajkulkarni/electcast-ns-stock-performance...")
    path = kagglehub.dataset_download("nitirajkulkarni/electcast-ns-stock-performance")
    files = os.listdir(path)
    print("Files:", files)
    if files:
        df = pd.read_csv(os.path.join(path, files[0]))
        print("Columns:", df.columns.tolist())
        print(df.head(3))
except Exception as e:
    print("Error:", e)

try:
    print("\nDownloading theakashkharb/rpower-historical-market-data...")
    path2 = kagglehub.dataset_download("theakashkharb/rpower-historical-market-data")
    files2 = os.listdir(path2)
    print("Files:", files2)
    if files2:
        df2 = pd.read_csv(os.path.join(path2, files2[0]))
        print("Columns:", df2.columns.tolist())
        print(df2.head(3))
except Exception as e:
    print("Error 2:", e)
