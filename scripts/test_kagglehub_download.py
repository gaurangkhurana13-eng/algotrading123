import kagglehub
import os

print("--- Attempting kagglehub dataset_download for yeashusemwal/nse-stocks-15min-data-from-2019-03-01 ---")
try:
    path = kagglehub.dataset_download("yeashusemwal/nse-stocks-15min-data-from-2019-03-01")
    print("Dataset downloaded successfully to path:")
    print(path)
    files = os.listdir(path)
    print(f"Total files in downloaded dataset: {len(files)}")
    print("Sample files:", files[:20])
except Exception as e:
    print("Error during kagglehub download:", e)
