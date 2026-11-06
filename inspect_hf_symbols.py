from huggingface_hub import HfFileSystem
import re

try:
    fs = HfFileSystem()
    files = fs.ls("datasets/xxparthparekhxx/indian-stock-market-minute-data", detail=False)
    print("Top-level files/dirs in dataset:")
    for f in files[:15]:
        print(" ", f)
        
    # Check minute folder files
    min_files = fs.ls("datasets/xxparthparekhxx/indian-stock-market-minute-data/minute", detail=False)
    print(f"\nTotal files in minute/: {len(min_files)}")
    print("First 10 minute files:")
    for f in min_files[:10]:
        print(" ", f)
        
    # Check if symbols like PCJEWELLER or RPOWER exist in file paths
    target_matches = [f for f in min_files if any(s in f for s in ["PCJEWELLER", "RPOWER", "ZEEL", "IDEA", "MTNL"])]
    print(f"\nMatching files for target stocks: {target_matches[:10]}")
except Exception as e:
    print("Error inspecting HfFileSystem:", e)
