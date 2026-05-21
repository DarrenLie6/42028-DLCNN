"""Debug script to check xview2 label loading and rasterization"""
import sys
import json
from pathlib import Path

import numpy as np
import rasterio
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("data/xView2/geotiffs/tier1")
stem = "guatemala-volcano_00000000"

# Load JSON
lbl_path = ROOT / "labels" / f"{stem}_post_disaster.json"
img_path = ROOT / "images" / f"{stem}_post_disaster.tif"

print(f"Label path exists: {lbl_path.exists()}")
print(f"Image path exists: {img_path.exists()}")

if lbl_path.exists() and img_path.exists():
    with open(lbl_path) as f:
        data = json.load(f)
    
    print(f"\nJSON keys: {data.keys()}")
    print(f"Features keys: {data.get('features', {}).keys()}")
    
    features = data.get("features", {})
    
    # Check what's in xy
    if "xy" in features:
        print(f"\nxy features: {len(features['xy'])} buildings")
        if features['xy']:
            first = features['xy'][0]
            print(f"First building keys: {first.keys()}")
            print(f"First building subtype: {first.get('subtype')}")
            print(f"First building wkt (first 200 chars): {first.get('wkt', '')[:200]}")
    
    # Check image dimensions
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        print(f"\nImage shape: {h} x {w}")
        print(f"Transform: {transform}")
    
    # Try to parse and rasterize one polygon
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    count = 0
    
    for feature_type in ["xy", "lng_lat"]:
        if feature_type not in features:
            continue
        
        for i, building in enumerate(features[feature_type][:5]):  # First 5
            if "wkt" not in building:
                continue
            
            try:
                wkt_str = building.get("wkt", "")
                if not wkt_str.startswith("POLYGON"):
                    continue
                
                # Parse coordinates
                coords_str = wkt_str.replace("POLYGON", "").replace("(", "").replace(")", "").strip()
                lon_lat_coords = []
                for coord_pair in coords_str.split(","):
                    parts = coord_pair.strip().split()
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        lon_lat_coords.append((lon, lat))
                
                if len(lon_lat_coords) < 3:
                    continue
                
                print(f"\nBuilding {feature_type}[{i}]:")
                print(f"  Properties: {building.get('properties')}")
                print(f"  Num coords: {len(lon_lat_coords)}")
                print(f"  First 3 coords: {lon_lat_coords[:3]}")
                
                # Convert to pixel
                pixel_coords = []
                for lon, lat in lon_lat_coords:
                    col = (lon - transform.c) / transform.a
                    row = (lat - transform.f) / transform.e
                    col = max(0, min(w - 1, col))
                    row = max(0, min(h - 1, row))
                    pixel_coords.append([int(col), int(row)])
                
                print(f"  First 3 pixel coords: {pixel_coords[:3]}")
                
                pixel_coords = np.array(pixel_coords, dtype=np.int32)
                subtype = building.get("subtype", "no-damage")
                damage_class = {"no-damage": 1, "minor-damage": 2, "major-damage": 2, "destroyed": 3}.get(subtype, 1)
                
                cv2.fillPoly(damage_mask, [pixel_coords], damage_class)
                count += 1
                
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\nRasterized {count} polygons")
    print(f"Mask unique values: {np.unique(damage_mask)}")
    print(f"Mask value counts: {[(v, (damage_mask==v).sum()) for v in np.unique(damage_mask)]}")
    
    # Save a test image
    test_img = damage_mask.astype(np.uint8) * 85  # Scale to 0-255 for visibility
    cv2.imwrite("test_damage_mask.png", test_img)
    print("\nSaved test_damage_mask.png")
