#!/bin/bash
# check_tiles.sh

BASE="E:\UTS\CNN and Deep Learning\Assignment 3\42028-DLCNN\data\dfc25_track2_trainval\dfc25_track2_trainval"
SPLITS=("train" "val")
FOLDERS=("pre-event" "post-event" "target")

for split in "${SPLITS[@]}"; do
    echo ""
    echo "==================== ${split^^} ===================="

    counts=()
    all_found=true

    for folder in "${FOLDERS[@]}"; do
        path="$BASE/$split/$folder"
        if [ -d "$path" ]; then
            count=$(ls "$path"/*.tif 2>/dev/null | wc -l)
            counts+=($count)
            printf "  %-12s : %s tiles\n" "$folder" "$count"
        else
            echo "  $folder : ❌ NOT FOUND"
            all_found=false
        fi
    done

    # Check all three counts match
    if [ "$all_found" = true ]; then
        if [ "${counts[0]}" -eq "${counts[1]}" ] && [ "${counts[1]}" -eq "${counts[2]}" ]; then
            echo "  ✅ All counts match (${counts[0]} tiles)"
        else
            echo "  ❌ COUNT MISMATCH — check your extraction"
            echo "     pre-event=${counts[0]} post-event=${counts[1]} target=${counts[2]}"
        fi
    fi
done

echo ""