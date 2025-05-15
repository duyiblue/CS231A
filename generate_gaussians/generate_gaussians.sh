#!/bin/bash

# Base directory where all objects are stored
BASE_DIR="./nerfdata"
OUTPUT_BASE_DIR="./outputs"
EXPORT_DIR="/iris/u/duyi/cs231a/get_a_grip/data/dataset/nano/nerfdata"

# Loop through each subdirectory (object) in the base directory
for OBJECT_DIR in "$BASE_DIR"/*; do
    if [ -d "$OBJECT_DIR" ]; then
        OBJECT_NAME=$(basename "$OBJECT_DIR")

        # Skip folders with names shorter than 20 characters
        if [ ${#OBJECT_NAME} -lt 20 ]; then
            echo "Skipping folder: $OBJECT_NAME (name is too short)"
            continue
        fi

        echo "Starting training for: $OBJECT_NAME"
        
        # Run the training with ns-train
        ns-train splatfacto --data "$OBJECT_DIR"
        
        # Find the latest output folder after training
        LATEST_OUTPUT=$(ls -dt "$OUTPUT_BASE_DIR"/$OBJECT_NAME*/splatfacto/* | head -n 1)
        CONFIG_FILE="$LATEST_OUTPUT/config.yml"
        
        # Check if the config file exists before exporting
        if [ -f "$CONFIG_FILE" ]; then
            EXPORT_PATH="$EXPORT_DIR/splats_$OBJECT_NAME"
            echo "Exporting PLY for: $OBJECT_NAME"
            ns-export gaussian-splat --load-config "$CONFIG_FILE" --output-dir "$EXPORT_PATH"
            echo "Exported PLY to: $EXPORT_PATH"
        else
            echo "Config file not found for $OBJECT_NAME. Skipping export."
        fi

        echo "Finished training and exporting for: $OBJECT_NAME"
    fi
done

echo "All objects have been processed and exported!"