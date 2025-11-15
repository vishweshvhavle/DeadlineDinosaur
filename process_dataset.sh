#!/bin/bash

# Batch process all SLAM datasets

DATASET_DIR="data/Dataset"
OUTPUT_DIR="data/ProcessedDataset"

echo "Starting batch conversion..."
echo "Input:  $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run batch conversion
python convert_gs.py "$DATASET_DIR" -o "$OUTPUT_DIR" --batch

echo ""
echo "============================================"
echo "Conversion complete!"
echo "============================================"
echo ""
echo "Processed datasets are in: $OUTPUT_DIR"
