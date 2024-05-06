#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <Total number of Rows> <Training Ratio> <Validation Ratio> <Test Ratio>"
    echo "Example $0 10000 0.7 0.2 0.1"
    exit 1
fi

TOTAL_ROWS=$1
TRAIN_RATIO=$2
VALIDATION_RATIO=$3
TEST_RATIO=$4

DATA_DIR="./data"
ATTACKS_DIR="${DATA_DIR}/attacks"
CLASSIFICATION_DIR="${DATA_DIR}/classification_data"
mkdir -p "$CLASSIFICATION_DIR"

# Extract header from one of CSV files
HEADER=$(head -n 1 "$(find $ATTACKS_DIR -type f | head -n 1)")

# Initialize CSV files for datasets with headers
echo "$HEADER" > "${CLASSIFICATION_DIR}/training.csv"
echo "$HEADER" > "${CLASSIFICATION_DIR}/validation.csv"
echo "$HEADER" > "${CLASSIFICATION_DIR}/testing.csv"

# Function to distribute rows with cycling
distribute_rows() {
    local file=$1
    local ratio_start=$2
    local ratio_end=$3
    local target_file="${CLASSIFICATION_DIR}/$4.csv"
    local lines_to_copy=$5
    local total_lines=$(($(wc -l < "$file") - 1)) # Excluding header
    
    # Calculate starting and ending line numbers based on ratios
    local start_line=$(echo "$total_lines * $ratio_start / 1 + 1" | bc | awk '{print int($1)}')
    local end_line=$(echo "$total_lines * $ratio_end / 1" | bc | awk '{print int($1)}')
    local segment_lines=$(($end_line - $start_line + 1))

    while [ $lines_to_copy -gt 0 ]; do
        if [ $lines_to_copy -ge $segment_lines ]; then
            # Copy segment and decrease lines_to_copy accordingly
            tail -n +$start_line "$file" | head -n $segment_lines >> "$target_file"
            lines_to_copy=$(($lines_to_copy - $segment_lines))
        else
            # Copy remaining lines
            tail -n +$start_line "$file" | head -n $lines_to_copy >> "$target_file"
            break
        fi
    done
}

# Determine how many rows each dataset should contribute per class
NUM_CLASSES=$(find "$ATTACKS_DIR" -type f | wc -l)
ROWS_PER_CLASS_TOTAL=$(echo "$TOTAL_ROWS / $NUM_CLASSES" | bc)
ROWS_PER_CLASS_TRAIN=$(echo "$ROWS_PER_CLASS_TOTAL * $TRAIN_RATIO / 1" | bc)
ROWS_PER_CLASS_VALID=$(echo "$ROWS_PER_CLASS_TOTAL * $VALIDATION_RATIO / 1" | bc)
ROWS_PER_CLASS_TEST=$(echo "$ROWS_PER_CLASS_TOTAL * $TEST_RATIO / 1" | bc)

for file in $ATTACKS_DIR/*.csv; do
    distribute_rows "$file" 0 $TRAIN_RATIO "training" $ROWS_PER_CLASS_TRAIN
    distribute_rows "$file" $TRAIN_RATIO $(echo "$TRAIN_RATIO + $VALIDATION_RATIO" | bc) "validation" $ROWS_PER_CLASS_VALID
    distribute_rows "$file" $(echo "$TRAIN_RATIO + $VALIDATION_RATIO" | bc) 1 "testing" $ROWS_PER_CLASS_TEST
done

# Clean up redundant headers
cleanup_headers() {
    local file=$1
    # Directly read header from file to be cleaned
    local header=$(head -n 1 "$file")
    # Create a temporary file for cleaned data
    local temp_file=$(mktemp)

    # Write header to temporary file
    echo "$header" > "$temp_file"

    # Check for and append rows not matching header
    awk -v hdr="$header" 'NR > 1 && $0 != hdr' "$file" >> "$temp_file"

    # Replace original file with cleaned version
    mv "$temp_file" "$file"
}

# Iterate through dataset CSVs to remove redundant headers
for dataset in training validation testing; do
    cleanup_headers "${CLASSIFICATION_DIR}/${dataset}.csv"
done

echo "Datasets prepared in $CLASSIFICATION_DIR."
