#!/bin/bash

# Validate input arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 TOTAL_ROWS TRAIN_RATIO VALIDATION_RATIO TEST_RATIO"
    exit 1
fi

TOTAL_ROWS=$1
TRAIN_RATIO=$2
VALIDATION_RATIO=$3
TEST_RATIO=$4

DATA_DIR="./data"
ATTACKS_DIR="${DATA_DIR}/attacks"
DETECTION_DIR="${DATA_DIR}/detection_data"
mkdir -p "$DETECTION_DIR"

BENIGN_FILENAME_PATTERN="*Benign*.csv"
BENIGN_MULTIPLIER=20  # Benign to non-Benign ratio

# Extract header from one of CSV files
HEADER=$(head -n 1 "$(find $ATTACKS_DIR -type f | head -n 1)")

# Initialize CSV files for datasets with headers
echo "$HEADER" > "${DETECTION_DIR}/training.csv"
echo "$HEADER" > "${DETECTION_DIR}/validation.csv"
echo "$HEADER" > "${DETECTION_DIR}/testing.csv"

# Calculate total number of non-Benign classes
NUM_CLASSES=$(find "$ATTACKS_DIR" -type f | grep -v "$BENIGN_FILENAME_PATTERN" | wc -l)

# Calculate total rows for Benign and non-Benign, ensuring total row count is not exceeded
TOTAL_NON_BENIGN_ROWS=$(($TOTAL_ROWS / (1 + $BENIGN_MULTIPLIER)))
TOTAL_BENIGN_ROWS=$(($TOTAL_NON_BENIGN_ROWS * $BENIGN_MULTIPLIER))

distribute_rows() {
    local file=$1
    local total_rows=$2
    local training_rows=$(echo "$total_rows*$TRAIN_RATIO" | bc | awk '{printf "%d\n", $1}')
    local validation_rows=$(echo "$total_rows*$VALIDATION_RATIO" | bc | awk '{printf "%d\n", $1}')
    local testing_rows=$(echo "$total_rows*$TEST_RATIO" | bc | awk '{printf "%d\n", $1}')
    
    # Training
    head -n $training_rows "$file" >> "${DETECTION_DIR}/training.csv"
    # Validation
    head -n $(($training_rows + $validation_rows)) "$file" | tail -n $validation_rows >> "${DETECTION_DIR}/validation.csv"
    # Testing
    tail -n $testing_rows "$file" >> "${DETECTION_DIR}/testing.csv"
}

# Process each CSV file
for file in $ATTACKS_DIR/*.csv; do
    if [[ $file == *Benign*.csv ]]; then
        distribute_rows "$file" $TOTAL_BENIGN_ROWS
    else
        distribute_rows "$file" $TOTAL_NON_BENIGN_ROWS
    fi
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
    cleanup_headers "${DETECTION_DIR}/${dataset}.csv"
done

echo "Datasets prepared in $DETECTION_DIR."
