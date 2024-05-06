#!/bin/bash

# Check if a file path was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_your_csv_file>"
    exit 1
fi

FILE_PATH=$1

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "File does not exist: $FILE_PATH"
    exit 1
fi

DATA_DIR="./data"
ATTACKS_DIR="${DATA_DIR}/attacks"

# Ensure the output directories exist
mkdir -p $ATTACKS_DIR

# Pre-process to get header
HEADER=$(head -n 1 "$FILE_PATH")


# Single-pass processing of file
awk -F, -v header="$HEADER" -v attacksDir="$ATTACKS_DIR" '
NR == 1 {next} # Skip header
{
    attackType = $(NF-1);
    # Sanitize attackType to be used as filename
    gsub(/[ \/]/, "_", attackType);
    filename = attacksDir "/" attackType ".csv";
    
    if (!seen[filename]++) {
        print header > filename; # Add header if file is new
    }
    print > filename; # Append row to appropriate file

    count[attackType]++; # Count occurrences
}
END {
    # Write distribution to a file
    for (attackType in count) {
        print attackType, count[attackType];
    }
}' "$FILE_PATH" > "${DATA_DIR}/data_distribution.csv"


# Function to randomize rows of CSV file, keeps header in place
randomize_csv_rows() {
    local file=$1
    # Extract header and shuffle rows except for header
    (head -n 1 "$file" && tail -n +2 "$file" | shuf) > "${file}.tmp" && mv "${file}.tmp" "$file"
}

# Randomize rows in each generated CSV file
for file in "${ATTACKS_DIR}"/*.csv; do
    randomize_csv_rows "$file"
done

echo "Attack-specific CSV files have been generated and randomized in $ATTACKS_DIR."
echo "Data distribution has been saved to ${DATA_DIR}/data_distribution.csv."
