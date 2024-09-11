#!/bin/bash

# Specify the directory to loop over
directory="input/"

# Check if directory exists
if [ ! -d "$directory" ]; then
    echo "Directory $directory does not exist."
    exit 1
fi

# Loop over all files in the directory
for file in "$directory"/*.parquet; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
        # Extract the stem of the filename (before ".parquet")
        filename=$(basename -- "$file")
        stem="${filename%.parquet}"
        python SIREN_to_gSeaGen.py -s input/${filename} -k output/${stem}
    fi
done

echo "All files processed."