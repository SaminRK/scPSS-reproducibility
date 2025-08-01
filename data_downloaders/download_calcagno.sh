#!/bin/bash

# Check for input argument
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/DATA_DIR"
    exit 1
fi

# Input argument
DATA_DIR="$1"
DATA_DIR="$(realpath "$DATA_DIR")"

# Create directory
mkdir -p "$DATA_DIR"

METADATA_URL="https://zenodo.org/records/7055957/files/Nikatag/Single-Cell-Spatial-Transcriptomics-for-Border-zone-BZ_Cell_Mapping.zip?download=1"
ZIP_OUTPUT="$DATA_DIR/Single-Cell-Spatial-Transcriptomics-for-Border-zone-BZ_Cell_Mapping_zenodo.zip"

echo "==> Downloading Zenodo ZIP..."
wget -c "$METADATA_URL" -O "$ZIP_OUTPUT"

echo "==> Unzipping Zenodo ZIP..."
unzip -o "$ZIP_OUTPUT" -d "$DATA_DIR"

mv "$DATA_DIR"/Nikatag-Single-Cell-Spatial-Transcriptomics-for-Border-zone-*/* "$DATA_DIR"/

echo "==> Downloading GEO dataset..."
wget -c "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE214611&format=file" \
     -O "$DATA_DIR/GSE214611_RAW.tar"

echo "==> Extracting GEO .tar..."
tar -xvf "$DATA_DIR/GSE214611_RAW.tar" -C "$DATA_DIR"

echo "==> Extracting all .tar.gz files..."
cd "$DATA_DIR"
for f in *.tar.gz; do
    tar -xvzf "$f" -C "$(dirname "$f")"
done

echo "==> Flattening filtered_feature_bc_matrix folders..."

find "$DATA_DIR" -type d -name "filtered_feature_bc_matrix" | while read -r ffbm_dir; do
    parent_dir="$(dirname "$ffbm_dir")"
    echo "  Moving contents of $ffbm_dir to $parent_dir"
    mv "$ffbm_dir"/* "$parent_dir"/
    rmdir "$ffbm_dir"
done


echo "✅ Done. Files are in: $DATA_DIR"
