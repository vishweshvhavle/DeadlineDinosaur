#!/bin/bash
# Setup script for FastLanczos CUDA extension

echo "Building FastLanczos CUDA extension..."
cd deadlinedino/submodules/lanczos-resampling

# Build the extension
python setup.py install --user

echo "FastLanczos build complete!"
echo ""
echo "To verify installation, run:"
echo "  python -c 'from FastLanczos import lanczos_resample; print(\"FastLanczos imported successfully!\")'"
