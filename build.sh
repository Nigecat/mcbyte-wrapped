set -e
set -x

uv build ./package/mask_propagation/Cutie --out-dir build
uvx cibuildwheel ./package --platform linux --output-dir build
