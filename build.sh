set -e
set -x

rm -r build
uv build ./package/mask_propagation/Cutie --out-dir build
uvx cibuildwheel ./package --only cp310-manylinux_x86_64 --output-dir build
uvx cibuildwheel ./package --only cp310-win_amd64 --output-dir build
