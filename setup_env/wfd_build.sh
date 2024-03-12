set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conda create --name wfd_build python=3.10 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate wfd_build
pip install -r $SCRIPT_DIR/wfd_requirements.txt
echo "$SCRIPT_DIR/.." | tee $(python -c "import site;print(site.getsitepackages()[0])")/workspace.pth > /dev/null
