#!/bin/bash
set -e

# Start from directory of script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

RECIPE="$SCRIPT_DIR/memit.yml"
ENV_NAME="wfd_eval"
echo "Creating conda environment ${ENV_NAME}..."

if [[ ! $(type -P conda) ]]
then
    echo "conda not in PATH"
    echo "read: https://conda.io/docs/user-guide/install/index.html"
    exit 1
fi

if df "${HOME}/.conda" --type=afs > /dev/null 2>&1
then
    echo "Not installing: your ~/.conda directory is on AFS."
    echo "Use 'ln -s /some/nfs/dir ~/.conda' to avoid using up your AFS quota."
    exit 1
fi

# Build new environment
conda env create --name=${ENV_NAME} -f ${RECIPE} -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo "$SCRIPT_DIR/../evaluate" | tee $(python -c "import site;print(site.getsitepackages()[0])")/workspace.pth > /dev/null
