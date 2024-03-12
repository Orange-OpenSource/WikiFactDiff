# MIT License

# Copyright (c) 2022 Kevin Meng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file was originaly developed by Kevin Meng but modified by Hichem Ammar Khodja

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
