set -e

cd ..
mkdir musc-test-installation
cd musc-test-installation

conda create -p venv python
conda activate ./venv
pip install ../musc
python -c 'from musc.high_level import *; print(dir())'
conda deactivate

cd ..
rm -r musc-test-installation
