set -e

cd ..
git clone musc musc-test-installation-dev
cd musc-test-installation-dev

conda create -p venv python
conda activate ./venv
pip install pdm
pdm install
python examples/test-installation-dev.py
conda deactivate

cd ..
rm -r --interactive=never musc-test-installation-dev
