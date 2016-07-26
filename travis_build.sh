set -e
git clone git://github.com/nocturnalastro/astropy.git
cd astropy
git checkout sherpa_bridge_v2
python setup.py install
conda install -c sherpa sherpa
cd ../

