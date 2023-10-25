set -x   # Show which command is being run

if [[ ${mode} == "testing" ]]; then

    # Download answers

    curl -OL http://astro.utah.edu/~u1281896/software_development/cluster_generator/${ANSWER_VER}.tar.gz
    tar -zxf ${ANSWER_VER}.tar.gz


fi

# Install dependencies using conda

PYVER=`python --version`

conda install --yes numpy pytest pip h5py astropy tqdm cython scipy yt
git clone https://github.com/eliza-diggins/cluster_generator -b pull-request-2
cd cluster_generator
pip install .
cd ..

if [[ ${mode} == "wheels" ]]; then
  conda install --yes wheel setuptools
fi

if [[ ${mode} == "testing" ]]; then
  # Install cluster_generator
  python -m pip install -e .
fi
