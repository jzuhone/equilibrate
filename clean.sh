find . -name "*.so" -exec rm -v {} \;
find . -name "*.pyc" -exec rm -v {} \;
rm -rvf build dist
rm -rvf cluster_generator.egg-info
