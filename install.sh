
# install thirdparty
cd thirdparty

cd DKM
python setup.py install

cd ../DepthCov/
pip install -e . 


cd ../../


# install BA
cd droid_ipf
python setup.py install
cd ..

# install ext
python ext/__init__.py

# install U2-mvd
pip install -e .
