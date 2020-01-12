# INSTALL

```
git clone git@github.com:thuyngch/transformers.git
cd transformers

conda create -n tfm python=3.7 -y
conda install numpy matplotlib cython -y
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
pip install opencv-python

pip install .
pip install -r ./examples/requirements.txt
```
