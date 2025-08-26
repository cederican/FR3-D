### Installation

**Step 1.** Set up conda environment.

```
conda create --name fr3d python=3.8 -y
conda activate fr3d
```

**Step 2.** Install PyTorch.
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 
```

**Step 3.** Install pytorch3d, torch-cluster, chamferdist packages.
```
# install pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
cd ..

# install torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

# install chamferdist
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist && python setup.py install
cd ..
```

**Step 4.** Install remaining packages.

```
git clone https://github.com/cederican/FR3-D.git
cd FR3-D/
pip3 install -r requirements.txt
```