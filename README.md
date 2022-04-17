 <div align="center">    
  
# Differentiable Time-Frequency Scattering in Kymatio

  [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
  <!--
  ARXIV   
  [![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
  -->

  <!--  
  Conference   
  -->   
  </div>
   
Examples of Time-Frequency Scattering in Kymatio, for audio signals

## How to run   

First, install dependencies

### Installation v1

Pip only, no IDE

```bash
# clone project   
git clone https://github.com/rastegah/kymatio-jtfs

# install project   
cd kymatio-jtfs
pip install -e .
pip install -r requirements.txt
pip install git+https://github.com/kymatio/kymatio.git@refs/pull/674/head
 ```   

### Installation v2

With Anaconda & Spyder

```bash
# clone project   
git clone https://github.com/rastegah/kymatio-jtfs

# install project
cd kymatio-jtfs
# for much faster install resolution
conda install mamba
conda create kymj
conda activate kymj
mamba env update --file conda-env-kymj.yml
pip install --file pip-env-kymj.txt

pip install -e git+https://github.com/mathieulagrange/doce.git@3ad246067c6a8ac829899e7e888f4debbad80629#egg=doce
pip install git+https://github.com/PyTorchLightning/metrics.git@3af729508289d3babf0e166d9e8405cb2b0758a2
pip install git+https://github.com/OverLordGoldDragon/kymatio.git@refs/pull/16/head
```

### Run Jupyter

```bash
# module folder
   cd kymatio-jtfs

# run notebooks
jupyter notebook
```

### Citation   
```
@article{YourName,
  title={Your Title},
    author={Your team},
      journal={Location},
        year={Year}
}
```   

