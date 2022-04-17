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

### ConvNet Classifier

[Download Medley-solos-DB](https://zenodo.org/record/3464194)

#### Extract Scattering Features
``` bash
python scripts/process_msdb_features.py --data_dir <your_msdb_dir> --feature <feature_to_extract>
```

#### Configure gin
In `/scripts/gin/config.gin` set `MSDB_DATA_DIR` and `MSDB_CSV` according to the absolute path of your MSDB download.

#### Run training

``` bash
python scripts/train_cnn.py
```

### Isomap Manifold Visualizations & K-NN Regression
this generates the AM/FM dataset and extracts its MFCCs, Time Scattering Coefficients, JTFS coefficients, STRFs (`/strf-like-model`) and Open-l3 embeddings. The Isomap algorithm is run on each feature and 3 principal components are visualized. K-NN regression is performed using the Isomap K-NN graph.

``` bash
python scripts/isomap.py
```
see the output in `/img`

### Scale-Rate Visualizations and Resynthesis
```
cd notebooks
jupyter notebook
```
See `Scale-Rate Visualization.ipynb` and `Resynthesis results.ipynb`

### Citation   
```
@article{YourName,
  title={Your Title},
    author={Your team},
      journal={Location},
        year={Year}
}
```   

