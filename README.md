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
```bash
# clone project   
git clone https://github.com/rastegah/kymatio-jtfs

# install project   
cd kymatio-jtfs
pip install -e .
pip install -r requirements.txt
pip install git+https://github.com/kymatio/kymatio.git@refs/pull/674/head
 ```   
  Run Jupyter   
   ```bash
# module folder
   cd kymatio-jtfs

# run notebooks
jupyter notebook
```

### Notebook Guide

#### Scale-Rate Visualisations

#### Synthetic amplitude-modulated chirp dataset

#### Manifold Embedding of the Nearest Neighbour Graph
* MFCCs
* Time Scattering
* Time-Frequency Scattering
* Open-L3
* Spectrotemporal Receptive Field (STRF)

#### K-NN regression of f0, chirp rate and tremolo rate

#### 2-D CNN classifier

#### Differentiable Resynthesis 

### Citation   
```
@article{YourName,
  title={Your Title},
    author={Your team},
      journal={Location},
        year={Year}
}
```   

