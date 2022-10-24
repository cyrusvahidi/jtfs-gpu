 <div align="center">    
  
# Differentiable Time-Frequency Scattering on GPU 🌊

### source code by
[Cyrus Vahidi]()<sup>2</sup> <br>
[Changhong Wang]()<sup>1</sup>, [Han Han]()<sup>1</sup> 
[Vincent Lostanlen]()<sup>1</sup> <br>
[John Muradeli]() <br>
LS2N/CNRS Centrale Nantes<sup>1</sup>  Queen Mary University of London<sup>2</sup>


  [![Paper](http://img.shields.io/badge/paper-arxiv.2204.08269-B31B1B.svg)](https://arxiv.org/abs/2204.08269)
  <!--
  ARXIV   
  [![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
  -->

  <!--  
  Conference   
  -->   
  </div>
  
  
![Accompanying webpage 🌐](https://cyrusvahidi.github.io/jtfs-gpu/) <br>
![Kymatio: open-source wavelet scattering in Python ‍💻](https://github.com/kymatio/kymatio/)
  
Many thanks to all open-source contributors to ![Kymatio](https://github.com/kymatio/kymatio) and its dependencies.
   
This repository contains code to replicate the paper "Differentiable Time-Frequency Scattering on GPU" (published at DAFx 2022, best paper award).

Time-frequency scattering is available in ![Kymatio](https://github.com/kymatio/kymatio/) in beta and will be released in v0.4. To use this implementation, install Kymatio from source. To replicate the results in this paper, follow the installation instructions below.

We assess Time-Frequency Scattering in Kymatio for 3 machine listening research applications:

* unsupervised manifold learning of spectrotemporal modulations
* hybrid jtfs + convnet supervised musical instrument classification
* texture resynthesis 

<!-- We also provide scale-rate visualizations:

<img src="https://user-images.githubusercontent.com/16495490/163857080-9ae52cad-9202-4fb8-a1f5-a7d008f19073.png" alt="signal" width="800">
<img src="https://user-images.githubusercontent.com/16495490/163851994-b35772b0-5f73-4eef-8417-26ad02bbb65c.png" alt="scale-rate" width="750">
 -->

## How to run   

First, install dependencies

### Installation

```bash
# clone project   
git clone https://github.com/cyrusvahidi/jtfs-gpu

# install project   
cd kym-jtfs
pip install -e .
pip install -r requirements.txt

# install kymatio from source
cd ../kymatio
python setup.py develop
```   

* The JTFS algorithm source code to replicate the paper can be found ![here](https://github.com/overLordGoldDragon/wavespin/tree/dafx2022-jtfs)
* The latest version of JTFS can be installed directly from the ![Kymatio](https://github.com/kymatio/kymatio/) source

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

### Isomap Visualizations & K-NN Regression
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

### Notebook Guide

#### Scale-Rate Visualisations

H E L L O

#### Synthetic amplitude-modulated chirp dataset
* Factors of variation: 
 * $f_c$ carrier frequency
 * $f_m$ amplitude modulation frequency
 * $\gamma$ chirp rate

#### Manifold Embedding of the Nearest Neighbour Graph
* MFCCs
* Time Scattering
* Time-Frequency Scattering
* Open-L3
* Spectrotemporal Receptive Field (STRF)

#### K-NN regression of synthesizer parameters

#### 2-D CNN classifier

#### Differentiable Resynthesis 

### Citation   
```
@article{muradeli2022differentiable,
  title={Differentiable Time-Frequency Scattering in Kymatio},
  author={John Muradeli, Cyrus Vahidi, Changhong Wang, Han Han, Vincent Lostanlen, Mathieu Lagrange, George Fazekas},
  journal={arXiv preprint arXiv:2204.08269},
  year={2022}
}
```   

