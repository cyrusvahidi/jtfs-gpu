 <div align="center">    
  
# Differentiable Time-Frequency Scattering in Kymatio 🌊

### source code by
[John Muradeli](github.com/overlordgolddragon), [Cyrus Vahidi](cyrusvahidi.com)<sup>2</sup> <br>
[Changhong Wang](https://changhongw.github.io/)<sup>1</sup>, [Han Han]()<sup>1</sup> <br>
[Vincent Lostanlen](lostanlen.com)<sup>1</sup>
LS2N/CNRS Centrale Nantes <sup>2</sup>  Queen Mary University of London 
 
many thanks to all opens-source contributors of ![Kymatio](https://github.com/kymatio/kymatio) and dependent libraries


  [![Paper](http://img.shields.io/badge/paper-arxiv.2204.08269-B31B1B.svg)](https://arxiv.org/abs/2204.08269)
  <!--
  ARXIV   
  [![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
  -->

  <!--  
  Conference   
  -->   
  </div>
   
This repository is for the paper "Differentiable Time-Frequency Scattering in Kymatio" (accepted to DAFx 2022).

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
git clone https://github.com/rastegah/kymatio-jtfs

# install project   
cd kymatio-jtfs
pip install -e .
pip install -r requirements.txt

# install kymatio from source
cd kymatio
pip install -r requirements.txt
python setup.py install
```   

An implementation of time-frequency scattering will be available in the unreleased ![WaveSpin](https://github.com/overLordGoldDragon/wavespin) package

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

### Citation   
```
@article{muradeli2022differentiable,
  title={Differentiable Time-Frequency Scattering in Kymatio},
  author={John Muradeli, Cyrus Vahidi, Changhong Wang, Han Han, Vincent Lostanlen, Mathieu Lagrange, George Fazekas},
  journal={arXiv preprint arXiv:2204.08269},
  year={2022}
}
```   

