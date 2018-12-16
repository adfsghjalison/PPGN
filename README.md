# Plug-and-Play Text Style Transfer

## Prerequisites

1. Python 2.7 or higher
2. tensorflow-gpu 1.0.0 or 1.1.0

## Clone this repository
We need three parts of this project.  
  
`PPGN`  
`|--PPGN_Style_Transfer`  
`|--PPGN_Style_Classifier`  
`'--PPGN_VRAE`  
  
1. `mkdir PPGN`
2. Clone `PPGN_Style_Transfer` to PPGN  
`git clone https://github.com/adfsghjalison/PPGN_Style_Transfer.git`  
3. Clone `PPGN_Style_Classifier` to PPGN  
`git clone https://github.com/adfsghjalison/PPGN_Style_Classifier.git`  
4. Clone `PPGN_VRAE` to PPGN  
`git clone https://github.com/adfsghjalison/PPGN_VRAE.git`  


### Train
*  Train Style Classifier for Plug-and-Play model  
Check `https://github.com/adfsghjalison/PPGN_Style_Classifier.git`  

* Train VAE-RNN for Plug-and-Play model  
Check `https://github.com/adfsghjalison/PPGN_VRAE.git`  

### Inference
`python main.py --mode stdin`

### Important Hyperparameters of the flags.py
`model_dir` : model directory of VRAE  
`date_dir` : data directory of VRAE  
`style_model_dir` : model directory of Style Classifier  
`data_name` : database name  
`batch_size` : batch size  
`latent_size` : latent dimension of VRAE  
`sequence_length` : max length of input and output sentence  
`KL_annealing` : use KL annealing or not  
`word_dp` : word dropout rate  

`g` : style gradient weight  
`l2` : L2 norm gradient wight  

## Files

### Files
`flags.py` : all settings  
`utils.py` : data processing  
`model.py` : model architecture  
`main.py` : main function  

