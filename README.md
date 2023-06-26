# Neural-Intrinsic-Embedding

This repository is a PyTorch implementation of [Neural-Intrinsic-Embedding](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_Neural_Intrinsic_Embedding_for_Non-Rigid_Point_Cloud_Matching_CVPR_2023_paper.pdf).


### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Installing PyTorch may require an ad hoc procedure, depending on your computer settings.

### Data & Pretrained models
You can find the data and the pre-trained models in:
```
data
models
```


### Evaluation

To evaluate the model  FAUST\SCAPE, run:

```eval
python code/faust/test_faust_sample.py
or
python code/scape/test_scape_sample.py
```
And in matlab the script:
```eval
code/eval/FAUST_5k.m
or 
code/eval/SCAPE_5k.m
```

### Training

Coming soon.




### License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

If you use this code, please cite our paper.

```
@inproceedings{jiang2023neural,
  title={Neural Intrinsic Embedding for Non-rigid Point Cloud Matching},
  author={Jiang, Puhua and Sun, Mingze and Huang, Ruqi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21835--21845},
  year={2023}
}
```

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). 
For any commercial uses or derivatives, please contact us.