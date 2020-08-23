### CA-Net: Comprehensive attention Comvolutional Neural Networks for Explainable Medical Image Segmentation
This repository provides the code for "CA-Net: Comprehensive attention Comvolutional Neural Networks for Explainable Medical Image Segmentation". The paper can be found at: 

![mg_net](./pictures/canet.png)
Fig. 1. Structure of CA-Net.

![uncertainty](./pictures/uncertainty.png)
Fig. 2. Skin lesion segmentation.

![refinement](./pictures/refinement.png)

Fig. 3. Placenta and fetal brain segmentation.

### Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.

Follow official guidance to install [Pytorch][torch_link]. Install the other required packages by:
```
pip install -r requirements.txt
```

[torch_link]:https://pytorch.org/

### How to use
After installing the required packages, add the path of `UGIR` to the PYTHONPATH environment variable. 
### Demo of MG-Net
1. Run the following commands to use MG-Net for simultanuous segmentation and uncertainty estimation. 
```
cd uncertainty_demo
python ../util/custom_net_run.py test config/mgnet.cfg
```
2. The results will be saved to `uncertainty_demo/result`. To get a visualization of the uncertainty estimation in an example slice, run: 
```
python show_uncertanty.py
```

### Demo of I-DRLSE
To see a demo of I-DRLSE, run the following commands:
```
cd util/level_set
python demo/demo_idrlse.py 
```
The result should look like the following.
![i-drlse](./pictures/i-drlse.png)

### Copyright and License
Copyright (c) 2020, University of Electronic Science and Technology of China.
All rights reserved. This code is made available as open-source software under the BSD-3-Clause License.
