# NetVLAD-pytorch-with-demo
NetVLAD training, testing code and a small demo for the model.

The model training procedure is based on [NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247).<br> 
The code has refered to [pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad/tree/8f7c37ba7a79a499dd0430ce3d3d5df40ea80581) , [MixVPR: Feature Mixing for Visual Place Recognition](https://github.com/amaralibey/MixVPR) , [GSV-cities](https://github.com/amaralibey/gsv-cities)

## Updates
Pretrained GhostNetVLAD checkpoints is available at `demo/resources/netvlad_ghost.pth.tar`

The testing results on Tokyo247 dataset is:

| Dim | R@1 | R@5 | R@10 |
|-----|-----|-----|------|
| 32768 | 82.2 | 92.6 | 93.6|
| 4096 | 89.2 | 93.6 | 94.6 |

Note : I randomly sampled 20000 features to perform PCA-whitening. Thus, the test results in 4096 dim may be different if you do PCA by yourself.


## Packages dependencies
>training and testing:
- pytorch 
- torchvision
- sklearn
- faiss
- scipy
- numpy
- tensorboardX
- pytorch_metric_learning
- h5py

(Additional packages if you want to see the demo)

>demo:
- flask


***

The training code is under `/training` folder and the small demo is under `/demo` folder

To run the training code, you should also put *pittsburgh* dataset and the *initdata* for initialization of the conv layer for the netvlad pooling layer in the training folder. These two files can be retrieved from [here](https://www.di.ens.fr/willow/research/netvlad/) under additional data and downloads from the paper's website or [here](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/) for the dataset.

To run the demo, copy the checkpoints that training code produce to the `/demo/resources` folder and rename the hardcode path for loading checkpoints in the python script. Then cd to the demo file and run command `python app.py`. This would run the flask application for the website. Then go to `localhost:5000` on your browser, you should be able to see a website like this
<img width="1434" alt="截屏2022-10-01 16 59 34" src="https://user-images.githubusercontent.com/92624596/193401785-331876d5-3a59-4d85-824f-fd11c53fc68a.png">

Then select your query image on your computer, the best 3 matching pictures from the database will show up. The database here only contains 10 images contain some of the landmarks in NTU and was retrived from Google streetview so make sure your query image contains these landmarks or you can replace the database with your own and encode the database with the model.

<img width="1359" alt="截屏2022-10-01 17 10 05" src="https://user-images.githubusercontent.com/92624596/193402163-79190696-63e6-4300-a6f6-0aa321698f23.png">
<img width="1310" alt="截屏2022-10-01 17 10 14" src="https://user-images.githubusercontent.com/92624596/193402164-099b7db1-96f3-49ab-96ed-bbeb800c740e.png">
