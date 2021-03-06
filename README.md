# CvTMorph
CvTMorph: Better Registration on Local Structures for Respiratory Motion Modeling

keyword: 4D-CT lung, image registration, vision transformer

## Create Conda Virtual Environment
```
conda create -n CvTMorph python=3.6 
```
```
conda activate CvTMorph
```

install python package.
For using a 2080Ti GPU. Use the following command:
```
pip install torch1.7.0+cu101 torchvision0.8.2+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Or, using a 3090 GPU. Use the following command:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Or choose a pip that matches your GPU
```
https://pytorch.org/get-started/previous-versions/
```
Then
```
pip install nibabel==3.2.1 ml_collections==0.1.0 pystrum==0.1 torchsummary==1.5.1 einops==0.3.2 torch-smmary==1.4.4
```
## Prepare Data
### Note that the training directory must be modified in the train.py code
### For ours experiment settings
|&nbsp;-&nbsp;Patients   
&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;patient_01    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;years1_month1_day1    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;vols    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung01.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung02.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;- . . .   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;years2_month2_day2    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;vols    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung01.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung02.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;- . . .   
&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;patient_02    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;years1_month1_day1    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;vols    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung01.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung02.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;- . . .   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;years2_month2_day2    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;vols    
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung01.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;lung02.nii.gz   
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;- . . .   

### Or subject to subject base on your dataset
|&nbsp;-&nbsp;TrainSet  
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp;vols  
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp; img0.nii.gz  
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp; img1.nii.gz  
&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;|&nbsp;-&nbsp; . . .

## Train&Test
```
python ./train.py
```
For testing, we provide a simple case to test and you need to adapt it for your own test cases.
```
python ./test.py
```

## Reference:
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[CvT](https://github.com/microsoft/CvT)

[VoxelMorph](https://github.com/voxelmorph/voxelmorph)

[ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)

[SyN/ANTsPy](https://github.com/ANTsX/ANTsPy)

[Elastix](https://github.com/SuperElastix/SimpleElastix)

## AboutUs
AI in medical image processing at XMUT  
Yifan Guo(?????????), Xuan Pei(??????), Ting Wu(??????), Jiayang Guo(?????????),  Dahan Wang(?????????), Peizhi Chen(?????????)
