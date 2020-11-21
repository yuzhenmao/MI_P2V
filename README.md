# MI_P2V

## Models

./runner.py: main function

./core/MIgenerate.py: train mutual-information-net

./core/train.py: the original train-net

./core/mi_train.py: mutual information train-net

./core/predict.py: generate visulization (input: image file  output: 32*32*32 voxel)


./models/MILoss.py: calculate MI loss

./models/Res3D.py: resnet (decode inouot voxels)

./models/MIFC.py: mutual information fully connected layer

./models/Trans_decoder.py: not used!

## Dataset

./Dataset: need to download data from:

ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

After downloading the data, you need to update settings in config.py. 

Please follow the code in:https://gitlab.com/hzxie/Pix2Vox/-/tree/master

## Get started

To train original Pix2Vox, you can simply use the following command:
```
python3 runner.py --batch-size=32
```
To train mi_loss net, you can use the following command:
```
python3 runner.py --mcmi --weights=/path/to/pretrained/MILoss/model.pth
```
To train mi_Pix2Vox, you can use the following command:
```
python3 runner.py --mitrain --weights=/path/to/generate/model.pth --mi_weights=/path/to/MILoss/model.pth --batch-size=16/32
```
To generate voxels, you can use the following command:
```
python3 runner.py --pred --weights=/path/to/pretrained/model.pth --img=/path/to/img
```
To test original Pix2Vox, you can use the following command:
```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```
