# MI_P2V

A pixel to voxel 3D generator. Based on Pix2Vox++ (https://arxiv.org/abs/2006.12250).

I improved this model based on (https://arxiv.org/abs/2007.02919).

All relevant information (results): (https://drive.google.com/drive/folders/1YAkheiplXTw39F4UUWsMr54SiWPiZt7I)

## Models

./runner.py: main function

./core/MIgenerate.py: train mutual-information-net

./core/train.py: the original train-net

./core/mi_train.py: mutual information train-net

./core/predict.py: generate visulization (input: image file  output: 32*32*32 voxel)


./models/MILoss.py: calculate MI loss

./models/Res3D.py: resnet (decode input voxels)

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
To generate, you can use the following command:
```
python3 runner.py --gen --mi_weights=output/checkpoints/2020-11-24T21:48:25.507368/plane_mitrain_checkpoint-best.pth --weights=output/checkpoints/2020-11-24T07:38:31.125174/plane_checkpoint-best.pth
```
