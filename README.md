## Super Resolution

#### * Work in progress *

This is an implementation in PyTorch of the super resolution CNN based in the paper: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

There are a few differences between the proposed network and my implementation:

- changed upsampling blocks to remove the "checkered board pattern" as sugested by [[2]](http://distill.pub/2016/deconv-checkerboard/)

- Changed deconv kernel size to 4 and added padding to all steps in order to get the same dimensions as the paper if you decide not to use the upsampling block from [2].

- Removed the gaussian noise when generating lower resolution images

The model was trained for 10 epochs on a random set of 20k images from the validation set of the MS-COCO dataset.


## Usage:

- To use provided model weights:
python run --model upsample4x.pth --target_image [PATH_TO_IMAGE_TO_UPSAMPLE]

The output of the model will be an image named 'up_IMAGENAME' on your current working folder.

- To train a new model:
python run.py --mode train --image_folder [PATH_TO_TRAIN_FOLDER] --trn_epochs [EPOCHS] --l_rate [LEARNING_RATE]

## To-do:

- Add a few good-to-haves on model (change loss function, process multiple images, prevent gray imgs, etc)

- Add list of dependencies

- Train on all of ImageNet dataset to improve results while avoiding overfitting.

## References

[1] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

[2] [Deconvolution and Checkerboard Artifacts](http://distill.pub/2016/deconv-checkerboard/)
