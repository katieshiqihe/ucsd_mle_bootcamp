# Image Processing and Computer Vision

There are lots of CNN architectures for computer vision problems. For the
colorization problem, I chose a GAN architecture with stride set to 1 for all
convolutional layers (no up/downsizing). The original papers argues that
in the image colorization, it is important to keep all spatial information for
more precise boundary detection and that the convolution operation itself is
capable of feature extraction without downsizing or pooling.

However, there are pre-trained models for image recognition/segmentation (VGG,
ResNet, EfficientNet, etc.) that might be suitable for the task. One approach is
to use a pre-trained model in the generator to reduce training time (and
perhaps complexity). A caveat is that all those models expect the input to have
three channels (`RGB` color space) while our input images in this task only has
one channel (grayscale). Thus the first input layer and corresponding weights
need to be modified. Another approach is to use a pre-trained model in the loss
function. Currently the generator compares pixel-wise squared loss, but it might
benefit from computing MSE on the reduced feature space by passing the generated
output and ground truth through an image recognition model.

Finally, both generator and discriminator models are very complex with millions
of weights each. The gradients might also vanish through the many layers and
cause the model to stop learning. Using smaller kernels and fewer filters might
not damage the overall performance by much while reducing the computation time
and memory usage significantly. Pruning and quantization can be applied to
further reduce complexity and avoid over-fitting. Skip connections can be used
to avoid the gradient vanishing problem.
