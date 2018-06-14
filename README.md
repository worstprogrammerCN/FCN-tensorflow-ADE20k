# FCN8s tensorflow ADE20k
## 1. introduction 
This is a **fully-connected network(8 strides)** implementation on the dataset **ADE20k**, using tensorflow.
The implementation is largely **based on** the paper [arXiv: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) and 2 implementation from other githubs: [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow) and [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch).

net | dataset | competition | framework | arXiv paper
------------ | ------------- | --------- | ------| -----|
FCN8s | [ADE20k](http://groups.csail.mit.edu/vision/datasets/ADE20K/) | [MIT Scene Parsing Benchmark (SceneParse150)](http://sceneparsing.csail.mit.edu/) | tensorflow 1.4 and python 3.6 | [arXiv: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
## 2. how to run
1. Download and extract the dataset.
    1. Download the .zip dataset. link: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    2. Put the .zip dataset under **./Data_zoo/MIT_SceneParsing/** and extract.
2. Start **training**. Simply run ```FCN_train.py```
3. To **test** the model, simply run ```FCN_test.py```. 
    It will test the first 100 validation images by default. The validation dataset contains 2000 images, so if you want to test more images, simply modified the variable ```TEST_NUM```, e.g. 1000.
4. To use the model to **infer** images, 
   1. Put the .jpg images you want to infer in the folder **./infer**
   2. Make sure there is a folder **./output** (to store the result). 
   3. Run ```FCN_infer.py```, it will process all **.jpg** images under **./infer** and put the predicted annotations under **./output**

## 3. code logic
- Each time process **a batch containing 2 images**. 
If the size of 2 images/annotations are different, **enlarge** the smaller image/annotation **so that they are the same size**. The enlarged size must be rounded so that **it can be divided by 32**(because the size is downsized 32 times at most, when processed through the FCN network).
- I use the function ```scipy.misc.imresize``` to **resize**. 
The function param ```interp```(interpolation) for **resizing images** is ```"bilinear"```, while that for **resizing annotations** is ```"nearest"```.
- The optimizer is **Adam Optimizer** with **learning rate 1e-5**.

## 4. running
### 4.1 train
When you run ```FCN_train.py```, you will see:
![](https://upload-images.jianshu.io/upload_images/7547741-5bcc53e384b8828e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- It will cost about 7~8 hours on *Nvidia GeForce GTX 1080 11GB*.
- The training loss is hard to be improved when it's around 0.8 ~ 1.5. This may be the limitation of FCN.

### 4.2 test
When you run ```FCN_test.py```, you will see:
![](https://upload-images.jianshu.io/upload_images/7547741-0d613e5e0192909a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
and after processing all validation images, It will print the metrics.
![](https://upload-images.jianshu.io/upload_images/7547741-8396505c4239ec91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
You can uncomment lines 130~143 to **see some well processed results**.
![](https://upload-images.jianshu.io/upload_images/7547741-8fa831a7a192bb51.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 4.3 infer
Just put the .jpg images in **./infer**, run **./FCN_infer.py** and the predicted annotations will be put in **./output**.

## 5. results
The metrics of testing 100 validation images is:

pixel_accuracy | mean_accuracy | mean IU(mean iou) | frequency weighted IU
---|---|---|---|
0.6739| 0.4332 | 0.3644 | 0.5024

(There are 151 classes, where class index 0 is *"others"*. You just need to care the accuracy on the 150 classes)

Here are some examples:

![](https://upload-images.jianshu.io/upload_images/7547741-ae150695ac950d02.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

