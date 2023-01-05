# Multiclass segmentation on Liver Tumor (HCC) using ResUnet
Contributors: Lu George george6.lu@polyu.edu.hk,

Department: HTI Hong Kong Polytechnic University

Deep learning framework: Pytorch

## 1. Dataset
This dataset is downloaded from kaggle competition *Liver Tumor Segmentation*, the solutions from this competition also provides a feasible method for solving this tumor segmentation.

### 1.1 Source

This dataset was extracted from LiTS – Liver Tumor Segmentation Challenge (LiTS17) organised in conjunction with ISBI 2017 and MICCAI 2017.

[Dataset Article](https://arxiv.org/abs/1901.04056)

[Kaggle Main Page](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation/code)

### 1.2 Acknowledgements
Splash banner

Image by ©yodiyim

Splash icon

Icon made by Freepik available on www.flaticon.com.

License

CC BY NC ND 4.0

BibTeX

@misc{bilic2019liver,
title={The Liver Tumor Segmentation Benchmark (LiTS)},
author={Patrick Bilic and Patrick Ferdinand Christ and Eugene Vorontsov and Grzegorz Chlebus and Hao Chen and Qi Dou and Chi-Wing Fu and Xiao Han and Pheng-Ann Heng and Jürgen Hesser and Samuel Kadoury and Tomasz Konopczynski and Miao Le and Chunming Li and Xiaomeng Li and Jana Lipkovà and John Lowengrub and Hans Meine and Jan Hendrik Moltz and Chris Pal and Marie Piraud and Xiaojuan Qi and Jin Qi and Markus Rempfler and Karsten Roth and Andrea Schenk and Anjany Sekuboyina and Eugene Vorontsov and Ping Zhou and Christian Hülsemeyer and Marcel Beetz and Florian Ettlinger and Felix Gruen and Georgios Kaissis and Fabian Lohöfer and Rickmer Braren and Julian Holch and Felix Hofmann and Wieland Sommer and Volker Heinemann and Colin Jacobs and Gabriel Efrain Humpire Mamani and Bram van Ginneken and Gabriel Chartrand and An Tang and Michal Drozdzal and Avi Ben-Cohen and Eyal Klang and Marianne M. Amitai and Eli Konen and Hayit Greenspan and Johan Moreau and Alexandre Hostettler and Luc Soler and Refael Vivanti and Adi Szeskin and Naama Lev-Cohain and Jacob Sosna and Leo Joskowicz and Bjoern H. Menze},

year={2019},

eprint={1901.04056},

archivePrefix={arXiv},

primaryClass={cs.CV}
}

## 2. Object
The main object of this project is to classify liver and tumor pixels on a series of 2d radiology slices. Therefore, it is a multilabel segmentation problem, which means each pixel belongs to background(label_0), liver(label_1) or the tumor on the liver(label_2).

## 3. Model
Unet is most widely used segmentation model framework as it can extract high-level features(encoding) and combine it with low-level features(decoding) at the same time. Resnet, Residual Network has a good performance on training the deep neural network earning reputation for its elegant solution to the problem of gradient vanish by introducing extra path of shortcuts to skip the convolution layers connection.

For this project, we remain the framework of unet but replace the convolution layers with residual blocks to smooth the training. Since the task is three-label classification, we take off the sigmoid layer as output layer. Therefore, input shape is: [B, 1, H, W] and output shape is: [B, 3, H, W] where each channel of the output images represent the confidence of classifying the pixel to the correspoinding class.

## 4. Preprocessing
The slices from the dataset are the raw radiology images with minimun value  -3024.0 and maximum value 1410.0 with size (512, 512, 1). The raw dataset is then windowed to [0, 255] and swap channel so that a single slice has shape [1, 512, 512], then we resize them to size [1, 256, 256]. We filter out the slices where there is no liver which means the segmentation mask contains no less than two labels. Finally, we store the image tp png format. Here, the segmentation labels only contain three possible labels: 0, 1, 2 of format uint8.

## 5. DataLoader and Augmentation
To address overfitting problem, we implement data augmentation: random horizon flipping with 50% chance, random rotation by 15 degree. The dataset is encapsulated into a torch.util.data.Dataset subclass, the dataset directly read the preprocessd data from disk. We split Dataset as portion with training set 0.8 and test 0.2. Then we pass the dataset to instanitiate a new DataLoader, the DataLoader will shuffle and batch the dataset before feeding the data to model.


## 6. Metrics
Metrics encompasses the loss function and validation metrics. Selecting loss fucntion for segmentation need very sophisticated design since the labels distribution for each slice is imbalanced. Specifically, most of the pixels belong to background so the model will easily fall to the pitfall of lack of distinguishing the minority labels. If we use multiclass similarity index(Jaccrad, Dice, Precision), the loss function might not be convex?

### 6.1 Loss function
We select PixelWiseDiceCE as the loss function, the loss fucntion inherits from torch.nn. It calculates crossentropy for each label pixel against the softmax of the corrsponding three-channel output values. It also calculate the Dice Index of label and image. The final loss is the sum of crossentropy and (1 - Dice).

We assign different weights for missing each label in the crossentropy, higheset weight goes to label 2(tumor), and lowest to label 0(background), the setting encourages the model to focus on the tumor and liver pixels, but will also raise the possibility of fasle positive. 

### 6.2 Accuracy
1. normal_accuracy: the loss function only measures the percentage of correctly classified pixel, this loss function equally trate each pixel label so that the gradient will descent extremly slow after a few epochs since most of the label belong to background.
2. weighted_accuracy: similar to the weighted loss in the loss function. Classitying correctly the tumor pixel will significantly increase the accuracy.
3. dice_coefficient: similarity between output and label, here we directly use the implementation of multiclass dice coefficient from library *torchmetrics*: [DICE](https://torchmetrics.readthedocs.io/en/stable/classification/dice.html?highlight=dice).
4. jaccard: similar to dice_coefficient, we use the implementation of multiclass jaccrad index from *torchmetrics*: [JACCARD INDEX](https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html#module-interface).
5. precision: $TP / (TP + FP)$ Where $TP$ and $FP$ represent the number of true positives and false positives respecitively, we use the implementation from *torchmetrics*: [PRECISION]([https://torchmetrics.readthedocs.io/en/stable/classification/precision.html?highlight=precision)

## 7. Training
In the training section, we made setting as follow:
1. Loss function: PixelWiseDiceCE, please refer to previous section for details.
2. Optimizer: Adam
3. Learning_rate_scheduler: MultiStepLR, the scheduler will decay learning rate by reaching ibe if milestones by gamma.
We record loss and metrics for each 100 steps and print the validation loss and metrics for each epcoh. 

The training setting and hyperparameter need further adjustments.

## 8. Furture Plan(temporary)
Notice: THIS SECTION IS NOT PART OF FINAL REPORT AND SHOULD BE DELETED AFTER FINISHING!

1. Add normalization layers to address the poor performance caused by different scale.
2. Figure out the best loss function.
3. Optimize the loss function.
4. Generalize model to 3d slices, set up a threshold portion of tumor volume to liver volume and finally classify the patients as hcc or not.
5. Collaborative work using git.
