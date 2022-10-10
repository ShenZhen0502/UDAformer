# UDAformer

------

This repository is the official PyTorch implementation of UDAformer: underwater image enhancement based on dual attention transformer. UDAformer achieves **state-of-the-art performance** in underwater image enhancement task.

*Underwater images suffer from seriously degraded due to wavelength-dependent light scatter and abortion of the underwater environment, which impacts the application of high-level computer vision tasks. Considering the characteristics of uneven degradation and loss of color channel of underwater images, a novel underwater image enhancement method, namely UDAformer, based on Dual Attention Transformer Block (DATB), including Channel Self-Attention Transformer (CSAT) and Shifted Window Pixel SelfAttention Transformer (SW-PSAT), is proposed. Specifically, the underwater image enhancement based on channel self-attention alone is not necessarily sufficient due to the severe and uneven degradation of underwater images. Therefore, a novel fusion method combining channel and pixel self-attention is proposed for efficient encoding and decoding of underwater image features. Then, the shifted window method for the pixel self-attention is proposed to improve computational efficiency. Further, convolution is constructed to calculate the self-attention weight matrix so that the proposed UDAformer can flexibly process input images of various resolutions and reduce network parameters. Finally, the underwater images are recovered through the design of skip connections based on the underwater imaging model. Experimental results demonstrate the proposed UDAformer surpasses previous state-of-the-art methods, both qualitatively and quantitatively.*

![](https://github.com/sz19980502/UDAformer/blob/main/figures/overall.png)

------

#### Results

You can see all the images and videos results in our [Google Drive](https://drive.google.com/drive/folders/13ehthtYe7GSnQWaTZsnTa2n_08RNlF22)

#### Training

If you need to train our UDAformer from scratch, you need to download UIEB dataset from [link](https://li-chongyi.github.io/proj_benchmark.html).  Put UIEB dataset into the folder "*data*": put underwater image into underwater subfolder and ground truth image into ground truth folder. 

Environmental requirements:

- einops == 0.3.0; numpy == 1.20.3;  Python == 3.9.7;  Torch == 1.11.0 + cu113   OpenCv-Python == 4.5.1.48

Run train.py, the model weight will automatilly save in checkpoint folder. In order to visualize the training process, the images generated during the training will be saved in the sample folder.

#### Testing

Put the model weight into checkpoint folder, and run test.py.  

After run the test.py, you can see the result in results folder.

#### Citation

```
@article{shen2022udaformer,
  title={UDAformer: Underwater image enhancement based on dual attention transformer},
  author={Shen, Zhen and Xu, Haiyong and Luo, Ting and Song, Yang and He, Zhouyan},
  journal={Available at SSRN 4162641},
  year={2022}
}
```

#### Acknowledgement

The codes are designed based on [Restormer](https://github.com/swz30/Restormer).