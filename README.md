# Quick Verification


```
python main.py --k=0
```

## Dataset Preparation

The training code expects the hyperspectral datasets in a folder named `Dataset/`:

```
Dataset/
├── Botswana.mat
├── Botswana_gt.mat
├── Indian_pines_corrected.mat
├── Indian_pines_gt.mat
├── SalinasA_corrected.mat
├── SalinasA_gt.mat
├── KSC.mat
└── KSC_gt.mat
```

Please download the required `.mat` files and place them under this directory.
If the files are missing, running `main.py` will fail with `FileNotFoundError`.

### CPU Training

GPU usage is enabled by default. To run on CPU, start the script with:

```
python main.py --cuda False
```


# Citation
If you use code or datasets in this repository for your research, please cite our paper.
```
@ARTICLE{10462168,
  author={Luo, Fulin and Liu, Yi and Duan, Yule and Guo, Tan and Zhang, Lefei and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SDST: Self-Supervised Double-Structure Transformer for Hyperspectral Images Clustering}, 
  year={2024},
  volume={62},
  pages={1-14},
  keywords={Transformers;Feature extraction;Clustering methods;Optimization;Self-supervised learning;Hyperspectral imaging;Clustering algorithms;Autoencoder;graph convolution;hyperspectral image (HSI) clustering;self-supervised learning;transformer},
  doi={10.1109/TGRS.2024.3374597}}
```
