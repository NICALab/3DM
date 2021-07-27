<h2 align="center">3DM: Deep decomposition and deconvolution microscopy</h2>

<p align="center">
<img width="60%" src="src/Visualization1_resize75.gif">
</p>
<h6 align="center">Demo video acquired using 3DM.

 Compression is applied. See Supplementary video 1.
 </h6>

Official source codes for "3DM: Deep decomposition and deconvolution microscopy", Optics Express.


## Running BEAR

You can run BEAR which does Robust PCA by very simple code.
Just download or clone this repository and use the code below.
`--D` refers the name of the data and `--d` enables the default settings of several hyperparameters.

```bash
python scripts/run_BEAR.py --D hall --d True
```

Results will be saved in `results/BEAR/hall_(Y/L/S)` directory.

## Hardware specification & Requirements

**Hardware specification**

```markdown
OS : Ubuntu 18.0.4
CPU : Intel Xeon Silver 4214 CPU @ 2.20GHz
GPU : GeForce RTX 2080 Ti 11GB
RAM : 128GM
```

**Requirements (May be okay with different versions.)**
```markdown
torch==1.7.0
numpy==1.19.4
psutil==5.7.2

scipy==1.5.3
scikit-image==0.17.2
mat73==0.46
```

## Reproduce the paper

We have already added small surveillance video, and calcium imaging data of mouse which are widely used in RPCA paper.
You can first try these data.
Calcium imaging data of zebrafish we have used in paper is in this [Google Drive](https://drive.google.com/file/d/115lCnwIVU0TtKedQ_31FDaOG8wksGmG9/view?usp=sharing) (~10GB).
Download, unzip, and move .tif and .mat files inside the data folder. 

1. Figure 3. Phase diagram.
```bash
python scripts/phase_diagram.py --D None --d None
```

2. Table 1. and Figure 4. Decomposition of the zebrafish caclium imaging data.
```bash
python scripts/run_Greedy_BEAR.py --D zebrafish_150 --d True
python scripts/run_Greedy_BEAR.py --D zebrafish_1000 --d True
```
Due to the size of data, loading files itself does take long time (Minutes in HDD).
And for 1000 length video, about 100GB of RAM is required.

3. Figure 5. and Figure 6. Cascaded BEAR for analysis of neuronal activity.
```bash
python scripts/run_Cascaded_BEAR.py --D demoMovie --d True
python scripts/run_Cascaded_BEAR.py --D spinning_confocal --d True
```
For accuracy, number of epochs in default settings of Cascaded BEAR is very large.
Can be observed that loss value does not decrease actually after small number of epochs.
You can reduce the `args.epoch` if you want.

## Citation
```markdown
@inproceedings{han2021efficient,
  title={Efficient neural network approximation of robust PCA for automated analysis of calcium imaging data},
  author={Seungjae Han and Eun-Seo Cho and Inkyu Park and Kijung Shin and Young-Gyu Yoon},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention},
  year={2021}
}
```