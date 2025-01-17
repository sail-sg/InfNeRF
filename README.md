# InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity (Siggraph Asia 2024)

| [Project Page](https://jiabinliang.github.io/InfNeRF.io/) | [Paper](https://dl.acm.org/doi/pdf/10.1145/3680528.3687646) |<br>

[Jiabin Liang](https://jiabinliang.github.io)<sup>1</sup>, Lanqing Zhang<sup>1</sup>, [Zhuoran Zhao](https://alicezrzhao.github.io/)<sup>1,2</sup>, [Xiangyu Xu](https://xuxy09.github.io/)<sup>3</sup>

<sup>1</sup> [Sea AI Lab](https://sail.sea.com/)
<sup>2</sup> [National University of Singapore](https://nus.edu.sg/),
<sup>3</sup> [Xi'an Jiaotong University](https://www.xjtu.edu.cn/)

## Overview
![alt text](./img/teaser.png "Title")

- InfNeRF extends the proven LoD technique to Neural Radiance Fields (NeRF) by introducing an octree structure to represent the scenes in different scales. 
- This innovative approach provides a mathematically simple and elegant representation with a rendering space complexity of $\mathcal{O}(\log n)$, aligned with the efficiency of mesh-based LoD techniques.
- We also present a novel training strategy that maintains a complexity of $\mathcal{O}(n)$. 
This strategy allows for parallel training with minimal overhead, ensuring the scalability and efficiency of our proposed method. 
- Our contribution is not only in extending the capabilities of existing techniques but also in establishing a foundation for scalable and efficient large-scale scene representation using NeRF and octree structures.

## Demo
Result of Window of the World, ShenZhen, rendering with < 17% of the model:

![winworld](./img/winworld_720.gif)

Result of UrbanScene3D Residence, rendering with < 16% of the model:

![residence](./img/residence_720.gif)

Result of UrbanScene3D Sci Art:

![residence](./img/sci_720.gif)

Result of Mill 19 Building:

![residence](./img/building_720.gif)

Result of Mill 19 Rubble:

![residence](./img/rubble_720.gif)

Refer to our [project page](https://jiabinliang.github.io/InfNeRF.io/) for more high-resolution rendering results.

## Data Preparation

### Mill 19

- The Building scene can be downloaded [here](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz).
- The Rubble scene can be downloaded [here](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz).

### UrbanScene 3D
Download the raw photo collections from the [UrbanScene3D](https://github.com/Linxius/UrbanScene3D?tab=readme-ov-file#urbanscene3d-v1) dataset
  - [Residence](http://szuvccnas.quickconnect.cn/d/s/lSvWkTMbFjecrEwZDx3cV72M5scS2tKA/OxnMJCCChFCGAqEHfVC09VJmO_f-qrga-_LFAaeS27Ag)
  - [Sci-Art](http://szuvccnas.quickconnect.cn/d/s/lT61obCnx48mOc1FrPtUiuZ8eNCOrEQd/27C8eKMNd1YBpLxJTbYY-jMWU7vRHhbs-5bHAJ9227Ag)
  - [Campus](http://szuvccnas.quickconnect.cn/d/s/lRrBh8QyqmVQnXgn6Lc41vqnpeZej5bm/Xj3MGE2nOmr9CR_q09lJzYzmtcUGc5XQ-67Hgr9-27Ag)

After downloading all the raw images, use COLMAP to obtain the camera poses:

```
ns-process-data images --data ./data/building-pixsfm/data/images --output-dir ./data/building-pixsfm/data --sfm-tool colmap --skip-image-processing --gpu
```

<br>

We have provided the COLMAP results for the Residence dataset: [Google Drive](https://drive.google.com/file/d/1X8EZRSwOopLWPQXjUCDczpaV0TW9ErHw/view?usp=sharing). The data structure for InfNeRF training would be like:

```
- Residence
  - sparse
    - 0
      - cameras.bin
      - images.bin
      - points3D.bin
      - project.ini
  - images
    - A
      - DJI_0413.JPG
      ...
    - B
      - DJI_0001.JPG
      ...
    - C
      - DJI_0001.JPG
      ...
```

## Install

Refer to the nerfstudio environment installation: [Installation](https://docs.nerf.studio/quickstart/installation.html).

## Training

Registering infnerf dataparser with nerfstudio:

```
pip install -e .
```

Training command:

```
ns-train inf-nerf --data ./Residence
```

You can use tensorboard to see the visualization of the evaluation results and metrics:
```
tensorboard --logdir=./outputs/Residence/inf-nerf/2025-01-06_143012 --port=6010
```
<img src="./img/tensorboard.png" width="700">

## Citation

If you find this project useful, please consider citing:

<pre><code>@inproceedings{10.1145/3680528.3687646,
    author = {Liang, Jiabin and Zhang, Lanqing and Zhao, Zhuoran and Xu, Xiangyu},
    title = {InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity},
    year = {2024},
    url = {https://doi.org/10.1145/3680528.3687646},
    doi = {10.1145/3680528.3687646},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
}</code></pre>
