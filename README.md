# InfNeRF: Towards Infinite Scale NeRF Rendering with O(log n) Space Complexity

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

<video controls src="https://jiabinliang.github.io/InfNeRF.io/video/winworld_720.mp4"></video>

Result of UrbanScene3D Residence, rendering with < 16% of the model:

<video src="https://jiabinliang.github.io/InfNeRF.io/video/residence_720.mp4" controls></video>


Result of UrbanScene3D Sci Art:

<video src="https://jiabinliang.github.io/InfNeRF.io/video/sci_720.mp4" controls></video>

Result of Mill 19 Building:

<video src="https://jiabinliang.github.io/InfNeRF.io/video/building_720.mp4" controls></video>

Result of Mill 19 Rubble:

<video src="https://jiabinliang.github.io/InfNeRF.io/video/rubble_720.mp4" controls></video>

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

## Install

Refer to the nerfstudio environment installation: [Installation](https://docs.nerf.studio/quickstart/installation.html).

## Training

Registering infnerf dataparser with nerfstudio:

```
pip install -e .
```

Training command:

```
ns-train inf-nerf --data ./data/building-pixsfm/data/images
```

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
