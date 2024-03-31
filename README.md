# Paint-it (CVPR 2024)
### [Project Page](https://kim-youwang.github.io/paint-it) | [Paper](https://arxiv.org/abs/2312.11360)
This repository contains the official implementation of the CVPR 2024 paper, 
"🎨 Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering".

<img width="960" alt="teaser_camready" src="https://github.com/postech-ami/paint-it/assets/55628873/e7069c3d-3fdf-4e01-b8dc-7e8a8f14d072">

## Highlights
**Paint-it is a text-driven high-quality PBR texture map synthesis method.** 

🌟 Our texture maps are ready for practical use in popular graphics engines like Blender and Unity, thanks to our Physics-based Rendering (PBR) parameterization, which includes diffuse, roughness, metalness, and normal information.

🎨 With our approach, the resulting texture maps are not only of superior quality but also offer the flexibility of relighting and material editing.

🔥 We've achieved impressive results without modifying the well-known Score-Distillation Sampling (SDS), instead focusing on optimizing variables through our texture map parameterization.

🔊 While many researchers are working on denoising the gradients from SDS, our work leverages the power of architectural bias, specifically Deep Image Prior, to robustly learn from noisy SDS gradients, even when dealing with PBR representations.


## Getting started
This code was developed on Ubuntu 18.04 with Python 3.8, CUDA 11.3 and PyTorch 1.12.0, using NVIDIA RTX A6000 (48GB) GPU. 
Later versions should work, but have not been tested.


### Environment setup

```
conda create -n paint_it python=3.8
conda activate paint_it

# pytorch installation
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# for pytorch3d installation
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# for python3.8, cuda 11.3, pytorch 1.12 (py38_cu113_pyt1120) -> need to install pytorch3d-0.7.2 
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html

pip install git+https://github.com/NVlabs/nvdiffrast/
pip install diffusers==0.12.1 huggingface-hub==0.11.1 transformers==4.21.1 sentence-transformers==2.2.2
pip install PyOpenGL PyOpenGL_accelerate accelerate rich ninja scipy trimesh imageio matplotlib chumpy opencv-python
pip install numpy==1.23.1
```

### Preparing 3D Mesh data
Currently, this repository contains a sample mesh from [Objaverse](https://objaverse.allenai.org/) dataset.
To download a subset of Objaverse, you can refer to the scripts provided [here](https://github.com/daveredrum/Text2Tex?tab=readme-ov-file#benchmark-on-objaverse-subset).


### Generate PBR texture maps for 3D mesh
Given a 3D mesh in `.obj` format and the text prompt, you can run below code to generate PBR texture maps.
```
# Generate PBR textures for .obj meshes
python paint_it.py
```

When generating PBR texture maps for a subset of Objaverse meshes, 
you can modify below dictionary (generate_pbr.py, L305) to handle multiple mesh object IDs and corresponding text prompts.

```
mesh_dicts = {
    '9ce8ab24383c4c93b4c1c7c3848abc52': 'a pretzel',
}
```

### Coming soon
- [x] Code release for general 3D object meshes (Objaverse)
- [ ] Code release for 3D human meshes
- [ ] Improved diffusion guidance (VSD, Stable-Diffusion XL, ControlNet-Depth, etc...)

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@inproceedings{youwang2024paintit,
    title = {Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering},
    author = {Youwang, Kim and Oh, Tae-Hyun and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024}
}
````


## Contact
Kim Youwang (youwang.kim@postech.ac.kr)


## Acknowledgement
We thank the members of [AMILab](https://ami.postech.ac.kr/members) and [RVH group](https://virtualhumans.mpi-inf.mpg.de/people.html) for their helpful discussions and proofreading. 

The implementation of Paint-it is largely inspired and fine-tuned from the seminal projects.
We would like to express our sincere gratitude to the authors for making their code public.
- Deep Image Prior (https://github.com/DmitryUlyanov/deep-image-prior)
- Stable-Dreamfusion (https://github.com/ashawkey/stable-dreamfusion)
- Fantasia3D (https://github.com/Gorilla-Lab-SCUT/Fantasia3D)

> The project was made possible by funding from the Carl Zeiss Foundation. This work is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 409792180 (Emmy Noether Programme, project: Real Virtual Humans), and the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039A. Gerard Pons-Moll is a Professor at the University of Tübingen endowed by the Carl Zeiss Foundation, at the Department of Computer Science and a member of the Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645. Kim Youwang and Tae-Hyun Oh were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2023-00225630, Development of Artificial Intelligence for Text-based 3D Movie Generation; No.2022-0-00290, Visual Intelligence for SpaceTime Understanding and Generation based on Multi-layered Visual Common Sense; No.2021-0-02068, Artificial Intelligence Innovation Hub).




