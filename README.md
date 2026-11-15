<div align="center">

# [DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation](https://utn-air.github.io/DUET-DINO/)

**Nisarga Nilavadi**<sup>1</sup>&nbsp;&nbsp;
**Ralf Römer**<sup>2</sup>&nbsp;&nbsp;
**Moritz Reuss**<sup>3,4</sup>&nbsp;&nbsp; 
**Michael Krawez**<sup>1</sup>&nbsp;&nbsp;
**Tobias Jülg**<sup>1</sup><br>
**Angela P. Schoellig**<sup>2,5</sup>&nbsp;&nbsp;
**Rudolf Lioutikov**<sup>3,5</sup>&nbsp;&nbsp; 
**Wolfram Burgard**<sup>1,5</sup>

<sup>1</sup> University of Technology Nuremberg &nbsp;&nbsp;
<sup>2</sup> Technical University of Munich &nbsp;&nbsp;
<sup>3</sup> Karlsruhe Institute of Technology  
<sup>4</sup> NVIDIA &nbsp;&nbsp;
<sup>5</sup> Robotics Institute Germany


</div>

## Installation

Clone and use a CUDA-enabled GPU device for training and inference.

```bash
git clone https://github.com/nisarganc/DUET-DINO.git
cd DUET-DINO
```

Create an environment with **Conda** or **venv**:

```bash
conda create -n duet-dino python=3.12 -y
conda activate duet-dino

python3.12 -m venv .duet-dino
source .duet-dino/bin/activate
```

Install dependencies in the activated environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download checkpoints

For DUET-DINO with DINOv3, download the **ViT-H+/16 LVD-1689M** encoder from
[Meta's DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
Use the download URL provided after accepting the license:

```bash
mkdir -p checkpoints
wget 'YOUR_DINOV3_DOWNLOAD_URL' -O checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

Download the trained DUET-DINO predictor:

```bash
wget https://huggingface.co/UTN-AIR/DUET-DINO/resolve/main/duet-dino.pt -O checkpoints/duet-dino.pt
```

## Docker Setup

Install Docker and the NVIDIA Container Toolkit, then build and run from the repository root:

```bash
docker build -t duet-dino:latest -f Dockerfile .
./run_docker.sh
```

The script opens a Bash shell with all GPUs, mounts the repository and local
`checkpoints/` and `output/` directories, and exposes port `8000`.
Run `./run_docker.sh --help` for configuration options.

## Start the policy server

Select the [config file](configs/inference), verify the checkpoint fields, and start the TCP policy server:

```bash
python -m inference.serve_policy \
      --cfg-file angledreach/droid-224px-8f-duet.yaml \
      --host 0.0.0.0 \
      --port 8000
```

`--cfg-file` is relative to `configs/inference/`. Connect your robot/evaluation client to this model server.

## Baseline Inference
For the V-JEPA 2 baseline, download the respective encoder:

```bash
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -O checkpoints/vitg.pt
```

Download the trained DUET-VJEPA predictor:
```bash
wget https://huggingface.co/YOUR_ORG/DUET-DINO/resolve/main/duet-vjepa2.pt -O checkpoints/duet-vjepa2.pt
```

## Development / training

Edit [configs/train/droid-224px-8f.yaml](configs/train/droid-224px-8f.yaml) to set:

- `data.datasets`: Your prepared DROID and RoboArena LMDB directories.
- `meta.pretrain_dinocheckpoint`: The downloaded DINOv3 encoder path.
- `folder`: To your output directory and adjust `data.batch_size` for your GPUs.
- `wandb.entity`: Your wandb entity or set `wandb.enabled: false`.
- `meta.resume_checkpoint: null`: Training resumes from `folder/latest.pt` if it exists. Otherwise, it starts fresh.


Start training. The training launcher starts one worker per listed GPU:

```bash
python -m train.main \
  --fname configs/train/droid-224px-8f.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

## Acknowledgments

Our code builds on [V-JEPA 2 / V-JEPA 2.1](https://github.com/facebookresearch/vjepa2)
and [DINOv3](https://github.com/facebookresearch/dinov3). We thank their authors
for releasing their code. Please also cite the relevant source papers:

- [Assran et al.](https://arxiv.org/abs/2506.09985), V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, 2025.
- [Mur-Labadia et al.](https://arxiv.org/abs/2603.14482), V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning, 2026.
- [Siméoni et al.](https://arxiv.org/abs/2508.10104), DINOv3, 2025.

## License

See [LICENSE](LICENSE) for the licensing scope and source headers for attribution.

- DUET-DINO's original code and the V-JEPA-derived code are licensed under [MIT](LICENSE).
- The files in `src_dino/` retain the [DINOv3 License](licenses/DINOv3.txt).
- The files `randaugment.py` and `randerase.py` in `train/video_datasets_utils/` retain [Apache 2.0](licenses/Apache-2.0.txt).
