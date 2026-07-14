<img width="1388" height="914" alt="Screenshot 2026-07-13 195305" src="https://github.com/user-attachments/assets/f4abdc85-acad-4e71-8473-957aef3c57a9" />

# FedorAiToolkit

A fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) that adds **DRaFT-K reward training for Krea 2**: after a normal SFT stage, the LoRA/LoKr is optimized *directly* on differentiable rewards — face identity and body geometry — computed on images the model generates during training. Everything upstream (all models, the UI, normal training) works unchanged. **Ideogram 4 DRaFT is in progress.**

## Quick start

**Already running stock ai-toolkit?** Clone this fork into its **own folder** — do not mix it with your existing install. Each copy needs its own `venv`.

```bash
git clone https://github.com/CliffNodes/FedorAiToolkit.git
cd FedorAiToolkit
```

### One-time setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Linux/macOS: same flow with `python3 -m venv venv` and `source venv/bin/activate`. See the upstream README below for pinned torch versions and platform notes.

Optional shortcut: if you use a `.bat` to start stock ai-toolkit, duplicate it and change every path from your old folder to `FedorAiToolkit`.

### Launch the UI

Requires [Node.js](https://nodejs.org/) 20+.

```bat
cd ui
npm run build_and_start
```

Open **http://localhost:8675**. Create a job → pick **Krea 2** → enable **Face / Body Reward Stage (DRaFT)**. The UI only needs to run while you start or monitor jobs; training continues if you close the browser.

### Train from the command line

```bat
venv\Scripts\activate
python run.py config/examples/krea2_lokr_draft.yaml
```

Edit the yaml first: set `trigger_word`, point `datasets` and `draft.reward.reference_images` at your image folder (photos + matching `.txt` captions).

### Example configs

| Config | GPU | Notes |
|---|---|---|
| `config/examples/krea2_lokr_draft.yaml` | 24–32 GB | LoKr + face/body DRaFT (reference) |
| `config/examples/krea2_lora_draft.yaml` | 24–32 GB | LoRA + face/body DRaFT |
| `config/examples/krea2_lokr_draft_5060_8gb.yaml` | ~8 GB (RTX 5060) | 512-only, low VRAM, **face-only** DRaFT (`body_weight: 0`) |

Face-only DRaFT works without SAM 3D Body. Full body reward needs extra setup — see below.

## What this fork adds

```
Stage 1 (SFT, stock ai-toolkit)          Stage 2 (krea2_draft_trainer)
subject image dataset                    differentiable sampling
        |                                (grad through last K flow steps
        v                                 + checkpointed VAE decode)
diffusion_trainer                                |
LoRA or LoKr on Krea 2   -- resumed -->          v
                                         face reward  +  body reward
                                         (ArcFace vs   (SAM 3D Body -> MHR
                                          reference     shape -> SOMA-X
                                          images)       canonical mesh)
                                                 |
                                                 v
                                         weighted reward loss -> backprop
```

| Piece | Where | What it does |
|---|---|---|
| `krea2_draft_trainer` | `extensions_built_in/krea2_draft_trainer/` | DRaFT-K for Krea 2. Ports the loop from [KONAKONA666/krea-2](https://github.com/KONAKONA666/krea-2) (Apache-2.0): no-grad for the first `steps - draft_k` denoising steps, gradient through the last K + VAE decode, optional DRaFT-LV samples. Resumes SFT-trained LoRA **and** LoKr; `draft.train_modules: qkvo` restricts updates to attention wq/wk/wv/wo adapters. |
| Face reward | `toolkit/rewards/face.py` | Vendored differentiable `FaceSimilarityReward` (ArcFace/antelopev2, auto-downloads). Saturated centroid reward vs your reference images, EOT augmentations, anti-copy entropy, duplicate-identity penalty. |
| Body reward | `toolkit/rewards/body.py` | New `BodyGeometryReward`: SAM 3D Body regresses MHR shape params from generated images *with gradients*, SOMA-X maps them to canonical neutral-pose vertices, compared pose-independently against a prototype built from your reference images. Pose is discarded by design — only build/proportions/height are rewarded. Tier-2 fallback compares raw shape params if the Warp vertex path is unavailable. |
| Combined reward | `toolkit/rewards/combined.py` | `face_weight * face + body_weight * body`, degrades gracefully to face-only when the gated SAM 3D Body checkpoint is missing. |
| Example configs | `config/examples/krea2_lokr_draft.yaml`, `krea2_lora_draft.yaml`, `krea2_lokr_draft_5060_8gb.yaml` | Two-stage (SFT then DRaFT) reference configs for LoKr/LoRA on Krea 2. |

### Extra requirements for the body reward

Body training (the SAM 3D Body mesh reward in the DRaFT stage) does **not** work out of the box. You must set up Meta's SAM 3D Body checkpoint first. Face-only DRaFT works without it — if the checkpoint is missing, body reward is skipped and training continues on face identity only.

**Before you enable body training:**

1. **Request access** on Hugging Face — the checkpoint is gated:
   - [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) (recommended)
   - Approval can take a little while; you cannot download until Meta grants access.
2. **Log in** with a token that has read access:
   ```bash
   hf auth login
   ```
   Or set `HF_TOKEN` in your environment.
3. **Clone the SAM 3D Body code** (not on PyPI):
   ```bash
   git clone https://github.com/facebookresearch/sam-3d-body repositories/sam-3d-body
   ```
4. **Download the checkpoint** — on first body-reward run it auto-downloads via the Hugging Face hub if you are authenticated. Or prefetch manually:
   ```bash
   hf download facebook/sam-3d-body-vith --local-dir models/sam-3d-body-vith
   ```
   Expected layout: `models/sam-3d-body-vith/model.ckpt` and `models/sam-3d-body-vith/assets/mhr_model.pt`.

**Other deps:**

- `pip install py-soma-x warp-lang insightface onnx onnxruntime-gpu` (plus SAM 3D Body's deps from its repo)
- Optional: `pip install pyrender` if you want mesh overlay exports from `scripts/export_sam_body_scans.py`

Without the checkpoint, you can still run Stage 1 (SFT) and face-only DRaFT (`body_weight: 0`). Set `body_weight` > 0 only after SAM 3D Body is installed and downloaded.

### Running a two-stage training

```bash
# CLI: runs Stage 1 (SFT) then Stage 2 (DRaFT) sequentially
python run.py config/examples/krea2_lokr_draft.yaml
```

Point `datasets` / `draft.reward.reference_images` at your dataset, set your trigger word, and adjust `draft.reward.face_weight` / `body_weight`. Stage 2 reward steps are typically **15–60** (`draft.num_reward_steps` in the UI). VRAM note: face-only DRaFT at 512px fits easily on a 24 GB card; face+body needs ~32 GB on Krea 2 — use `low_vram: true`, qfloat8, and close other GPU apps.

---

Original upstream README follows.

# Ostris AI Toolkit

AI Toolkit is an easy to use all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.



## Supported Models

### Image
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)

### Instruction / Edit
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)

### Video
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)

### Audio
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)

### Experimental
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)

## Installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

MacOS:

Experimental support for Silicon Macs is available. I do not have a Mac with enough RAM to fully test this
so please let me know if there are issues. There is a convience script to install and run on MacOS 
locates at `./run_mac.zsh` that will install the dependencies locally and run the UI. To run this, 
do the following:

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
chmod +x run_mac.zsh
./run_mac.zsh
```


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 20

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Ostris Cloud

You can use many cloud providers to rent GPUs. If you want to help support this project in the largest way possible, please consider using [Ostris Cloud](https://cloud.ostris.com). Ostris Cloud is owned and operated by me, Ostris, and every dollar earned goes directly back into funding the development of this project.

<a href="https://cloud.ostris.com" target="_blank"><img src="https://cloud.ostris.com/api/og" alt="Ostris Cloud" style="max-width:100%;width:600px;height:auto;"></a>


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [my Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support this project.


I maintain an official Runpod Pod template here which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

I have also created a short video showing how to get started using AI Toolkit with Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

<a href="https://ostris.com/sponsors" target="_blank"><img src="https://ostris.com/wp-content/uploads/2025/05/support-banner2.png" alt="Support my work" style="max-width:100%;height:auto;"></a>

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

<a href="https://ostris.com/sponsors"><img src="https://ostris.com/sponsors.svg" alt="Sponsors" style="width:100%;height:auto;"></a>
