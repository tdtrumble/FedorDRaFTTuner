# Fedor DRaFT Tuner

A command-line-only, local-model runner for applying **DRaFT-K reward optimization** to an existing **Krea 2 LoRA or LoKr** adapter.

This fork deliberately does not provide normal supervised fine-tuning, dataset training, generation jobs, model extraction/merging, a web UI, Docker images, or support for architectures other than Krea 2. Start with a compatible `.safetensors` adapter trained elsewhere; this repository only performs the reward stage.

The runner also never downloads model weights. It enables offline mode for Hugging Face Hub, Transformers, and Diffusers, validates every required path before importing PyTorch, and loads Krea 2 components with `local_files_only=True`.

## Requirements

- Windows or Linux with an NVIDIA CUDA GPU
- Python 3.11 recommended
- Git; Git LFS is additionally required for the optional SOMA-X body reward
- A Krea 2-compatible LoRA or LoKr `.safetensors` file
- Roughly 24 GB VRAM for the reference 512px face-only configuration; body reward usually needs more. Actual use depends on quantization, adapter rank, batch size, and reward models.

Create a virtual environment and install PyTorch for your CUDA version first. For CUDA 12.8, for example:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For the optional SOMA-X body reward:

```bash
pip install -r requirements-body.txt
```

SAM 3D Body has additional dependencies in its own `INSTALL.md`; install those into the same environment after cloning it below.

## Download the model files yourself

The paths below match the sample configs. You may download with a browser or run the shown `hf download` commands manually. These commands are setup instructions only—the trainer never invokes them.

### Required for every run

| Component | Source | Local destination |
|---|---|---|
| Krea 2 Raw transformer | `krea/Krea-2-Raw`, file `raw.safetensors` | `models/krea2/raw.safetensors` |
| Qwen3-VL text encoder | `Qwen/Qwen3-VL-4B-Instruct` full snapshot | `models/qwen3-vl-4b-instruct/` |
| Qwen-Image VAE | `Qwen/Qwen-Image`, the `vae/` subtree | `models/qwen-image/vae/` |
| AntelopeV2 detector | `immich-app/antelopev2`, `detection/model.onnx` | `models/antelopev2/detection/model.onnx` |
| AntelopeV2 recognizer | `immich-app/antelopev2`, `recognition/model.onnx` | `models/antelopev2/recognition/model.onnx` |
| Your existing adapter | A Krea 2 LoRA/LoKr you already trained | `models/adapters/<name>.safetensors` |

Example manual commands:

```bash
hf download krea/Krea-2-Raw raw.safetensors --local-dir models/krea2
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/qwen3-vl-4b-instruct
hf download Qwen/Qwen-Image --include "vae/*" --local-dir models/qwen-image
hf download immich-app/antelopev2 --include "detection/model.onnx" "recognition/model.onnx" --local-dir models/antelopev2
```

Expected layout:

```text
models/
├── adapters/subject_lora.safetensors
├── antelopev2/
│   ├── detection/model.onnx
│   └── recognition/model.onnx
├── krea2/raw.safetensors
├── qwen-image/vae/...
└── qwen3-vl-4b-instruct/...
```

### Optional body reward

SAM 3D Body is gated. Request access to `facebook/sam-3d-body-vith`, then manually place its snapshot here:

```bash
hf download facebook/sam-3d-body-vith --local-dir models/sam-3d-body-vith
git clone https://github.com/facebookresearch/sam-3d-body repositories/sam-3d-body
git lfs install
git clone https://github.com/NVlabs/SOMA-X repositories/SOMA-X
git -C repositories/SOMA-X lfs pull
```

The required body layout is:

```text
models/sam-3d-body-vith/
├── model.ckpt
└── assets/mhr_model.pt
repositories/
├── sam-3d-body/sam_3d_body/...
└── SOMA-X/assets/...
```

SOMA-X normally downloads assets on first use; this fork prevents that by requiring `draft.reward.body.soma_data_root` and passing the local directory to `SOMALayer`.

## Prepare references and prompts

Put clear images of the same subject in `data/reference/`. Face reward needs visible faces in the reference set. Body reward benefits from uncropped, full-body views with varied poses.

Prompts may be listed directly under `draft.prompts`, or one per line in a text file. `[trigger]` is replaced by `trigger_word`. The tracked `config/examples/prompts/body_match.txt` pool emphasizes head-to-toe visibility, readable body proportions, several camera angles and poses, and a visible face so it works with the combined body and face reward.

```text
[trigger] portrait photo, natural light
[trigger] full body photo, standing outdoors
```

## Configure and run

Copy one of the samples:

- `config/examples/krea2_lora_draft.yaml`: LoRA, face-only, smallest setup
- `config/examples/krea2_lokr_draft.yaml`: LoKr, face plus body/SOMA-X

Edit at least these values:

- `adapter.path`, `adapter.type`, `adapter.rank`, and `adapter.alpha`; for LoKr also match `factor`, `full_rank`, and `old_format`
- all paths under `model`
- `draft.reward.reference_images` and the face/body model paths
- `draft.prompts` or `draft.prompts_path`
- `train.steps`, which is exactly the number of DRaFT optimizer updates

Validate without loading PyTorch or touching the GPU:

```bash
python run.py config/examples/krea2_lora_draft.yaml --validate
```

Then run:

```bash
python run.py config/examples/krea2_lora_draft.yaml
```

Multiple config files can be processed sequentially. Add `--recover` to continue after a failed job and `--log output/run.log` to save console output.

## Configuration reference

`adapter` is the input contract:

| Key | Meaning |
|---|---|
| `path` | Required local `.safetensors` input |
| `type` | `lora` or `lokr` only |
| `rank`, `alpha` | Must match the input adapter |
| `factor`, `full_rank`, `old_format` | LoKr construction values; must match the input |

Important DRaFT settings:

| Key | Meaning |
|---|---|
| `draft.steps` | Denoising steps used to generate each reward image |
| `draft.draft_k` | Number of final denoising steps kept in the gradient graph |
| `draft.train_modules` | `qkvo` for attention projections only, or `all` |
| `draft.save_every` | Adapter checkpoint interval in DRaFT steps; `0` disables intermediate checkpoints |
| `draft.save_images_every` | Reward-image interval; `0` disables images |
| `train.steps` | Total DRaFT optimizer updates for this invocation |

`draft.draft_k`, resolution, batch size, and DRaFT-LV samples have the largest memory impact. Begin with the face-only sample at 512px, batch size 1, `draft_k: 1`, and `lv_samples: 0`.

## Outputs and resume behavior

Updated adapters are written beneath `training_folder` in safetensors format. Reward images and captions are written to the job's `draft_step_images/` directory.

The configured `adapter.path` is always treated as the starting adapter, not as an SFT checkpoint whose step count should be continued. `train.steps: 30` therefore means 30 DRaFT updates. To continue a prior run, point `adapter.path` at the saved DRaFT `.safetensors` file and start a new invocation.

## Safety rails

Preflight rejects:

- normal `train`, `generate`, `extract`, `merge`, or extension jobs
- more than one process or any `datasets`/legacy `network` section
- any architecture other than `krea2`
- hub IDs or missing local model paths
- adapters other than local LoRA/LoKr safetensors
- body reward without local SAM 3D Body and SOMA-X assets

Downloaded weights, datasets, references, repositories, output adapters, caches, and local configs are ignored by Git. Do not force-add licensed or gated model files.

## Upstream

This fork retains selected trainer infrastructure from [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) and the Krea 2 DRaFT loop is based on [KONAKONA666/krea-2](https://github.com/KONAKONA666/krea-2). See `LICENSE` and upstream component licenses before redistribution.
