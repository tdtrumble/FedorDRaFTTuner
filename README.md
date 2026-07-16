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

The main requirements include `prodigyopt`, `timm`, and `open_clip_torch`.
Although the Python module is imported as `open_clip`, its package name in pip
is `open_clip_torch`. If upgrading an existing environment after pulling this
change, reinstall the requirements:

```bash
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

Run these commands from the repository root. Install the standalone Hugging Face
CLI first if `hf` is not already available:

```bash
python -m pip install --upgrade huggingface_hub
```

The trainer does not run any of the download commands below. If a repository is
gated, authenticate the CLI with `hf auth login` and accept its terms in a
browser before downloading.

#### 1. Krea 2 Raw transformer

- Source repository: `krea/Krea-2-Raw`
- Required repository filename: `raw.safetensors`
- Config key: `model.name_or_path`
- Exact local path used by both examples: `models/krea2/raw.safetensors`

```bash
hf download krea/Krea-2-Raw raw.safetensors --local-dir models/krea2
```

#### 2. Qwen3-VL-4B-Instruct text encoder

- Source repository: `Qwen/Qwen3-VL-4B-Instruct`
- Required content: the complete snapshot, including all weight shards,
  configuration files, tokenizer files, and processor files
- Typical root files include `config.json`, `generation_config.json`,
  `model.safetensors.index.json`, `model-*.safetensors`, tokenizer files, and
  preprocessor/processor configuration files. Do not download only one weight
  shard.
- Config key: `model.model_kwargs.text_encoder_path`
- Exact local directory used by both examples: `models/qwen3-vl-4b-instruct/`

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/qwen3-vl-4b-instruct
```

#### 3. Qwen-Image VAE

- Source repository: `Qwen/Qwen-Image`
- Required repository subtree: `vae/`, including its configuration and all VAE
  weight files
- Expected files include `vae/config.json` and the safetensors weight file or
  shards published beneath `vae/`
- Config key: `model.model_kwargs.vae_path`
- The config points to `models/qwen-image/`; the loader then reads its `vae/`
  subdirectory.
- Exact local directory: `models/qwen-image/vae/`

```bash
hf download Qwen/Qwen-Image --include "vae/*" --local-dir models/qwen-image
```

#### 4. AntelopeV2 face detector and recognizer

These are the two ONNX models used by the face-identity reward. Both are
required by both sample configs.

- Source repository: `immich-app/antelopev2`
- Required filenames: `detection/model.onnx` and `recognition/model.onnx`
- Config key: `draft.reward.face.model_dir`
- Exact local directory used by both examples: `models/antelopev2/`

```bash
hf download immich-app/antelopev2 --include "detection/model.onnx" --local-dir models/antelopev2
hf download immich-app/antelopev2 --include "recognition/model.onnx" --local-dir models/antelopev2
```

The resulting paths must be:

```text
models/antelopev2/detection/model.onnx
models/antelopev2/recognition/model.onnx
```

#### 5. Your existing Krea 2 adapter

This repository cannot download or create the starting adapter. Copy an
existing Krea 2 LoRA or LoKr `.safetensors` file into `models/adapters/`, then
set `adapter.path` to its exact filename. Its type and construction settings in
the YAML must match the file.

Windows PowerShell example:

```powershell
New-Item -ItemType Directory -Force models/adapters | Out-Null
Copy-Item "C:\path\to\subject_lora.safetensors" "models/adapters/subject_lora.safetensors"
```

Linux example:

```bash
mkdir -p models/adapters
cp /path/to/subject_lora.safetensors models/adapters/subject_lora.safetensors
```

Use `models/adapters/subject_lora.safetensors` with the LoRA example or
`models/adapters/subject_lokr.safetensors` with the LoKr example.

#### Verify the required files

Windows PowerShell:

```powershell
Test-Path models/krea2/raw.safetensors
Test-Path models/qwen3-vl-4b-instruct/config.json
Test-Path models/qwen-image/vae/config.json
Test-Path models/antelopev2/detection/model.onnx
Test-Path models/antelopev2/recognition/model.onnx
Test-Path models/adapters/subject_lora.safetensors
```

Linux:

```bash
test -f models/krea2/raw.safetensors
test -f models/qwen3-vl-4b-instruct/config.json
test -f models/qwen-image/vae/config.json
test -f models/antelopev2/detection/model.onnx
test -f models/antelopev2/recognition/model.onnx
test -f models/adapters/subject_lora.safetensors
```

Expected required layout:

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

The LoKr body-match example additionally references SAM 3D Body and SOMA-X.
They are not needed by the face-only LoRA example.

#### 6. SAM 3D Body ViT-H checkpoint and source code

- Gated model repository: `facebook/sam-3d-body-vith`
- Required model files: `model.ckpt` and `assets/mhr_model.pt`
- Required source repository: `https://github.com/facebookresearch/sam-3d-body`
- Config key: `draft.reward.body.sam3d_checkpoint_path`
- Exact checkpoint path used by the LoKr example:
  `models/sam-3d-body-vith/model.ckpt`
- Exact source destination expected by the reward implementation:
  `repositories/sam-3d-body/`

Request access to the gated model repository and accept its license before
running:

```bash
hf download facebook/sam-3d-body-vith --local-dir models/sam-3d-body-vith
git clone https://github.com/facebookresearch/sam-3d-body repositories/sam-3d-body
```

SAM 3D Body has additional Python dependencies in the cloned repository's
`INSTALL.md`; install them into this project's active environment.

#### 7. SOMA-X body model assets

- Source repository: `https://github.com/NVlabs/SOMA-X`
- Required content: the complete Git LFS checkout, especially the model data
  beneath `assets/`
- Config key: `draft.reward.body.soma_data_root`
- Exact local directory used by the LoKr example: `repositories/SOMA-X/assets/`

```bash
git lfs install
git clone https://github.com/NVlabs/SOMA-X repositories/SOMA-X
git -C repositories/SOMA-X lfs pull
```

Verify the body-reward dependencies:

```powershell
# Windows PowerShell
Test-Path models/sam-3d-body-vith/model.ckpt
Test-Path models/sam-3d-body-vith/assets/mhr_model.pt
Test-Path repositories/sam-3d-body/sam_3d_body
Test-Path repositories/SOMA-X/assets
```

```bash
# Linux
test -f models/sam-3d-body-vith/model.ckpt
test -f models/sam-3d-body-vith/assets/mhr_model.pt
test -d repositories/sam-3d-body/sam_3d_body
test -d repositories/SOMA-X/assets
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

Training prompts may be listed directly under `draft.prompts`, or one per line
in `draft.prompts_path`. `[trigger]` is replaced by `trigger_word`. The tracked
`config/examples/prompts/body_match.txt` pool emphasizes head-to-toe visibility,
readable body proportions, several camera angles and poses, and a visible face
so it works with the combined body and face reward.

Training prompts are not all generated for every optimizer update. They are
shuffled deterministically once, then selected in round-robin order: one prompt
per batch item. With 21 prompts and `train.batch_size: 1`, a full traversal takes
21 optimizer updates; with batch size 2, each update consumes two prompt slots.
Each selected prompt generates an on-policy reward image, and that image is used
immediately to update the adapter.

`train.gradient_accumulation` creates real DRaFT microbatches. Each microbatch
is generated, rewarded, backpropagated, and released before the next one; the
optimizer, LR scheduler, and EMA advance only after all microbatches finish.
The effective prompt batch is:

```text
train.batch_size × train.gradient_accumulation
```

For example, `batch_size: 1` with `gradient_accumulation: 2` approximates the
gradient average of batch size 2 while keeping peak image/reward graph memory
near batch size 1. It takes roughly twice the generation/reward time per
optimizer update and may not be bit-identical to a true batch because random
samples are generated sequentially. `train.steps` continues to count optimizer
updates, not microbatches. The legacy `gradient_accumulation_steps` key is not
supported by this fork.

```text
[trigger] portrait photo, natural light
[trigger] full body photo, standing outdoors
```

Checkpoint sampling uses a separate one-prompt-per-line file configured with
`draft.sample_prompts_path`. Keep this list short and fixed. At each
`draft.sample_every` interval, every sampling prompt is generated sequentially
with the newly updated adapter and a stable per-prompt seed. This makes the same
prompt/seed directly comparable between checkpoints without increasing peak
VRAM by batching the entire validation set. The tracked
`config/examples/prompts/body_match_sample.txt` provides six body-match views.
Checkpoint samples use `draft.sample_width` and `draft.sample_height`; each
defaults to the corresponding training reward dimension when omitted. They
reuse `draft.steps`, `draft.guidance_scale`, and the same Krea 2 schedule.

## Configure and run

Copy one of the samples:

- `config/examples/krea2_lora_draft.yaml`: LoRA, face-only, smallest setup
- `config/examples/krea2_lokr_draft.yaml`: LoKr, face plus body/SOMA-X

Edit at least these values:

- `adapter.path`, `adapter.type`, `adapter.rank`, and `adapter.alpha`; for LoKr also match `factor`, `full_rank`, and `old_format`
- all paths under `model`
- `draft.reward.reference_images` and the face/body model paths
- `draft.prompts` or `draft.prompts_path`
- `draft.sample_prompts_path` when `draft.sample_every` is positive
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
| `draft.sample_every` | Generate the complete fixed sampling-prompt set after every N optimizer updates; `0` disables checkpoint sampling. The final updated adapter is always sampled when enabled. |
| `draft.sample_width`, `draft.sample_height` | Checkpoint-sample dimensions, each a positive multiple of 16. They default to `draft.width` and `draft.height`, but may be larger or use a different aspect ratio. |
| `draft.sample_prompts_path` | Separate local text file containing one fixed checkpoint-evaluation prompt per line |
| `draft.sample_seed` | Base seed for checkpoint samples. Prompt index is added, so every prompt keeps the same seed across checkpoints. |
| `train.batch_size` | Number of reward images generated together in each microbatch; higher values increase peak VRAM |
| `train.gradient_accumulation` | Number of sequential DRaFT microbatches averaged before one optimizer update; increases effective batch with little additional peak VRAM |
| `train.steps` | Total DRaFT optimizer updates for this invocation |

If checkpoint samples will be used to choose an intermediate adapter, make
`draft.sample_every` equal to `draft.save_every` or a multiple of it. Otherwise
the sampler may show an intermediate weight state for which no safetensors
checkpoint was retained. The final adapter is always saved and, when checkpoint
sampling is enabled, always sampled.

Optimizer examples:

```yaml
# Conservative baseline
train:
  optimizer: adamw
  lr: 0.0001
  optimizer_params:
    weight_decay: 0.0001
```

```yaml
# Prodigy uses its standard self-adjusting scale with lr: 1.0
train:
  optimizer: prodigyopt
  lr: 1.0
  optimizer_params:
    weight_decay: 0.0001
```

`prodigyopt` is installed by `requirements.txt`. Do not reuse Prodigy's
`lr: 1.0` if you switch back to AdamW. For a new reward configuration, establish
an AdamW baseline before comparing Prodigy because reward-gradient scale and
noise differ from ordinary supervised tuning.

Reward settings used by the examples:

| Key | Meaning |
|---|---|
| `draft.reward.reference_images` | File or directory of subject reference images used to build the face/body target |
| `draft.reward.face.model_dir` | Local AntelopeV2 root containing `detection/model.onnx` and `recognition/model.onnx` |
| `draft.reward.face.target_similarity` | ArcFace cosine-similarity saturation threshold, not a required final score. The example value `0.45` keeps useful identity pressure below the threshold and rapidly tapers it above the threshold to reduce reference-copy overfitting. It is a starting point and should be validated on your references. |
| `draft.reward.body.sam3d_checkpoint_path` | Exact local SAM 3D Body checkpoint file. The body example uses `models/sam-3d-body-vith/model.ckpt`; its sibling `assets/mhr_model.pt` must also exist. |
| `draft.reward.body.soma_data_root` | Local SOMA-X model-data directory. The body example uses `repositories/SOMA-X/assets`; it must be a complete Git LFS checkout. |

For `target_similarity: 0.45`, the face reward is approximately zero once
similarity is comfortably above `0.45`; it does not penalize a score for being
higher. Raising the value demands a closer identity match but increases the
risk of overfitting to reference-specific expression, pose, lighting, or image
artifacts. Lowering it makes the face constraint weaker. Compare saved samples
and raw identity metrics before changing it rather than treating the threshold
as a universal ArcFace acceptance cutoff.

`draft.draft_k`, resolution, batch size, and DRaFT-LV samples have the largest memory impact. Begin with the face-only sample at 512px, batch size 1, `draft_k: 1`, and `lv_samples: 0`.

## Outputs and resume behavior

Updated adapters are written beneath `training_folder` in safetensors format.
On-policy images used to calculate training rewards are written to the job's
`draft_step_images/` directory when `draft.save_images_every` is enabled; prompt
sidecar files are not created.

Fixed checkpoint-comparison images are written beneath:

```text
<training_folder>/<config.name>/checkpoint_samples/
```

Each filename records the exact post-update weights, prompt index, and fixed
seed, for example:

```text
weights_000010_prompt_003_seed_1003.jpg
```

All `prompt_003` images use the fourth active prompt in
`draft.sample_prompts_path` and seed 1003,
so they can be compared directly across sampled checkpoints. Sampling displays
a per-prompt progress bar while the main training progress bar is paused.
Starting another invocation with the same `training_folder` and `config.name`
will overwrite sample files with matching weight-step, prompt-index, and seed
names; change `config.name` when prior runs must be preserved.

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

This fork retains selected trainer infrastructure from [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) and the Krea 2 DRaFT loop is based on https://github.com/CliffNodes/FedorAiToolkit / [KONAKONA666/krea-2](https://github.com/KONAKONA666/krea-2). See `LICENSE` and upstream component licenses before redistribution.
