# FAQ

## Why does validation say a model path is missing?

This runner is intentionally offline. Replace every repository ID with a local path and download the files listed in `README.md` yourself. Run the same command again with `--validate` before starting training.

## Can this train a new LoRA or LoKr from images?

No. Train a Krea 2 adapter elsewhere, then set `adapter.path` to its `.safetensors` file. Reference images here define the differentiable reward; they are not an SFT dataset.

## How many reward steps run?

Exactly `train.steps`. The old two-stage convention and `draft.num_reward_steps` are gone.

## Why does my adapter fail to load?

`adapter.type`, rank/alpha, and the LoKr factor/full-rank/old-format settings must match the file that created it. The file must target Krea 2's transformer.

## Can body setup fail while face training continues?

Runtime body initialization failures degrade to face-only with a warning, but preflight requires the configured local SAM 3D Body and SOMA-X asset paths whenever `body_weight` is non-zero. Set `body_weight: 0.0` for an intentional face-only run.

## Where are results saved?

Under `training_folder` (default `output/`). Intermediate adapters follow `draft.save_every`; generated reward images follow `draft.save_images_every`.

## Is there a UI or Docker image?

No. Both were removed; `python run.py <config>` is the only supported interface.
