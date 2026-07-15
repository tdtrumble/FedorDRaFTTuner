# Changelog

## 2.0.0 - DRaFT-only fork

- Merged upstream `ostris/ai-toolkit` through `3e6bd87`.
- Removed the web UI, UI scripts/assets, Docker files, Modal runner, notebooks, generic examples, and non-Krea built-in model implementations.
- Restricted the CLI to `job: krea2_draft` with exactly one `krea2_draft` process.
- Replaced the former SFT-plus-DRaFT config with a required existing local LoRA/LoKr safetensors adapter.
- Made `train.steps` the exact number of DRaFT updates and disabled implicit adapter/optimizer resume from the output directory.
- Added local path preflight and `python run.py <config> --validate`.
- Forced Hugging Face/Transformers/Diffusers offline mode and removed direct Hub/download helpers.
- Restricted model registration to Krea 2 and made all Krea 2 component loads local-only.
- Added LoRA face-only and LoKr face-plus-body sample configs.
- Added manual model layout/setup documentation and Git ignores for local weights, repositories, references, and outputs.
