# Changelog

All notable changes to [FedorAiToolkit](https://github.com/CliffNodes/FedorAiToolkit) are documented here.

## [Unreleased]

### Changed — DRaFT speed defaults (face + body, quality preserved)

DRaFT reward training defaults are tuned for faster iteration on 24–32 GB GPUs without switching to face-only mode or the weaker `shape_params` body loss tier. Body reward still uses **`body_weight: 0.5`** and **`loss_tier: somax`**.

| Setting | Before | After |
|---------|--------|-------|
| `draft.steps` | 12 | **8** |
| `draft.num_reward_steps` (UI default) | 60 | **10** |
| Total steps (200 SFT + DRaFT) | 260 | **210** |
| `draft.save_every` (after SFT) | 10 | **5** |
| `draft.save_images_every` | 10 | **0** |
| Face ONNX providers | CPU only | **CUDA → CPU fallback** |
| DRaFT stage previews | 1024×30×3 | **512×16×1** (end of run only) |

Checkpoints during DRaFT now land at steps **205** and **210** (with the default 200-step SFT stage).

**Not changed** (intentionally, to preserve body geometry accuracy):

- `draft.reward.body_weight` remains **0.5**
- `draft.reward.body.loss_tier` remains **somax** (not `shape_params`)
- Face-only / 5060 presets unchanged for low-VRAM users

### Files

- `config/examples/krea2_lokr_draft.yaml`
- `config/examples/krea2_lora_draft.yaml`
- `ui/src/app/api/jobs/route.ts` — DRaFT stage sync defaults
- `ui/src/app/jobs/new/SimpleJob.tsx` — new-job form defaults

---

## Prior releases

See git history for earlier work: Krea 2 DRaFT-K training, SAM 3D Body body reward, UI Quick start, elapsed-time column, and related fixes.
