import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { isMac } from '@/helpers/basic';

const prisma = new PrismaClient();

const clone = (obj: any) => JSON.parse(JSON.stringify(obj));

const DRAFT_TRAINER_TYPES = new Set(['krea2_draft_trainer', 'ideogram4_draft_trainer']);

const draftStageDefaults = (type: string) => {
  if (type === 'ideogram4_draft_trainer') {
    return {
      steps: 12,
      draft_k: 1,
      guidance_scale: 7.0,
      width: 512,
      height: 512,
      lv_samples: 0,
      seed: 42,
      checkpoint_vae: true,
      train_modules: 'qkvo',
      save_images_every: 10,
      save_every: 10,
    };
  }
  return {
    steps: 8,
    draft_k: 1,
    guidance_scale: 4.5,
    width: 512,
    height: 512,
    lv_samples: 0,
    high_noise_shift: 0.5,
    seed: 42,
    checkpoint_vae: true,
    train_modules: 'qkvo',
    save_images_every: 0,
    save_every: 15,
  };
};

/**
 * Multi-stage jobs: the DRaFT reward stage (process[1+]) is derived from the
 * SFT stage (process[0]). The form only edits reward-specific settings, so on
 * every save we re-sync the shared fields (model, network, folders, trigger)
 * from process[0] and keep step counts cumulative -- the DRaFT stage resumes
 * the SFT checkpoint from the same training folder.
 */
const syncDraftStages = (jobConfig: any) => {
  const processes = jobConfig?.config?.process;
  if (!Array.isArray(processes) || processes.length < 2) return;
  const p0 = processes[0];
  for (let i = 1; i < processes.length; i++) {
    const p = processes[i];
    if (!DRAFT_TRAINER_TYPES.has(p?.type)) continue;

    p.training_folder = p0.training_folder;
    p.sqlite_db_path = p0.sqlite_db_path;
    p.device = p0.device;
    p.trigger_word = p0.trigger_word;
    p.performance_log_every = p0.performance_log_every;
    p.model = clone(p0.model);
    p.logging = clone(p0.logging);
    p.sample = clone(p0.sample);
    // resume happens via the shared training folder; no pretrained path
    p.network = clone(p0.network);
    delete p.network.pretrained_lora_path;

    const draft = p.draft || {};
    const reward = draft.reward || {};
    const rewardSteps = Number(draft.num_reward_steps ?? 12);
    const sftSteps = Number(p0.train?.steps ?? 0);
    const draftSaveEvery = 15;
    p.save = {
      ...clone(p0.save),
      save_every: 0,
      max_step_saves_to_keep: Math.max(
        p0.save?.max_step_saves_to_keep ?? 5,
        Math.ceil(rewardSteps / draftSaveEvery) + 2,
      ),
    };
    // the DRaFT stage samples its own images from prompts; no datasets
    delete p.datasets;

    p.draft = {
      ...draftStageDefaults(p.type),
      save_every: draftSaveEvery,
      save_after_step: sftSteps,
      final_sample_width: 512,
      final_sample_height: 512,
      final_sample_steps: 16,
      final_sample_count: 1,
      prompts: null,
      prompts_path: null,
      ...draft,
      num_reward_steps: rewardSteps,
      save_every: draftSaveEvery,
      save_after_step: draft.save_after_step ?? sftSteps,
      reward: {
        reference_images: p0.datasets?.[0]?.folder_path ?? null,
        face_weight: 1.0,
        body_weight: 0.5,
        // CUDA first for speed; CPU fallback if ORT CUDA is unavailable. Face
        // detection is non-differentiable — body reward still uses somax tier.
        face: {
          target_similarity: 0.45,
          providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        },
        body: { loss_tier: 'somax', sam3d_hf_repo: 'facebook/sam-3d-body-vith' },
        ...reward,
      },
    };

    // steps are cumulative: the stage resumes at p0's final step
    p.train = {
      ...clone(p0.train),
      steps: Number(p0.train?.steps ?? 0) + rewardSteps,
      optimizer: 'adamw',
      optimizer_params: { weight_decay: 1e-4 },
      lr: 0.0001,
      max_grad_norm: 1.0,
      skip_first_sample: true,
      cache_text_embeddings: false,
      unload_text_encoder: false,
    };
  }
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const only_active = searchParams.get('only_active');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return NextResponse.json(job);
    }

    const where: any = {};
    if (job_type) {
      where.job_type = job_type;
    }
    if (only_active === 'true') {
      where.status = { in: ['running', 'queued', 'stopping'] };
    }

    const jobs = await prisma.job.findMany({
      where,
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config } = body;
    let gpu_ids: string = body.gpu_ids;

    syncDraftStages(job_config);

    if (isMac()) {
      gpu_ids = "mps";
    }

    const extra: any = {};
    if ("job_ref" in body) {
      extra["job_ref"] = body.job_ref;
    }

    if ("job_type" in body) {
      extra["job_type"] = body.job_type;
    }

    if (id) {
      // Update existing training
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          ...extra,
        },
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      // Create new training
      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
          ...extra,
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
