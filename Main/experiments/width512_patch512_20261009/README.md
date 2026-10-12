# Direct FC width512 / patch512 development trial

The requested model is **direct N→512**, two mesh Graph-CNN blocks of width512,
**512 fixed geometric context tokens**, two Transformer layers with four heads,
vertexwise VAE latent32, and complete FC reconstruction. There are no intermediate
512→256→128 projection stages. Patch count and feature width are separate settings.
The left-hemisphere model has 36,012,097 active parameters at N=29,696.

Outputs are isolated under `CoBRA_Experiments/direct_fc_width512_patch512_20261009_v1`.
Existing code, neural checkpoints and parcellation results are not overwritten.
This exploratory experiment explicitly reuses the existing **450-training-person
group FC**, column scales, mesh/LPE, and the corrected connected group label prior.
It is separate from the clean, from-raw `CoBRA_Rerun` submission-check workflow.

## Order

1. `s_step2_train_trial.py --phase test`: CPU dense/implicit value and gradient
   checks for one/two/four runs, plus a 512-token forward/backward check.
2. `--phase inspect`: actual model parameter count and input prerequisites.
3. `--phase smoke`: a real full-cortex group update and four-run personal update;
   saves gradient, peak-memory, and timing audits in `execution_check/`.
4. `--phase launch`: detach a resumable worker, which trains a new neural group
   prior for1,000 updates, then30 personal updates for each of the12 development
   identities, performs connected inference, and evaluates matched baseline maps.
5. Inspect `status.json`, `workflow.log`, `group/checkpoint.pt`, `individual/`,
   `summary.json`, `comparison_subjects.csv`, and `paired_contrasts.json`.

All preprocessing is **0.01–0.1 Hz**, applied independently to each train/evaluation
segment. Each personal fit uses Day1+Day2, LR+RL, first240 frames per run (960 total).
Evaluation uses separately filtered frames720:1200 from each run, pooling LR/RL
within day. These are heldout temporal segments of represented days, **not an
unseen-day or independent-subject test**. Retest is not used in this pilot.

Baseline is the saved width128/patch128 `joint_equal` arm on the same participants,
hemisphere, granularity, segments and connected label prior. Both width and patch
count change together, so this trial cannot identify their separate effects.
Parameter comparisons and actual peak memory are recorded, not inferred from FC
compression ratios. H/DCBC are evaluated on the opposite day's heldout signal.

Complete FC is calculated algebraically as `(Y.T @ Y/T) @ W.T = Y.T @ (Y @ W.T)/T`,
including column scaling. Reconstruction uses **every** output column and diagonal,
with exact ordered-distinct-run cross-reconstruction. Chunking and activation
checkpointing bound memory; they do not introduce anchors or sample FC entries.

GPU0 allocations are capped at7GiB; startup waits for at least10GiB free. Other
users' processes and the ongoing CoBRA validation workers are never interrupted.
