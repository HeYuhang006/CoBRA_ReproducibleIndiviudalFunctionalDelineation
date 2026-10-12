# Experimental source snapshots and scope

The numbered method entrypoints and `README_运行顺序.md` define the clean manual main rerun. Experimental folders below preserve the actual current research implementations; their historic path dependencies are explicit and must not be confused with a verified portable end-to-end rerun.

## Group NMF

`TraditionalMethod/03_population/f_group_nmf.py` extracts the exact production NMF numerical body with function-prefix renaming and a new-work-root path. Source snapshot: `docs/provenance/production_baseline_sources/paper_figure_campaign/group_subject_boxplot_experiment.py.txt`. Compare the numerical body directly when auditing.

The positive part of the500-person signed mean FC (diagonal zero) is factorized at rank32. Initialization is NNDSVDa with a randomized range and two power iterations; multiplicative Frobenius updates use at most1000 iterations, checking every10 and the original1e-4 relative-improvement stopping rule. K-means uses50/100/150/200 left parcels,10 starts,300 iterations, seed20261008. NMF uses no CoBRA embedding. The fresh entry is `TraditionalMethod/s_step2_train_population.py --mode group_nmf`, after DRFC data preparation. It only fits NMF; expanded production evaluation is separate.

## RNCut (surface adaptation) and production evaluation

`TraditionalMethod/experiments/group_nmf_rncut_20261009` contains the production preparation, fitting, evaluation, bundle and K300 initialization-repair scripts. Their W/output constants still refer to the production `GPT_workplace` installation. They depend on its original group benchmark helpers and external author optimizer; the helper source snapshots are retained in `docs/provenance/production_baseline_sources`. They are **not** automatically run by Main or Classical step2.

Inspect the experiment README and script arguments before execution. For the original production directory, the intended order is preparation/fitting (`s_step1_prepare_and_fit_rncut.py`), matched evaluation (`s_step3_evaluate_group.py`), then bundle (`s_step5_bundle.py`). Recovery scripts are only for an inspected failed initialization; do not start them alongside an active optimizer. To rerun this experiment in a new root, explicitly port and audit its production helper paths first; this release does not claim that work has been validated.

RNCut uses500-person `mean(abs(individual FC))`, the original optimizer, and a surface neighborhood adapter checked against the native interface. Frozen author files are preserved externally. The latest initialization-guard sources are included, without custom final-label repair. Nominal100/200/300/400 denote bilateral counts; production displays/evaluation currently use the left hemisphere. Actual native/evaluation parcel counts must be disclosed where parcels disappear under the common mask. Optimizer completion, output hashes, topology checks and50-person metrics are separate gates. At the source snapshot on2026-10-12, K300 initialization had been repaired while final optimization was still running; the source upload does not imply its completion.

Matched production evaluation preserves0.01–0.1Hz filtering, the same evaluation mask and metric definitions, and the60-comparison Holm family. An unfinished fit is not an available result. This repo contains source and provenance, no measured statistics or atlas outputs.

## Width512 / patch512 development trial

`Main/experiments/width512_patch512_20261009` contains the direct full-FC-to512 projection, two512-wide Graph-CNN layers,512 context tokens, two4-head Transformer layers and32-dimensional latent. The left model with29696 cortical vertices has36,012,097 trainable parameters. Width and token count change together.

This is a12-person development experiment using existing group-FC references, duration-matched cross-day training segments, and separately filtered held-out time segments. It does not establish independent long-term Retest gains. Its production paths, GPU memory guard and evidence limits remain in the original scripts. It is not activated by the canonical Main workflow and does not replace the frozen final model.

## Author and diagnostic baselines

DRFC is a500-person group reference; CoBRA uses450 training identities. GPIP jointly estimates subject inverse covariance per visit and uses the signed Ncut author optimizer, with explicit MATLAB compatibility adapters. The20-iteration budget is not proof of outer-loop label stability. Wang-style18 uses a later HFR author release; Wang-style100 is diagnostic. The18-network setting permits distributed anatomical components; do not treat its component count as100-parcel fragmentation error.
