# Experimental source snapshots and scope

The numbered method entrypoints and `README_运行顺序.md` define the clean manual main rerun. Experimental folders below preserve the actual current research implementations; their historic path dependencies are explicit and must not be confused with a verified portable end-to-end rerun.

## Group NMF

`TraditionalMethod/03_population/f_group_nmf.py` extracts the exact production NMF numerical body with function-prefix renaming and a new-work-root path. Source snapshot: `docs/provenance/production_baseline_sources/paper_figure_campaign/group_subject_boxplot_experiment.py.txt`. Compare the numerical body directly when auditing.

The positive part of the500-person signed mean FC (diagonal zero) is factorized at rank32. Initialization is NNDSVDa with a randomized range and two power iterations; multiplicative Frobenius updates use at most1000 iterations, checking every10 and the original1e-4 relative-improvement stopping rule. K-means uses50/100/150/200 left parcels,10 starts,300 iterations, seed20261008. NMF uses no CoBRA embedding. The fresh entry is `TraditionalMethod/s_step2_train_population.py --mode group_nmf`, after DRFC data preparation. It only fits NMF; expanded production evaluation is separate.

## RNCut (surface adaptation) and production evaluation

`TraditionalMethod/experiments/group_nmf_rncut_20261009` contains the production preparation, fitting, evaluation, bundle and K300 initialization-repair scripts. Their W/output constants still refer to the production `GPT_workplace` installation. They depend on its original group benchmark helpers and external author optimizer; the helper source snapshots are retained in `docs/provenance/production_baseline_sources`. They are **not** automatically run by Main or Classical step2.

Inspect the experiment README and script arguments before execution. For the original production directory, the intended order is preparation/fitting (`s_step1_prepare_and_fit_rncut.py`), matched evaluation (`s_step3_evaluate_group.py`), then bundle (`s_step5_bundle.py`). Recovery scripts are only for an inspected failed initialization; do not start them alongside an active optimizer. To rerun this experiment in a new root, explicitly port and audit its production helper paths first; this release does not claim that work has been validated.

RNCut uses500-person `mean(abs(individual FC))`, the original optimizer, and a surface neighborhood adapter checked against the native interface. Frozen author files are preserved externally. The latest initialization-guard sources are included, without custom final-label repair. Nominal100/200/300/400 denote bilateral counts; production displays/evaluation currently use the left hemisphere. Actual native/evaluation parcel counts must be disclosed where parcels disappear under the common mask. Optimizer completion, output hashes, topology checks and50-person metrics are separate gates. The original cc1d958 source snapshot preceded K300 completion. The later accepted Group comparison completed K300 with150 native left parcels and148 on the common evaluation mask; the final measured package and editable-figure delivery are verified by the production figure_delivery_complete.json. Accepted K400 retains195 native/192 evaluated left parcels. This completes the user-accepted Group comparison while preserving the realized-count mismatch.

Matched production evaluation preserves0.01–0.1Hz filtering, the same evaluation mask and metric definitions, and the60-comparison Holm family. An unfinished fit is not an available result. This repo contains source and provenance, no measured statistics or atlas outputs.

## Width512 / patch512 development trial

`Main/experiments/width512_patch512_20261009` contains the direct full-FC-to512 projection, two512-wide Graph-CNN layers,512 context tokens, two4-head Transformer layers and32-dimensional latent. The left model with29696 cortical vertices has36,012,097 trainable parameters. Width and token count change together.

This is a12-person development experiment using existing group-FC references, duration-matched cross-day training segments, and separately filtered held-out time segments. It does not establish independent long-term Retest gains. Its production paths, GPU memory guard and evidence limits remain in the original scripts. It is not activated by the canonical Main workflow and does not replace the frozen final model.

## Author and diagnostic baselines

DRFC is a500-person group reference; CoBRA uses450 training identities. GPIP jointly estimates subject inverse covariance per visit and uses the signed Ncut author optimizer, with explicit MATLAB compatibility adapters. The20-iteration budget is not proof of outer-loop label stability. Wang-style18 uses a later HFR author release; Wang-style100 is diagnostic. The18-network setting permits distributed anatomical components; do not treat its component count as100-parcel fragmentation error.

## Completed K300 queue and final figure source snapshot

`TraditionalMethod/experiments/group_complete_revision_20261012` preserves the finite K300 evaluation queue and the final 20-map/60-test figure builders, including the saved-PPT reimport review. These are production snapshots. The Python queue imports helpers from the original `group_nmf_rncut_20261009` installation and reads the preceding production bundle. The figure builders use the original Windows experiment build/evidence directory and the installed Artifact Tool runtime; copying this folder into a code checkout alone does not provide those inputs. No data bundle, atlas label map, statistic, model checkpoint or author optimizer is included. Do not launch this completed historical queue again. The canonical fresh-rerun entrypoints remain separate.

## Matched author-GPIP application provenance

The production `runs/fig2_applications_20261010_v1` and `runs/fig2_group_applications_20261010_v1` completed task and connectome applications using the matched36-person,29696-vertex left-cortex50-parcel setting (nominal bilateralK100). GPIP uses supplied author core with the external compatibility adapter at20 fixed outer iterations; Wang100 is adapted. The Group extension uses the same connected group prior and measurement support. These completed production extensions are distinct from the original bilateralK200 Group/CoBRA application and from the portable numbered stage6 API, whose GPIP/Wang guards remain in place. They do not establish completed author-GPIP K18 or bilateral all-scale applications.
