Historical production snapshot: this K300 queue and figure delivery completed on 12 October 2026. The steps below record that completed workflow; do not relaunch the queue. Its original data bundles, workstation paths and external rendering runtime are required. Use the repository's numbered entrypoints for a fresh rerun.

# Complete the accepted Group Figure with K300

This continuation preserves the already accepted native K400 result: 195 left
parcels before masking, 192 on the shared evaluation mask, versus 200 requested.
The original exact realized-count campaign is therefore not marked complete.

## Execution order

1. Leave the healthy MATLAB fit in
   `runs/group_nmf_rncut_20261009_v2/RNCut/initialization_guard_20261012_v2`
   running. It uses the original author optimizer and the recorded fixed-seed,
   training-only rounding fallback. Do not start another fit or repair labels.
2. Run `s_step6_finish_k300.py --check`, then `--launch` once on server `142`.
   This finite queue waits up to 12 hours for the existing launcher's validated
   150-parcel candidate. It does not train a model or optimize a new atlas.
3. The queue validates original hashes and the previously accepted 34-map
   bundle, evaluates K300 on the same 50 people, and recomputes all 60 planned
   paired Wilcoxon tests with exact Holm adjustment. The numerical evaluator is
   unchanged; only the evaluated resolution list is restricted to the one
   missing map. Shared mask and 0.01–0.1 Hz preprocessing remain unchanged.
4. Inspect `postfit_status.json`, the new bundle manifest, candidate receipt,
   all 35 maps / 5,250 measurements, and the 60 actual comparisons. Record
   native and evaluation-mask parcel counts; do not infer them from filenames.
5. Create a new version of the accepted Figure builder. The previous partial
   builder has hardcoded 19-map / 57-test gates and missing-K300 annotations;
   it must not be used unchanged for the completed 20-map / 60-test figure.
6. Edit only the Group Figure slide (slide4) in the user's latest seven-slide
   final deck, snapshotted before editing. Preserve the other six slides and all
   existing non-RNCut brain positions. The separate older Group reference slide
   is user content and is not removed by this focused update.
   Use native editable objects, the established Individual palette, compact
   unconnected statistics, and explicit actual-count disclosures.
7. Reimport the actual saved PPTX, inspect its previews, and pass the Nature
   and PPT integrity checks before delivering the new deck and source tables.

## Server paths

- Script: `/dataspace/yuhang_data/IndividualFunctionalParcellation/CoBRA/TraditionalMethod/experiments/group_nmf_rncut_20261009/s_step6_finish_k300.py`
- Prior accepted results: `/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace/runs/group_accepted_revision_20261012_v1`
- New results: `/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace/runs/group_accepted_revision_20261012_v2_k300`
- Queue evidence: `postfit_launch.json`, `postfit_status.json`, `postfit.log`
- Scientific bundle: `figure_data.tar.gz`, `figure_data/manifest.json`,
  `figure_data/complete.json`, `figure_data/summary.json`

`ready_for_figure_authoring` means the measured comparison bundle is ready.
It does not mean that an editable PPT has passed visual review or been delivered.

The old 20261009 finite queue and strict post-fit queue are superseded; do not
restart them. The hourly `cobra` automation was paused before this continuation;
this finite dependent queue does not reactivate that automation.

## Figure authoring order

After downloading and hash-verifying the new scientific archive, extract its
`figure_data` directory to `build/evidence`. Do not run with the partial bundle.

1. `s_plot_revision.py` checks all source and statistical gates, writes the
   3,000 displayed source observations,60 summary rows,20 topology rows,
   all60 tests and48 displayed comparisons, then exports the Python scene.
2. `s_step2_native.mjs` exports the scene as native editable objects.
3. `s_step3_splice.py` replaces the old generated Group content on slide4 only,
   keeps the16 supplied brain groups and creates a bounded edited-slide package.
4. `s_step4_finalize.mjs` runs the normal installed Presentations finalizer.
5. `s_step5_assemble.py` inserts that checked slide into a new seven-slide copy.
6. Reimport/render the complete output and standalone Group slide. Inspect them
   before delivery; source/PDF/package checks alone do not establish visual QA.

Use the bundled Python/Node runtimes and the already installed scientific
vendor path from `reproduction/nature_figures_20261007/build/vendor`.
The supplied source SHA256 at this continuation is
`9249e9192e847967bd6d7f421f51096d9b7ba3d396b26a40cec7d9c774a7a94c`.
