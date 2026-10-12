# Group-level NMF and RNcut comparison

This extension was explicitly requested on 9 October 2026. It preserves all
existing methods, original author sources, output arrays, and ongoing jobs.
It supersedes the previous decision to leave RNcut unavailable: two spatial
interfaces can now be adapted while retaining the author optimizer, with native
voxel equivalence demonstrated before real-data processing.

## Scientific protocol

- Same frozen 500 training identities/REST1_LR inputs as the existing DRFC/NMF
  baseline. No overlap with the same 50 evaluation identities.
- Same MSMAll cortical vertex order and 0.01–0.1 Hz band-pass. Existing filtered
  caches are checked against source data/filter samples by the existing loader.
- Display and evaluate the left hemisphere at 50/100/150/200 parcels, described
  as nominal bilateral K100/200/300/400. No bilateral coupling is claimed.
- NMF: reuse the completed, source-hash-verified rank32 Frobenius factorization
  of positive training-group FC, with diagonal zero, NNDSVDa initialization,
  multiplicative updates (720 iterations, declared stopping rule), and K-means.
  No VAE embeddings enter this baseline. It is not TReND regularized NMF.
- RNcut: compute the mean of **individual absolute Pearson correlations**,
  diagonal zero, as in the author similarity script. It is never replaced with
  the absolute value of a signed group average, feature-kNN, or mesh-only Ncut.
- RNcut retains unchanged author RNcut P-code, Ncut initialization, KBound,
  NcutEnergy, pair weights, component label cost, and alpha expansion. The
  existing compiler/memory-ownership compatibility MEX is reused.
- Only `getNeighbours` and `bwlabeln` are replaced with mesh one-ring adjacency
  and induced-graph components. A padded volume is an index container; its
  geometry is never used. Component IDs can differ by a permutation; component
  partitions/counts agree. Whole native RNcut labels agree **exactly** in two
  voxel tests, with 60 matching random component-partition/count tests.
- Label RNcut as **RNCut (surface adaptation)**, not the original volumetric
  experiment. This verifies algorithmic compatibility, not identical scientific
  behavior of a mesh and a voxel lattice. No CoBRA connected refinement,
  empty-parcel repair, reseeding, or evaluation-based tuning is added.
- Author outputs with fewer occupied labels are preserved and reported; the
  matched-resolution evaluator stops instead of silently repairing them.
- All new metrics use the exact same 29,251-vertex support, complete-FC profile
  definition and training-geodesic DCBC pairs as the existing 27-map benchmark.
  H, silhouette and DCBC are checked against the unchanged DRFC100 measurements.
- Combined comparison: 35 maps, 50 people, 3 metrics = 5,250 observations.
  CoBRA-versus-Schaefer/DRFC/raw/NMF/RNcut paired Wilcoxon comparisons at matching
  K are Holm-corrected together over 60 tests. Different-K fixed atlases are
  descriptive only. DRFC/NMF/RNcut train on500; CoBRA variants on450.

## Locations and run order

Server code:
`/dataspace/yuhang_data/IndividualFunctionalParcellation/CoBRA/TraditionalMethod/experiments/group_nmf_rncut_20261009`

Server results:
`/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace/runs/group_nmf_rncut_20261009_v2`

1. `f_probe_rncut_interface.m`: one-time native interface/profile diagnostic.
2. `f_rncut_surface_driver('validate', .../RNCut)`: native equivalence gate.
3. `s_step3_evaluate_group.py --family NMF`: completed NMF maps, new common-mask
   metrics. Runs independently while RNcut inputs are prepared.
4. `s_step1_prepare_and_fit_rncut.py`: 500-person absolute-FC preparation,
   checkpoint every25 subjects, four author-RNcut fits, then automatically calls
   `s_step3_evaluate_group.py --family RNCut` and `--family merge`.
5. `s_step5_bundle.py`: only packages a complete 35-map/50-person/60-test result.
6. Local `s_step7_finish_figures.ps1 -WaitForResults`: finite, one-shot dependent
   figure queue (24-hour bound). Download/hash verification; Python compact
   comparison and multiresolution matrix; Nature source/font/collision/alignment
   gates; native two-slide PPT export/reimport and editability audit. No recurring
   automation is created or reactivated. Inspect both exported previews before
   presenting the PPT as visually approved.

Never launch a second compute worker while its lock is held. Source hashes and
input file identities are frozen. Do not modify fitting code while it runs.

## Evidence and outputs

- `RNCut/native_equivalence.json`: the passing native-domain checks.
- `RNCut/protocol.json`, `affinity_checkpoint.json`, `affinity_complete.json`:
  frozen scientific protocol, cache/source checks and complete affinity hash.
- `RNCut/preparation_status.json`, `status.json`, `matlab_fit.log`: live progress.
- `NMF/status.json`, `NMF/participants`, `NMF/summary.json`: common-mask evaluation.
- `figure_data/complete.json`: only written after both methods' measurements.
- Local `build/finish_status.json`: waiting, failure, or ready-for-visual-review.
- Expected PPT: `output/CoBRA_Group_NMF_RNCut_Editable_20261009.pptx`.

The first v1 attempt produced no new HCP measurements: its source-loader wrapper
needed the original verified input record. It remains preserved. Early component
tests also distinguished arbitrary component ID ordering from component
membership; the final gate checks membership/counts and full RNcut labels.

Official sources: https://www.nitrc.org/projects/rncut and
https://pmc.ncbi.nlm.nih.gov/articles/PMC7410361/ .
