# CoBRA

**CoBRA: Reproducible Individual Cortical Parcellation via Connected Boundary Refinement and Cross-Session Adaptation**

Self-supervised cortical functional parcellation using full-FC Graph-CNN–Transformer VAE embeddings, a connected group prior, and individual boundary adaptation.

**从头复跑请先读 [README_运行顺序.md](README_运行顺序.md)。** 本仓库包含代码、配置和冻结的 cohort 身份元数据；不包含 HCP 数据、FC 缓存、checkpoint、已生成的图谱或统计结果。上传不启动新训练，也不移动原实验结果。

## Workflow and layout

`Main` 放我们的方法；`DRFC`、`Wang_style`、`GPIP`、`TraditionalMethod` 按相同阶段组织。自有函数模块为 `f_*.py`，执行入口为 `s_stepX_*.py`。PyTorch 类接口及作者 MATLAB 必须匹配的函数名保留原名。

| Stage | Main directory | Operation |
|---|---|---|
| 0 | `Main/s_step0_check_environment.py` | 环境及 CPU 数值检查 |
| 1 | `Main/01_data` | CIFTI、0.01–0.1 Hz 过滤、motion QC、网格 |
| 2 | `Main/02_network`, `03_population` | 完整 FC 编码/重建、群体与跨扫描训练 |
| 3 | `Main/04_parcellation` | embedding+LPE、K-means、连通群体图谱、个人 neural prior |
| 4 | `Main/05_individual` | 每个 visit 独立微调、连通边界推断、等时长消融 |
| 5 | `Main/06_validation` | 独立数据质量、ARI、Dice、inter/intra、连通性 |
| 6 | `Main/06_validation` | task 与独立连接组应用 |
| 7 | `Main/07_figures`, `Common` | 统一统计、脑表面及可编辑 PPT |

每个方法目录包含 step0–step7 入口；不适用的阶段明确报告 N/A。跨方法依赖和详细命令见运行顺序，不应把五个方法的步骤任意互换。

## Network code

- [f_surface_prior_vae.py](Main/02_network/f_surface_prior_vae.py)：两层皮层网格 Graph-CNN、128 context tokens、两层 Transformer、门控融合、顶点级 μ/logσ²、32维 latent。
- [f_full_fc_vae.py](Main/02_network/f_full_fc_vae.py)：完整 N×N FC 的精确代数投影与重建损失，不抽取 FC 边。
- [f_model_ops.py](Main/02_network/f_model_ops.py)：跨扫描重建、一致性、group/core/mesh/参数正则。
- [f_validation_exact_runtime.py](Main/05_individual/f_validation_exact_runtime.py)：低内存个人训练及 dense full-FC 损失/梯度对照。
- [f_connectivity.py](Main/04_parcellation/f_connectivity.py)：固定种子、邻接边界移动、源区域移出后的连通性检查。
- [f_group_objective.py](Main/04_parcellation/f_group_objective.py)：训练群体 FC 上的连通边界优化。

FC 行是顶点的功能特征；Graph-CNN 的邻接来自双向皮层网格边。Transformer 门控融合发生在同宽度特征上，之后 VAE heads 才输出32维 embedding。完整定义与局限见 [METHODS_AND_LIMITATIONS.md](METHODS_AND_LIMITATIONS.md)。

## First checks

Prepare external inputs according to [docs/EXTERNAL_RESOURCES.md](docs/EXTERNAL_RESOURCES.md). Use a dedicated Python environment with `requirements.txt`; Workbench and MATLAB are separate requirements for the relevant stages.

```bash
cd /path/to/CoBRA_ReproducibleIndiviudalFunctionalDelineation
python Common/s_step0_audit_release.py
# Edit configs/project.json: raw paths, tools, GPU, and a NEW work_root.
python Main/s_step0_check_environment.py
python Main/s_step1_prepare_data.py --dry-run
# Then follow README_运行顺序.md manually, step1 through step7.
```

`work_root` must be separate from the repository and historical results. Code/config changes require a new output root. `--resume` explicitly resumes only the same new root. `--dry-run` prints the plan without loading MRI, allocating GPU tensors or creating scientific outputs.

## Protocol and experimental extensions

The clean main rerun freezes **450-person VAE training**, 400-person validation identities and 36 matched Retest identities before QC; DRFC uses500 training identities. These counts are not interchangeable. Nominal100/200/300/400 parcellations mean bilateral totals with50/100/150/200 parcels per hemisphere. The distributed18-network extension has different topology and interpretation.

Current NMF/RNCut production scripts and the width512/patch512 development experiment are preserved under method-specific `experiments/` folders. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for their commands, dependencies and separation from the clean main workflow. RNCut is labeled **RNCut (surface adaptation)** and retains the author optimizer; it is not ordinary spectral clustering. The512 experiment is a12-person development study, not a replacement for the final model or evidence of independent Retest improvement.

Author GPIP/RNCut/HFR distributions and atlas inputs are installed separately; their availability and reuse terms remain with their original authors. Original provenance records are retained in `docs/provenance/`. New release checks and per-file hashes are in `RELEASE_AUDIT.json` and `RELEASE_MANIFEST.json`.

Structural and numerical checks do not replace the planned full manual rerun. This source release makes no universal superiority or complete-convergence claim.
