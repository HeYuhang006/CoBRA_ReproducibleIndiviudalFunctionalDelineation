# 从头检查：执行顺序与 I/O

## 0. 固定新实验路径和环境

先按 [docs/EXTERNAL_RESOURCES.md](docs/EXTERNAL_RESOURCES.md) 准备未上传的作者包、图谱、模板和原始数据。安装新环境或复用经过检查的环境，按下面顺序手动执行；GitHub 上传本身不会启动训练。

```bash
cd /dataspace/yuhang_data/IndividualFunctionalParcellation/CoBRA
PY=/home/yuhang/miniconda3/envs/pytorch/bin/python
```

先手动检查 `configs/project.json`。`work_root` 必须是新的检查目录，不能设成旧 `GPT_workplace` 或代码目录。默认 GPU0；执行前自己确认 GPU 空闲。脚本只用配置中的一张卡，不会终止别人的进程。

数据仍读取 `/dataspace/yuhang_data/HCP-YA`、`HCP-Retest`。`ancillary_input_root` 只允许读取已有的原始 motion 文本及结构表面；`task_input_root` 是已下载的原始 task z-statistic 输入。不会读取旧 checkpoint/FC/label/FD.npy/area.npy。模板、Workbench、MATLAB 路径在 config 中注明。

```bash
$PY Main/s_step0_check_environment.py
$PY Main/s_step1_prepare_data.py --dry-run
```

step0 做 CPU 数值检查，不启动训练，不创建科学实验输出。它检查 2/4 扫描 full-FC 损失/投影/梯度、连通约束、ICC(A,1)、FD 角度单位与面积加权 FC。

当前 pytorch 环境缺少最终PPT需要的 `python-pptx` / `shapely`。在运行step7之前，可只安装到代码包的独立依赖目录，避免改动正在运行任务的环境：

```bash
$PY -m pip install --no-deps --target .deps python-pptx shapely lxml
```

NumPy、Pillow 等使用现有环境；如果有其他缺项，按 `requirements.txt` 在专门的检查环境中补齐。不要在当前训练使用的环境里直接整体升级。

## 1. CoBRA 主流程（必须按顺序）

| 顺序 | 命令 | 输入 → 输出 |
|---|---|---|
| 1 | `$PY Main/s_step1_prepare_data.py` | 原始 CIFTI、motion、表面、冻结身份 → 新 mapping、edges、LPE、32 人几何均值、128 context patches、重新计算的 FD/QC/面积、400/36 人 cohort |
| 2 | `$PY Main/s_step2_train_population.py` | 450 人训练数据 → 全 FC 群体均值/scale → width64 group 1000 steps → 450 人跨天共享训练 50 epochs；左右半球 |
| 3 | `$PY Main/s_step3_build_group_prior.py` | 完成的 population checkpoint → 真正 group FC embedding → embedding+LPE1 K-means → connected functional refinement，K100/200/300/400；另外从头训练 width128 personal neural group prior（双侧各1000 steps） |
| 4a | `$PY Main/s_step4_adapt_individual.py --cohort retest` | 每个 visit 的 REST1 或 REST2 LR/RL → 独立重置 group prior → 30 steps personal fit → connected maps；Day1、Day2、Retest |
| 4b | `$PY Main/s_step4_adapt_individual.py --cohort YA400` | 同上，独立400人；两个 baseline day；所有K |
| 5 | `$PY Main/s_step5_validate.py --cohort all` | 完成的个人图谱 + 独立 REST2 → H、adjacent silhouette、DCBC、分区相似度/连通性；同时生成 secondary maps 的比较所需连接测量 |
| 6a | `$PY Main/s_step6_applications.py --mode connectome --cohort all` | 固定校准 ROI + 独立测量 → 边级 ICC(A,1)、FC相关/MSE/identifiability、识别率、wrong-person、配对bootstrap/permutation |
| 6b | `$PY Main/s_step6_applications.py --mode task` | 第一批锁定40名YA被试的七类task输入 → 原始task方差、面积归一化方差、R²、配对CI与Holm；缺失/QC失败记录，不替补有利个体 |

step1 的过滤缓存不会来自旧试验。0.01–0.1 Hz、三阶 Butterworth、detrend、零相位过滤；训练和评估片段**分别过滤**，先过滤再 motion censor，再重新标准化。不同 PE/run 不是连续时间序列。

step3 为 DCBC 和 z-score 消融重建450训练人的 MSMAll 平均 geodesic/分箱统计，耗时与磁盘开销较大。所有数值数组在新输出位置生成。

stage 状态在 `work_root/stage_status/<method>/`。失败后先读日志/状态，确认配置和代码未变化，再在**原命令**末尾加 `--resume`。这只续接新检查目录，不会载入旧实验。

## 2. 必补的消融分支

在主流程 step3 后执行：

```bash
$PY Main/s_step4_adapt_individual.py --mode controlled --cohort retest
$PY Main/s_step4_adapt_individual.py --mode controlled --cohort YA400
```

真正单天：Day1 LR/RL 各480帧；另一单天对照 Day2 同样。等时长 cross-session：两天×LR/RL×240帧，总计同为960 nominal frames，统一初始化和30 steps。实际保留帧数报告，不能宣称 censor 后时长完全相等。独立过滤的尾部片段与长期 Retest 做验证；Retest 不参与这一 cross-day 拟合。

同一个模型/中心/初始图谱/能量函数的 `no_hard_connectivity` 已包含在 step4/5/6，用于连通性对照。

FC distance z-score 的**population 训练**消融：先完成 raw step2，再运行：

```bash
$PY Main/s_step2_train_population.py --mode distance
```

此入口保留论文式训练群体 FC 分箱统计，不在每个验证被试上重拟合统计。它与主方法 raw-FC 分支分开，不能把 normalized population 输出冒充当前最终 personal prior。

## 3. DRFC / MICCAI26 group baseline

先完成 Main step1（共享身份与结构输入）。随后：

```bash
$PY DRFC/s_step1_prepare_data.py
$PY DRFC/s_step2_train_population.py
$PY DRFC/s_step3_build_group_prior.py
$PY DRFC/s_step5_validate.py
$PY Main/s_step5_validate.py --mode group
```

DRFC step1 从原始数据重算500/42/50人的FC、训练群体距离分箱 z-score、对比样本池。step2=200 epochs MLP；step3=K100/200/300/400；step5=其原有检查。Main group验证再用相同50名 heldout个体逐人计算 H、exact FC-profile cosine silhouette、DCBC，纳入 Schaefer、多种可用外部图谱、CoBRA raw/connected。两种 silhouette 定义分开标注。

DRFC 是固定 group atlas，step4 个人拟合不适用。固定 atlas 跨扫描 ARI=1 是标签相同的数学事实，不是个体化成功。

## 4. 三时间点个体对照：K100 + 18网络

前提：Main step1、step3；Main step4 retest 完成后才比较。默认36个匹配被试、Day1/Day2/Retest。

### 4.1 先准备作者18网络与matched输入

```bash
$PY Wang_style/s_step1_prepare_data.py --K 18
$PY Wang_style/s_step3_build_group_prior.py --K 18
$PY Wang_style/s_step4_adapt_individual.py --K 18
$PY Main/s_step4_adapt_individual.py --mode network18 --cohort retest
$PY TraditionalMethod/s_step4_adapt_individual.py --K 18
```

Wang step1=官方 fsaverage4/6mm 输入准备；step3=Python移植与作者MATLAB标签算术等价检查；step4=后续HFR18作者算法。Main network18 使用同样输入、分布式18网络prior、每个原始连通分量的固定种子，保留原有分布式网络，不制造新的组件。它不是把100区 decoder 的“每区1个组件”直接搬过来。

### 4.2 100区对照

```bash
$PY Wang_style/s_step1_prepare_data.py --K 100
$PY Wang_style/s_step4_adapt_individual.py --K 100
$PY TraditionalMethod/s_step4_adapt_individual.py --K 100
$PY TraditionalMethod/s_step5_validate.py --K 100
```

PCA/Ward 默认也生成100/200/300/400，只有正式匹配图使用100。Wang100 明确为适配诊断，不得写成原文 Wang2015 原生100区。

### 4.3 GPIP 官方私有源码

```bash
$PY GPIP/s_step2_train_population.py
$PY GPIP/s_step1_prepare_data.py --K 18
$PY GPIP/s_step1_prepare_data.py --K 100
$PY GPIP/s_step4_adapt_individual.py --K 18
$PY GPIP/s_step4_adapt_individual.py --K 100
$PY GPIP/s_step5_validate.py
```

GPIP step2 在这里是 MATLAB 运行时/协方差/Ncut/checkpoint 数值检查，**不是**神经网络训练。GPIP step1 要先通过这项检查。step4 每个visit单独联合拟合36人，固定20轮；不会跨visit借数据。`--visit Day1/Day2/Retest` 可逐个手动运行。18是分布式网络扩展，100是左50/名义双侧100扩展，都非GPIP论文原生58/338设置。

GPIP step5 等所有5方法×2尺度×3时间点完整后才生成正式指标。ARI、Hungarian macro Parcel Dice、fragment/excess-component、intra/inter相似性、specificity gap、tie-aware识别率均从新图谱计算；被试为统计单位。

## 5. 统一统计与作图（最后执行）

```bash
$PY GPIP/s_step5_validate.py
$PY Main/s_step5_validate.py --mode group
$PY Main/s_step7_draw_figures.py
```

输出 `work_root/reports/`、`work_root/figures/CoBRA_Editable_Comparisons.pptx`。个体图左18、右100，5方法×Day1/Day2/Retest；中间箱线图、配对Holm P值。group图含多K脑表面以及全部可用图谱的独立点/误差线，分类K轴明确标注，无连线。脑区是原生可编辑PPT多边形，统计是原生形状和文字，零图片对象；CSV和固定Day1显示色标映射一并保存。后续应人工检查视图与投稿版式。

## 6. 本包边界

- 不自动启动全量重跑；不创建后台队列；不删除、迁移或替换旧结果。
- 默认复跑冻结的 **450人 VAE + 400人独立验证 + 36人匹配Retest** 协议，不能写作500人VAE。
- frozen cohort JSON 记录实验设计身份和原始输入定位；FD、面积、FC、模型和标签都重新计算。
- 本次发布的文件哈希与检查记录见 `RELEASE_MANIFEST.json`、`RELEASE_AUDIT.json`。整理前的审计放在 `docs/provenance/`，不能作为本次全量复跑完成的证据。
- GPIP/Wang100 尚无已验证的双侧 frozen-ROI task/connectome正式对照；对应step6明确停止，避免把诊断版本伪装为发表方法结果。
- RNCut 已通过作者原生优化器与表面适配层等价性检查，正式名称为 RNCut (surface adaptation)。当前生产扩展、初始化修复和路径依赖见 `docs/EXPERIMENTS.md`；它与干净复跑主入口分开。cc-SUNET 原生任务不对等，仍未纳入。
- 数据/模板/author resources 缺失会直接报告缺项，不回退到旧模型或旧标签。

## 7. Group NMF（可选群体对照）

完成 Main step1 和 DRFC step1 后手动执行：

```bash
$PY TraditionalMethod/s_step2_train_population.py --mode group_nmf --dry-run
$PY TraditionalMethod/s_step2_train_population.py --mode group_nmf
```

此入口使用新 work_root 中500人 group FC 的正值部分，保持生产版本的 rank32、NNDSVDa 初始化、MU 停止条件与 K-means 配置；没有使用 VAE embedding。输出位于新 work_root 的 `runs/group_subject_boxplots_20261008_v1/NMF`。这只生成 NMF 图谱；它尚未接入 Common 的统一 group 绘图入口。生产版 NMF/RNCut 50人评估、60项 Holm 校正与打包脚本保存在独立实验目录，依赖与限制见 `docs/EXPERIMENTS.md`。
