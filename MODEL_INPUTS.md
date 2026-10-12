# 两种图和两个128
原始FC C=YᵀY/T 对称，可解释为功能图边权。但当前代码采用 C 的每一行作为一个皮层顶点的连接profile；Graph-CNN消息传播使用皮层三角网格。

个人模型：N×N FC → Linear(N,128)+LayerNorm+GELU → N×128 → 两层mesh Graph-CNN → 几何patch汇总128×128 → 两层Transformer → 广播到N×128并门控融合 → μ/logσ²，各N×32 → decoder重建N×N。

N是同一半球所有皮层grayordinates；投影不改变N、不减少顶点、不平均原始BOLD。W为N×128，h_ip=Σ_j C_scaled,ij W_jp+b_p，所有连接参与但有损压缩到128维。

第一个128是个人模型每个顶点的特征维数；第二个128是Transformer上下文token数量，两者独立。population模型宽度64，因此其token矩阵为128×64，潜变量仍32维。

Mesh消息传播边是局部解剖邻接，双向支持；row-normalized D⁻¹A 的数值可不对称，但拓扑仍无向。长程FC通过顶点profile参与；Transformer再提供全局上下文。当前模型不是FC加权消息传播GCN。功能边权图或mesh/FC双图是新的消融方案，不在本次代码整理中偷偷加入。

输入逐列scale可能破坏数值对称；decoder未强制对称/PSD，full N²损失含上下三角与对角。连接组应用直接从ROI时序计算对称Pearson FC，不使用decoder输出作为连接组。完整限制见METHODS_AND_LIMITATIONS.md。
