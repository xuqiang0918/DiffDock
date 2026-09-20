# SDAA 适配说明（DiffDock）

DiffDock 是 [gcorso/DiffDock](https://github.com/gcorso/DiffDock) 的分子对接扩散模型（配体盲对接 + confidence 排序）。本文件记录其从 NVIDIA CUDA 迁移到 Teco SDAA 加速卡的适配过程：环境、改动、配置、结果。

---

## 1. 适配环境

| 项目         | 版本/位置                                                     |
| ---------- | --------------------------------------------------------- |
| 模型         | DiffDock-L（`adapt/sdaa` 分支，基线 commit `85c49b6`）           |
| 硬件         | 太初 SDAA 卡 × 32（`torch.sdaa.device_count()==32`）           |
| 操作系统       | LoongArch64 / Linux 6.6.52-1.7.lns23 / glibc 2.38 / 128 核 |
| PyTorch    | 2.12.0                                                    |
| Torch-SDAA | 3.2.1                                                     |
| Python     | 3.12.13（`/home/py312`）                                    |
| CUDA 基线    | NVIDIA H100（8 卡机）+ rdkit 2022.03.3，同批输入对照                 |

实测依赖版本（`+sdaa` 为太初已适配版本）：

```
torch 2.12.0            torch_sdaa 3.2.1        e3nn 0.5.1
torch-geometric 2.8.0.post1                     torch_scatter 2.1.2+sdaa
torch_sparse 0.6.18+sdaa                        torch_cluster 1.6.3+sdaa
rdkit 2026.3.5          ProDy 2.6.1             openbabel 3.2.1
spyrmsd 0.7.0           fair-esm 2.0.1          opt-einsum-fx 0.1.4
numpy 2.5.2             scipy 1.18.0            pandas 2.3.3
biopython 1.88
```

> 关键事实：Teco 的 PyTorch 构建里 `torch.cuda` **存在但 `is_available()==False`**，真正的加速器由  
> `torch.sdaa` 暴露。因此原代码里所有 `torch.cuda.is_available()` 判断在 SDAA 上都会错误地落到  
> CPU —— 所有设备选择必须显式探测 `torch.sdaa`（见第 2 节）。

---

## 2. 适配策略

- 验收主线为官方推理入口 `python -m inference`（README 用法），在 SDAA 上直接运行，命令行参数不改；
- 代码改动分两类：**平台适配**（设备派发，标 `[sdaa-adapt]`）与**通用兼容**（rdkit 版本差异，标  
  `[upstream]`），均为最小改动，不触碰模型结构与数据流程（见第 3 节）；
- 精度与性能见第 6 节。

---

## 3. 改动清单（已跟踪文件 6 个，59 insertions / 14 deletions）

### 3.1 新增文件

| 文件                        | 用途                                                   |
| ------------------------- | ---------------------------------------------------- |
| `utils/accelerator.py`    | 设备后端抽象层（95 行），加速器 / CPU 由它统一判定                       |
| `sdaa/scripts/dd_acc.py`  | 精度评测（复用官方 `utils/molecules_utils.get_symmetry_rmsd`） |
| `sdaa/SDAA_ADAPTATION.md` | 本文档                                                  |
| `sdaa/patches/`           | 补丁归档                                                 |

模块被 import 时自动调用一次 `apply_platform_defaults()`，任何入口只要碰到这个模块补丁即生效。接口：

```
accel_available()          是否有可用加速器（CUDA 或 SDAA）
device_count()             加速器数量
get_device()               返回 torch.device('cuda' / 'sdaa' / 'cpu')
empty_cache()              释放当前加速器缓存
jit_script_available()     探测 torch.jit.script 是否真的能编译
apply_platform_defaults()  注入当前后端需要的库默认值
```

放在 `utils/` 而非 `sdaa/`：官方入口直接 import 仓库内模块，无需 `sys.path` 拼装。

### 3.2 修改文件

| # | 位置                                             | 修改                                                             | 类型                  |
| - | ---------------------------------------------- | -------------------------------------------------------------- | ------------------- |
| ① | `inference.py:153`                             | `device` 探测改 `get_device()`                                    | `[sdaa-adapt]` 平台适配 |
| ② | `evaluate.py:238`                              | 同上                                                             | `[sdaa-adapt]` 平台适配 |
| ③ | `evaluate.py:350`                              | `torch.cuda.empty_cache()` → `empty_cache()`                   | `[sdaa-adapt]` 平台适配 |
| ④ | `utils/inference_utils.py:76`                  | ESM token 上卡改走 `accel_available()` / `get_device()`            | `[sdaa-adapt]` 平台适配 |
| ⑤ | `utils/inference_utils.py:108`                 | ESM 结构生成里的 `empty_cache()`                                     | `[sdaa-adapt]` 平台适配 |
| ⑥ | `utils/inference_utils.py:145`                 | ESM2-650M `model.cuda()` → `model.to(get_device())`            | `[sdaa-adapt]` 平台适配 |
| ⑦ | `utils/inference_utils.py:173`                 | ESMFold `model.eval().cuda()` → `.to(get_device())`            | `[sdaa-adapt]` 平台适配 |
| ⑧ | `utils/print_device.py:7`                      | 默认设备先问后端，再回落到 MPS / CPU                                        | `[sdaa-adapt]` 平台适配 |
| ⑨ | `utils/sampling.py:109`                        | 重新 batch 后补 `.to(device)`                                      | `[sdaa-adapt]` 平台适配 |
| ⑩ | `datasets/process_mols.py:305` / `398` / `411` | 新增 `restore_legacy_remove_hs_semantics()`，并在两处 `RemoveHs` 之后调用 | `[upstream]` 通用兼容   |

> `[upstream]` 类修改为平台无关的通用修复，建议单独提交并可回馈上游；`[sdaa-adapt]` 为平台适配本体。

### 3.3 无需改动的部分

- 模型结构与数据流程零改动；
- 官方入口 `inference.py` / `evaluate.py` 的命令行参数与 yaml 配置不变；
- 其余原仓代码零改动。

### 3.4 平台适配的三个卡点

**① 设备派发**：原代码全部依赖 `torch.cuda.is_available()`，SDAA 上恒为 False，会静默落到 CPU。已全部改走抽象层。

**② `torch.jit.script` 是空实现** —— 该 Torch-SDAA 构建里 `torch.jit.script(f)` 直接返回原来的 `<class 'function'>`，没有编译；`torch.jit.is_jit_enabled` 甚至不存在。而 e3nn 0.5.1 的 `e3nn/util/codegen/_mixin.py:44` 会断言 `isinstance(scriptmod, torch.jit.ScriptModule)`，导致 **score model 第一层 TensorProduct 就构造不出来**。

修法用 e3nn 官方逃生口 `e3nn.set_optimization_defaults(jit_script_fx=False)`（`_mixin.py:42` 的 `opt_defaults["jit_script_fx"]` 分支）：改注册 `fx.GraphModule`、eager 执行，绕过 TorchScript。`apply_platform_defaults()` 只在探测为假时才动手。

**③ PyG 重新 batch 后 `batch` 张量落 CPU**：`torch_geometric/data/collate.py:112-115` 用 `value.is_cuda` 决定 `batch`/`ptr` 辅助向量建在哪个设备上，SDAA 张量上这个探测恒为假，于是重建出的图里真实属性都在卡上、`batch` 却在 CPU，模型随即在 `batch[edge_index[0]]`（`models/cg_model.py:301`）崩掉。

触发条件是 `crop_beyond` 分支：官方权重 `workdir/v1.1/score_model/model_parameters.yml:28` 默认 `crop_beyond: 20`，于是**每个去噪步**都要 `to_data_list()` → `crop_beyond()` → `Batch.from_data_list()`。`utils/sampling.py` 该处原本漏了 `.to(device)`（同一个函数里 confidence 分支本来就带着），补上即可。

### 3.5 通用兼容性修复（`[upstream]`）

`datasets/process_mols.py` 的三处改动由**依赖版本**引起、与设备无关，因此不标 `[sdaa-adapt]`：官方  
`environment.yml` 钉的 `rdkit==2022.03.3` 是训练时的语义，本机是 rdkit 2026.03.5，两者 `RemoveHs`  
对氢的记账方式不同，会改变配体节点特征、进而改写对接结果（现象与实测见 5.4）。

补丁新增 `restore_legacy_remove_hs_semantics()`，把隐式氢补进 `NumExplicitHs` 并置 `NoImplicit`，  
恢复官方钉版本的语义。

**补丁归档**：

| 补丁                                                                 | 内容                    | 基线        |
| ------------------------------------------------------------------ | --------------------- | --------- |
| `sdaa/patches/0001-upstream-compat-DiffDock-rdkit-removehs.patch` | 通用兼容（3.5）             | `85c49b6` |
| `sdaa/patches/0002-sdaa-adapt-DiffDock-device-backend.patch`      | 平台适配（3.1 / 3.2 / 3.4） | `85c49b6` |

---

## 4. 运行前配置

```bash
# ① 容器：镜像 pytorch:3.2.1-torch_sdaa3.2.1，以 xuqiang 身份进入
#    PyG 会调 getpass.getuser()，镜像里 uid 1007 无 /etc/passwd 条目时会抛 KeyError
docker run ... -e USER=xuqiang -e LOGNAME=xuqiang -e HOME=/data/application/xuqiang ...

# ② 容器内激活运行时
source /opt/tecoai/setvars.sh
export PATH=/home/py312/bin:$PATH

# ③ 权重：官方 diffdock_models.zip（129.8 MB）解到 <model_dir>，
#    得到 workdir/v1.1/score_model 与 workdir/v1.1/confidence_model；
#    ESM2-650M（esm2_t33_650M_UR50D.pt 及 contact-regression 头）放
#    $HOME/.cache/torch/hub/checkpoints/（$HOME 决定该缓存目录，见 ①）

# ④ 首跑一次性预计算：utils/torus.py 在模块导入时算 SO(3)/torus 查找表
#    （约 5×10⁹ 次 np.exp，grad() 再来一遍），龙芯上耗时约 12 分钟，
#    结果缓存为仓库根的 .p.npy / .score.npy（各约 200 MB，已被 .gitignore 覆盖）。
#    只付一次，之后启动不再重算；中途被杀会留下不完整缓存，删掉重跑即可。
```

> SSH 会话断开会连带杀掉 `docker exec` 起的进程；长跑用 `docker exec -d` 分离启动并把输出重定向到日志。

---

## 5. 运行测试（实测记录）

### 5.1 输入与运行

输入沿用官方 csv 格式：

```csv
complex_name,protein_path,ligand_description,protein_sequence
1a46,examples/1a46_protein_processed.pdb,examples/1a46_ligand.sdf,
```

`protein_path` 与 `protein_sequence` 二选一（都没有会走 ESMFold 现场折叠，未验证）；  
`ligand_description` 可以是 sdf 路径或 SMILES。

```bash
cd <repo>
python -m inference \
  --config default_inference_args.yaml \
  --protein_ligand_csv <input.csv> \
  --out_dir <out_dir>
```

其余可调项见 `default_inference_args.yaml`（`inference_steps: 20`、`actual_steps: 19`、  
`samples_per_complex: 10`、`batch_size: 10`）。

> **上游行为提醒**：`inference.py:114-118` 会用 yaml 内容**无条件覆盖**命令行解析结果，所以  
> `--samples_per_complex N` 传在命令行是无效的，改样本数必须改 yaml。这是官方代码行为，不是平台问题。

输出目录结构：

```
<out_dir>/<complex_name>/
├── rank1.sdf                          # 无 confidence 的原始第一名
├── rank1_confidence-<score>.sdf       # 按 confidence model 排序后的 1..N
├── ...
└── rank10_confidence-<score>.sdf
```

文件名里的数字是 confidence model 打分，越大越好；rank1 是最终推荐位姿。

### 5.2 设备识别

```
torch.cuda.is_available(): False
torch.sdaa.is_available(): True
torch.sdaa.device_count(): 32
```

### 5.3 官方 CLI 端到端

```
python -m inference --config default_inference_args.yaml \
  --protein_ligand_csv inputs/single.csv --out_dir results/acceptance
```

`START 05:20:17 → END 05:27:52`，**455 s**，`INFER_RC=0`，产出 11 个文件（`rank1.sdf` + 10 个位姿），  
10 个位姿按 confidence 严格单调递减，日志无报错。运行中 teco-smi 显示该卡  
`SPE-Util 98~100%`、进程 RSS 3.0 GB，确认计算真的落在 SDAA 上。

### 5.4 rdkit 版本兼容 —— 精度问题的根因与修复

**现象**：同一份代码、同一个输入、同一条确定性轨迹（固定起点 + 零噪声 + 同一份权重），gpu3 出 2 Å  
档、SDAA 出 22.65 Å。

**根因**：配体节点特征 `data['ligand'].x` 完全由 rdkit 版本决定，与设备/平台无关。官方  
`environment.yml:29` 钉的 `rdkit==2022.03.3` 是训练时的语义；rdkit 2024.03 起 `RemoveHs` 不再给重原子  
置 `NoImplicit` 标记，`GetImplicitValence()` 不再恒为 0，特征列随之漂移。本机是 python 3.12.13 +  
loongarch64，2022.03.3 早于 py3.12、PyPI 无 loongarch64 wheel，源码编译亦不可行，只能在代码层兼容。  
修法见 3.5。

**修复效果**（本机 SDAA，同批输入各 10 位姿，top1 RMSD）：6moa 1.82 → 0.85、1a46 32.38 → 1.56、  
6w70 8.63 → 2.43、6o5u 23.63 → 7.55，四个可跑复合物无一变差。

**回归 —— 补丁对官方钉的 rdkit 零影响**（gpu3，rdkit 2022.03.3，上游原版 vs 修正版同环境对照）：  
`data['ligand'].x` md5 五个可读分子逐个逐位相同；固定轨迹 1.99 / 1.18 / 0.71 对 2.00 / 1.18 / 0.72，  
修正版并行重跑 3 次得 2.00 / 1.97 / 1.99 —— 0.01 Å 是跑间抖动，补丁在 1a46 上不触发任何 setter；  
`matching=False`（`inference.py` 走）与 `matching=True`（`evaluate.py` 走）两支的 md5 均相同。

### 5.5 上游已知问题

- 官方 `examples/1cbr_ligand.sdf` RDKit 无法 sanitize（`AtomValenceException: Explicit valence for atom # 14 C, 5`），上游代码会打印 warning 后跳过该复合物，两侧环境都读不出来。
- `inference.py:305` 的 `logger.warning("Failed on", ...)` 缺格式符，真实异常会被 `TypeError: not all arguments converted during string formatting` 盖掉，排查时需从 `Arguments:` 里把原始栈读回来。

---

## 6. 精度与性能

### 6.1 精度（与晶体结构比 RMSD）

`examples/` 下官方测例，SDAA 每复合物独占一卡跑 10 个位姿，与晶体配体算对称化 RMSD（官方口径 `sdaa/scripts/dd_acc.py` → `utils/molecules_utils.get_symmetry_rmsd`，阈值 2 Å）。CUDA 基线为 H100 + rdkit 2022.03.3（官方钉的版本），同批输入、同口径：

| 复合物             | SDAA top1 / 最优  | CUDA 基线 top1 / 最优 |
| --------------- | --------------- | ----------------- |
| 6moa            | **0.85 / 0.83** | 0.91 / 0.91       |
| 1a46            | **1.56 / 1.56** | 7.38 / 3.71       |
| 6w70            | **2.43 / 2.43** | 2.47 / 2.47       |
| 6o5u            | **7.55 / 4.79** | 8.23 / 4.63       |
| 6ahs            | 进程异常（0 位姿）      | 7.09 / 7.09       |
| Top-1 命中（< 2 Å） | **2 / 5**       | 1 / 5             |

SDAA 五个可跑复合物中四个不差于 CUDA 基线，Top-1 命中 2/5（6moa、1a46）。6ahs 在 SDAA 上采样起步即崩，见第 7 节第 2 条；`1cbr` 两侧都因参考 SDF 不可 sanitize 被跳过，见 5.5。两侧各只跑一轮随机采样，后端 RNG 实现不同（同 seed 也不是同一条噪声序列），逐例优劣不构成系统性结论。

样本是仓库自带的 `examples/`（加工过的单口袋蛋白），不是 PDBBind 测试集，5 个样本的命中率不构成对模型精度的评价。

### 6.2 性能与峰值显存

同一批输入、同一组参数（10 位姿 / seed 1234），各侧独占一张卡：

| 指标             | SDAA（龙芯单卡）                          | CUDA 基线（H100）                      | 倍率    |
| -------------- | ----------------------------------- | ---------------------------------- | ----- |
| 单复合物 10 位姿 总耗时 | 556.6 s                             | **26.5 s**                         | 21.0× |
| 4 复合物批量 总耗时    | 1555.2 s                            | **44.0 s**                         | 35.3× |
| 峰值显存（单复合物）     | alloc 4025 MiB / reserved 11952 MiB | alloc 2634 MiB / reserved 7122 MiB | —     |
| 峰值显存（4 复合物批量）  | alloc 4026 MiB / reserved 12108 MiB | alloc 2714 MiB / reserved 7260 MiB | —     |
| 同机 CPU 单复合物    | 851.5 s（64 线程）                      | —                                  | —     |

倍率为 SDAA ÷ CUDA。reserved 峰值占 SDAA 单卡可见容量（15296 MiB）的 78%，余量约 3.0 GB，仍  
**建议独占一张卡**：与别的进程共卡会被挤到 OOM（实测出现  
`RuntimeError: sdaa out of memory. Tried to allocate 974.00 MiB`）。

---

## 7. 已知限制

1. **需独占单卡**：见 6.2。
2. **偶发设备级异常**：采样阶段偶有 `SDAA_ERROR_INVALID_ADDRESS_SPACE` / `SDAA_ERROR_DMA_RMA_ACE_UNFINISH`（pc 落在 fatbin 内、无 Python traceback，进程直接 abort），换卡重跑可过；12 张卡的基础算子（matmul / svd）自检全部正常，不是坏卡。同一进程内串行跑多个复合物时更容易触发，实测改为**每复合物一进程**后稳定。
3. **官方 `inference.py` 无卡号开关**：`get_device()` 只返回 `torch.device("sdaa")`，落默认卡。多卡并跑时通过 `sitecustomize` + 环境变量 `DD_SDAA_DEV` 在进程外选卡，无需改官方代码。
4. **e3nn 走 FX codegen 而非 TorchScript**：该 Torch-SDAA 构建没有可用的 `torch.jit.script`，只能退回 eager 执行。数值已验证一致，但速度与脚本化实现相比的差距未量化。
5. **手性特征列的差异未消除**：rdkit 2022.09 起立体感知改动导致，补丁不涉及（5 个可读分子中 4 个有，6moa 无）。
6. **性能为跨代比较**：6.2 的 CUDA 基线是 H100，与 SDAA 不是同一代硬件，倍率只作量级参考。
7. **ESMFold 分支未验证**：`protein_path` 与 `protein_sequence` 都为空时会调 ESMFold 现场折叠，该路径只做了设备派发层面的改动，未跑过。
8. **`evaluate.py`（PDBBind 全量评测）未跑**：只做了设备派发改动，未在 SDAA 上执行过完整评测流程。
9. **未适配训练 / 微调**。

---

## 附：验证记录

- `torch.sdaa` 正确识别（32 卡）；
- `torch.cuda.is_available()==False`、`torch.sdaa.is_available()==True`；
- 12 张卡的基础算子自检（matmul / svd）全部正常。

```bash
# ① 官方推理入口（在 115 宿主上执行；输入 1a46，结果落 results/acceptance）
docker exec -d sdaa bash -c "source /opt/tecoai/setvars.sh && export LANG=C.UTF-8 && cd /data/application/xuqiang/DiffDock && /home/py312/bin/python -m inference --config default_inference_args.yaml --protein_ligand_csv /data/application/xuqiang/diffdock_runs/inputs/examples_1a46.csv --out_dir /data/application/xuqiang/diffdock_runs/results/acceptance > /data/application/xuqiang/diffdock_runs/logs/acceptance.log 2>&1"
# 产出 11 个文件；单复合物 10 位姿约 455 s，进度看 logs/acceptance.log

# ② 精度（官方口径 utils/molecules_utils.get_symmetry_rmsd）
#    --results 给「一批结果」的父目录，其子目录名即复合物名（可给多个批次）
docker exec sdaa bash -c "source /opt/tecoai/setvars.sh && cd /data/application/xuqiang/DiffDock && /home/py312/bin/python sdaa/scripts/dd_acc.py --ref-tmpl 'examples/{name}_ligand.sdf' --results /data/application/xuqiang/diffdock_runs/results/acc7_corr --cutoff 2.0 --json /data/application/xuqiang/diffdock_runs/results/acc7_recheck.json"
# 预期：top1 RMSD 0.85 / 1.56 / 2.43 / 7.55，其中 2 个 < 2 Å
```

