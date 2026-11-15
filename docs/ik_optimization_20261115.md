# 2026-11-15 Base IK 优化记录

所有项目代码改动均在 gpu1 的 /export/ra/heqijun/EgoWhale。没有修改 depth、retarget、render/composite、用户 start.sh、机器人资产或配置中的 batch_size。81 个候选、基座几何筛选、基座微调和独立节点接口保留。工作区原有未提交修改不等于本轮全部由本任务产生。

## 生产实现

- egowhale/action/sequential_ik.py：按形状缓存 solver；L-BFGS CUDA graph，当前依赖的 LM seed 阶段保留 eager，避免已复现的第二次调用非法显存访问；尾批按剩余 episode 的 2 次幂收缩，仍完整补齐该形状。GPU 上完成基座平均位置误差选优，仅选中轨迹回传。
- 同文件：多有效解按与前帧距离选解，对资产中接近整圈的转动关节使用圆周距离，避免将 +179° 到 -179° 误判为 358° 而切换分支。短 episode 完成即输出，新 episode 可续入，互不继承 current_state。
- egowhale/action/base_ik.py：新增 _RefinementFK，通过公开几何 Jacobian 计算普通 PyTorch 位置/四元数损失的正确反向传播；复制库的可复用 FK 缓冲。未修改第三方源码。
- 同文件：接近整圈的 hinge 关节在连续展开的角度上 refinement/平滑，输出再映射回资产限位。禁止用线性平滑把 ±π 表示切换绕经 0°。其他关节仍按线性坐标及限位优化。没有硬编码 PiPer 关节名。
- 同文件：refinement 的 120 次 Adam 迭代使用每 episode 独立 CUDA graph；保持原位置/方向/连续性/jerk 权重。取消旧 _solve 在受保护平滑之后又执行的一次无保护平滑。
- sequential_ik.py：现有短区间修复按真实旋转步长识别跳变，允许相对该区间修复前最多额外 10° 方向误差；位置保护为 max(10 mm, 原误差 + 2 mm)。这不是整个节点对原始人手的绝对误差上限，也不是相对旧版本逐帧误差保证。
- 保存 jump_edges_before/after 为周期性处理后的姿态步长告警，新增 raw_coordinate_jump_edges 保留原始数值角度跳变计数。两种统计不能混淆。
- 此会话的 node_bench/ik_test.sh 已统一总体统计和多 worker 调度；总体 FPS 使用成功帧数/整次运行墙钟，支持单卡/多卡。不是各 worker FPS 相加。

## 为什么修复梯度

当前 thirdparty/curobo 提交 78fd485fa82d9b9a063fb4985e371814587e666a（2026-09-10）。旧 quaternion_gradient_to_angular_velocity 将四元数 covector 转到 body frame，随后与 base/world frame 关节轴相乘。普通 PyTorch quaternion loss 在这里的梯度不一致。

sort_beads/9 一个关节分量：旧自动梯度约 0.015894，有限差分约 0.002444，修正后约 0.002393。原 refinement 120 步把总损失从约 0.961 推高至 5.109，同条件修正后约 0.141。使用 q_dot = 0.5 * [0, omega_base] * q 将 covector 转到同一坐标系，再乘几何 Jacobian。

ARX/PiPer 全关节和基座参数真实 CUDA 有限差分通过。CPU 非共线双轴机构 double gradcheck、复用 FK 缓冲测试通过。不能据此认为所有残余跳变都来自梯度问题。

## 周期性处理的适用范围

_PERIODIC_RANGE_GAP=1°：仅资产定义为 hinge 且限位跨度至少 359° 的关节参与圆周距离和角度展开。当前 PiPer 的两侧 joint6 符合，ARX 的较短范围关节不符合；prismatic 不参与。

这是逐帧运动学合成的处理。Render 直接逐帧设置 qpos 并 mj_forward，不在两个 qpos 之间线性插值。输出值始终位于资产限位，但某个坐标可能从接近 +π 变为 -π。其渲染姿态变化小，不代表真实有硬限位执行器能越过机械止挡。若未来用于真机控制，应使用相应轨迹规划约束，不能直接把此合成输出当执行命令。

## 参数位置

- cfgs/bench8_sttn_balanced.yaml：nodes.action.ik_method=base_ik_seq；batch_size=5184，PiPer 每 actor 为 5184/(2*81)=32 episode。
- node_bench --workers N：单节点 worker 数，可在一个已分配 GPU 上配置多个本任务 worker。
- base_ik.py：_PERIODIC_RANGE_GAP=deg2rad(1)、_STEPS=120、_POSITION_WEIGHT=20、_ORIENTATION_WEIGHT=5。
- sequential_ik.py：_REPAIR_SPEED=3*pi rad/s，18 FPS 约 30°/帧；_REPAIR_REGIONS=4；_repair_jumps 的方向额外预算 10°、位置保护及局部迭代额度。
- 没有增加专门的腕关节惩罚，也没有改变 81 候选数。

## 已完成的测试及结果

实验根目录：/data/datasets/hqj/EgoWhale/datasets/fixed500_fps18/experiments/ik_opt_20261115

1. node_full（提速版，尚无正确 Jacobian）：物理 GPU 0/3，各一个 worker。500 输入，475 成功、25 几何拒绝、0 失败，117560 帧，108.923484 overall FPS。原始关节坐标 >30°/90° 帧边从旧系统 6121/1038 变为 6063/1056；仅提速未解决质量。
2. node_jacobian_full（正确 Jacobian，尚无完整周期性优化）：唯一空闲物理 GPU 0、两个 worker。475/25/0、117560 帧，107.175965 overall FPS。原始坐标 >30°/90° 为 5718/1108，仍不能称为连续性已解决。
3. production_periodic_smoke（正式周期性实现）：4 代表案例 3263 帧均成功，输出有限且在资产限位。两边采用相同姿态步长统计：>30° 从 99 到 22，>90° 从 17 到 0；新原始坐标 >90° 仍有 51 处，主要说明不能以 raw angle cut 代替视觉跳变。位置中位数从 1.883 到 2.625 mm，方向中位数从 26.22° 到 9.08°。sweep_dustpan/5 有原本就很大的跟踪偏差及局部退化，不能只报告好的分位数。
4. node_periodic_full：500 条已完整完成，controller 退出码 0。475 成功、25 几何拒绝、0 失败；与上一轮全部 500 条状态一致，成功 episode/帧数一致。源代码与 periodic_source_hashes.json 全部一致。详细结果见下节。

测试：production_tests.log 记录 66 passed、2 failed。失败为既有 tests/fixtures/arx_adapter_golden.npz 缺失及 tests/test_fps.py 引用不存在的 approach._resample_path；对应生产文件未改。新增 GPU 梯度及 ARX/PiPer、长度 1/16 的 graph refinement 测试均通过。最后的 raw 计数改动后 test_base_ik_seq.py/test_node_bench.py 再验证 22 passed；Ruff 和 git diff --check 通过。

与原系统结果跨时段、服务器共享负载及 GPU/worker 分配不同，单节点 overall FPS 不能写成整条 pipeline 的 system/fps，也不能把所有增益归因于某一项改动。

## cuRobo 新版核对（2026-11-15）

官方 v0.9.0（11 月 5 日发布）：
- 四元数反向传播已改为 quaternion_cotangent_to_angular_cotangent，明确采用 world frame 左扰动，公式与本次差分验证的修正一致。新版通过同一组回归后，可评估移除本地 Jacobian 适配，直接复用上游修正和 fused backward。
- LM IK step acceptance 修复：拒绝无效或使目标变差的更新，拒绝后保留已接受的目标值。值得与当前初值/连续选解回归单独 A/B，但这不证明当前 LM graph 崩溃已修复。
- MotionPlanner/BatchMotionPlanner.plan_pose_waypoints：有序 Cartesian 目标、位置或位姿跟踪、路径/终点分开容差。适合作为困难区间的轨迹优化候选，不应立即替换全部 81 候选搜索。官方示例默认路径 5 cm/15° 对我们偏松，必须另设更严位置预算，并验证固定视频帧的时间对齐。
- C++ runtime、加速运动学/Jacobian、GPU B-spline 优化值得后续性能试验。TimeOpt 和力矩/载荷约束以真实运动执行为主，不是当前固定帧率合成的优先项；优化时长可能破坏视频对齐。
- 当前 gpu1 驱动 570.211.01；最新官方安装文档要求 >=580.65.06。未升级依赖/驱动，未修改共享服务器全局配置。10x motion planning 发布标题不能直接解释为 EgoWhale FPS 提升 10 倍。

来源：
https://github.com/NVlabs/curobo/releases/tag/v0.9.0
https://raw.githubusercontent.com/NVlabs/curobo/v0.9.0/CHANGELOG.md
https://raw.githubusercontent.com/NVlabs/curobo/v0.9.0/curobo/_src/curobolib/kernels/common/quaternion_util.cuh
https://nvlabs.github.io/curobo/latest/getting-started/cartesian_waypoints.html
https://nvlabs.github.io/curobo/latest/getting-started/installation.html

## 未合入方案

多基座并行 refinement、更长迭代、全局 CPU 优化、容差候选图 DP 等试验在实验目录保存。部分产生较大末端误差，部分改善不足或过慢，没有合入生产节点。IKLink 的多样逐帧 IK + 动态规划提供后续困难区间处理思路，但本轮未完整实现该论文算法（https://arxiv.org/html/2402.16154）。

全量质量与视觉核查仍在继续，不将速度通过、方向梯度修复和全部跳变消失混为一谈。

## 最终全量回归结果（本轮版本，不代表残余问题已经全部解决）

- 输出：/data/datasets/hqj/EgoWhale/datasets/fixed500_fps18/experiments/ik_opt_20261115/node_periodic_full
- 117560 成功帧，wall_s=798.2810555，overall_fps=147.2664285。
- 单物理 GPU 0、两个本任务 worker、每 worker batch_size=5184。node_s 为两个 worker 累计服务时间，node_fps=77.5458493；用户关心的整次并行吞吐为上述 overall_fps，不能混用。
- 启动时 GPU 0 空闲，运行期间出现他人计算进程。观察记录 periodic_shared_gpu_observation.json；没有终止他人进程，没有在其占用后追加 GPU 测试。不能称为全程独占 GPU 的受控性能对照。
- 475 条的全部输出通过 finite/资产限位/帧数/关节顺序检查，235120 个有效手帧参与 FK 误差比较。
- 全量原始数据：validated_comparison.json；summary.json；results.jsonl。图：continuity_comparison.png。

| 指标 | 原 experiment_004_system | 本轮 |
|---|---:|---:|
| 几何步长 >30° 的帧边 | 6121 | 1191 |
| 几何步长 >90° 的帧边 | 1038 | 23 |
| 存在 >90° 几何步长的 episode | 297 | 17 |
| 原始数值坐标 >90° 的帧边 | 1038 | 2370 |
| 末端位置误差中位数 / 95 分位，mm | 1.909 / 46.065 | 1.782 / 57.950 |
| 末端方向误差中位数 / 95 分位，degree | 32.499 / 155.616 | 9.296 / 52.006 |

圆周统计对新旧两边采用同一资产规则。原始数值坐标的 2370 处包含 ±π 表示切换，不能直接当作渲染跳变；但也不能把数值连续性说成已经改善或将输出直接用于真机。

仍有 23 处 >90° 几何步长。部分目标本身方向突变，例如 fidget_magnetic_spinner_rings/1 的对应目标单帧约 104°；另一些仍是解分支问题。位置误差尾部退化必须保留：dry_hands/15 的最大误差从 15.1 mm 变为 570.9 mm，clip_unclip_papers/1 从 238.7 mm 变为 952.7 mm。全量另有 33272 个有效手帧位置误差相对旧结果增加超过 2 mm、2501 个方向误差增加超过 10°。这些不是该局部修复的预算违规结论，因为预算仅约束局部修复相对它自己的输入；它们说明整条节点尚没有逐帧质量下界保护。

扩大 CPU 修复窗口的离线试验能将 clip_unclip_papers/1 的最大几何步长 140.1° 降至 48.5°，约 9 秒；对另三个案例无有效改善，尚未合入。没有为取得漂亮指标而用旧结果替换失败帧。

## 视觉核查

全部 GPU 被占用后，使用 Mesa surfaceless EGL + llvmpipe 软件渲染，复用生产 _render 和 _composite，未修改节点源代码或服务器配置、未使用 GPU。回看原版最大跳变附近短片，预览高度 270 像素。不是全分辨率全 episode 的 render 性能测试。

路径：/data/datasets/hqj/EgoWhale/datasets/fixed500_fps18/experiments/ik_opt_20261115/visual_review_cpu/
color_2、clean_tableware_12、sort_beads_9、sweep_dustpan_5 各有 H.264 MP4 和相邻帧 PNG。左右分别原版/本轮。已核看 color 和 clean 的截图：相邻帧姿态更连贯；clean 原版近镜头的手臂遮挡明显减少，但局部手臂/物体遮挡的合理性并非这个计数能完全证明。

速度目标已达到，整体方向及严重跳变大幅改善；残余分支跳变与局部位置退化仍待处理，当前不能将整个优化目标标记为全部完成。


## 后续 CPU 离线候选试验（2026-11-15，未合入生产）

在所有 GPU 被其他计算进程占用时，使用四个 CPU worker 对 node_periodic_full 的 475 条成功结果检查，未启动 CUDA。每组存在 >90° 几何跳变或最大位置误差 >50 mm 时，固定该组现有 base，分别从首帧正向、末帧反向生成连续 IK 候选。沿用模型中的关节、限位、TCP 与周期关节信息，无 PiPer 特定腕部名称。该实验入口选用 PiPer，尚不能代替 ARX 等资产回归。

候选生成使用 MuJoCo 解析 Jacobian、方向 5° deadzone 和前一帧 seed。每组每方向的起点均与中心差分核对 Jacobian，475 条检查无失败。选路保留原轨迹作为候选，逐帧只接受位置误差相对输入增加不超过 2 mm、方向误差增加不超过 10° 的结果；采用严格字典序目标，先减少 >90° 帧边，再减少 >30° 帧边，再比较位姿/关节代价。10° 是本轮试验采用的数值，并非用户指定的精确容限。多 TCP 共基座资产仍需逐 TCP 预算回归。

- 475/475 检查完成，0 失败；191 条 episode、248 组手臂触发候选生成。
- 这些手臂中的 >90° 帧边：23 -> 19；>30° 帧边：762 -> 632。这里是触发组统计，不能将其冒充完整节点的全量 >30° 指标。
- 实测最大额外位置误差 1.966 mm，最大额外方向误差 9.994°，均相对本轮输入，而非原 experiment_004_system。
- dry_hands/15 左臂：最大位置误差 570.916 -> 3.241 mm，最大关节步长 79.377 -> 5.802°。
- clip_unclip_papers/1 右臂：最大位置误差 952.663 -> 234.601 mm，最大步长 140.134 -> 46.683°，仍有不可忽略的位置跟踪偏差。
- tie_and_untie_shoelace/8 右臂：最大步长 106.469 -> 90.657°，仍未消除 >90° 跳变。
- 191 个任务记录的 elapsed 时间之和约 1420.70 s（并行任务的墙钟耗时之和，不是总墙钟时间，也不是严格的 CPU time）；DP 全部选择约 3.85 s。这不能用于宣称整节点仍达到 147.27 FPS。

候选只保存在实验根目录 analytic_full/，汇总 analytic_full_selection.json，执行记录 analytic_full.log；未覆盖现有 IK、base、render 或 composite 输出。部分完整轨迹优化虽然能压低关节步长，却违反位姿预算，已拒绝合入。此轮新增试验尚不能证明全部残余跳变已经解决，也不应仅为改善个别片段把耗时的完整 CPU 逐帧重解设为默认行为。


## 局部约束修复已接入（2026-11-15，尚未重新运行完整 GPU node_bench）

项目修改：
- egowhale/action/sequential_ik.py：在原局部修复后，对剩余 >90° 几何跳变增加短区间 SLSQP 补救。保持该组 base 不变，读取资产的 joints、limits、TCP、periodic metadata。使用 MuJoCo 解析 Jacobian，不依赖调度、I/O、特定关节名称或新的 Python 包。
- tests/test_base_ik_seq.py：增加独立 MuJoCo 双 TCP 模型的约束/边界验收、误差额度不累加、重试次数上限、SO(3) Jacobian 差分、最后迭代失败时保留此前有效轨迹的回归测试。

参数位于 sequential_ik.py 的 _CONSTRAINED_* 常量和 _repair_severe_jumps：
- 触发阈值 90°；每组至多处理 _REPAIR_REGIONS=4 个区间。
- 窗口半径约 fps/3、2fps/3；18 FPS 时为 6、12 帧。
- 区间关节步长上限优先 min(2×原修复速度/fps, 60°)，失败后仅增加一次较宽松重试，至多 85°。不是把 85° 宣称为无视觉跳变。
- 相对 _repair_jumps 入口的冻结参考轨迹，每个有效 TCP、每一帧额外位置误差 <=2 mm、额外方向误差 <=10°。原修复流程既有的允许误差不是本次新增预算；新补救不因重复或重叠区间而累加额度。
- SLSQP 最多 120 迭代。内部留数值裕量，最终对实际导出的 float32 关节重新做 FK、限位及两端衔接验收；不以 optimizer.success 标志代替物理验收。SLSQP 使用 float64 角度 lift，输出类型保持原数据类型。
- 保留计算过程中预算内的有效候选，避免最后一个失败步覆盖它。若找不到合格且严重跳变计数改善的轨迹，则保留输入。
- 重试接受采用严重跳变计数、普通跳变计数、最大步长的字典序；因此消除大跳变可能增加较小跳变数量。

真实生产函数 CPU 重放：interval_production_replay_v4.json，输入为 node_periodic_full 已保存结果，涉及 17 条 episode / 19 组手臂。重放直接调用正式 _repair_severe_jumps，独立 MuJoCo FK 再校验全部有效手帧，检查输出有限值和模型限位。
- >90°：23 -> 0。
- 这些触发组中的 >30°：172 -> 190；最大残余步长 85.000000°。
- 最大额外位置误差 1.990025 mm，最大额外方向误差 9.942712°。
- 局部求解耗时累计 24.813 s。它不是完整节点额外墙钟时间，更不能据此直接声称新增版本总体 FPS。
- 输出独立保存在 interval_production_replay_v4/，没有覆盖既有实验。
- interval_final_tests.log：29 passed。Ruff 与 git diff --check 通过；源校验值见 interval_production_source_hashes.json。

限制：这只是现有结果上的生产函数重放。当前所有 GPU 均有其他计算任务，因此尚未从 cuRobo 求解入口重新跑 node_bench；147.27 FPS 属于新增补救前的完整测试。实际连续性、全节点吞吐以及此前位置误差尾部退化都仍须验证，不将本轮 >90° 为零当作整个目标已经完成。


## 软件渲染与更严格步长复核（2026-11-15）

使用 Mesa llvmpipe 软件渲染，无 GPU 推理。复用生产 render/composite，对 tie_and_untie_shoelace/8、knead_slime/6、clip_unclip_papers/1 原最大跳变附近各 26 帧生成左右对照，左为局部修复前、右为局部修复后。文件位于实验目录 visual_review_interval_cpu/，包括 MP4 和四帧对照 PNG。已人工检查三张 PNG：
- knead_slime/6 的明显翻转减轻，但仍有手臂姿态变化。
- tie_and_untie_shoelace/8 的相邻帧仍可见快速姿态变化，不能把 >90° 归零当作视觉连续性完全通过。
- clip_unclip_papers/1 仍有机械臂遮挡大面积画面和局部位置异常；局部修复没有解决该案例的整体质量问题。

进一步对上述三个案例尝试将区间步长上限压至 30°，半径 6/12/18 帧，两种初值，共 18 个 CPU 试验，保持同一冻结参考的 2 mm/10° 预算。结果保存在 interval_tighten.json/log。共有 3 个试验通过：knead_slime/6 的半径 6 和 12 帧、沿用当前轨迹初值两次试验使该组 >30° 数量 9 -> 8，但整条轨迹最大步长仍为 72.204°；clip_unclip_papers/1 的半径 18 帧、线性插值初值试验使 >30° 数量 22 -> 11，最大步长 85.0 -> 48.491°。其余 15 次未同时通过位姿预算和边界步长约束。未合入这些输出或增加生产默认重试次数。这是局部求解器未找到合格解的证据，不是证明所有基座/初值下不存在连续解。

正式代码校验值与上一轮 interval_production_source_hashes.json 一致，本次未继续增加未经 GPU 回归的算法修改。

后续验收仍需要可用 GPU：
1. 捕获当前 cuRobo 初始轨迹及 refinement 前后位姿误差，区分原始解分支问题与 refinement 导致的位置退化。现有最终输出不足以还原这些中间状态。
2. 在真实节点入口上验证后续修复与完整流式多 episode 求解、CUDA graph、写入的组合，测完整 overall FPS >=80。
3. 对同一成功 episode/帧集合复核位置/方向尾部及渲染视频，不能通过删除困难 episode、改时间基准或仅统计 >90° 来宣布通过。

GPU 0–9 当前均有其他计算进程，0% utilization 的卡也有显存占用和活跃计算进程，未视为空闲。没有在这些卡上追加任务或终止他人进程。


## 当前阻塞记录

2026-11-15T07:33:08.930150+00:00：再次通过 nvidia-smi 确认物理 GPU 0–9 均有计算进程；没有可用空卡。此前连续跟进也遇到相同资源阻塞。CPU 重放、约束测试和软件渲染已完成，正式源码与已验证哈希一致，未遗留上述离线检查进程。具体快照和恢复步骤保存在实验目录 GPU_BLOCKER.json。下一步需要真实 cuRobo 中间轨迹和完整 node_bench GPU 回归；现有最终输出不能还原 refinement 前输入。目标尚未完成，等待可用 GPU 后继续，不修改其他任务或共享服务器配置。


## GPU 9 恢复测试与全量核验（2026-11-15，本次已完成）

用户明确授权在物理 GPU 9 上与现有 deploy.py 共用 GPU。本次没有等待空卡，没有终止、修改或优先级调整其他用户任务。此次恢复阶段没有修改生产算法，使用已接入局部约束修复的版本；额外实验只在实验输出目录独立运行。之前“所有 GPU 被占用而阻塞”的记录已被这次授权和实际完成的测试取代。

完整 node_bench：node_interval_gpu9，物理 GPU 9，两个 worker，每 worker batch_size=5184，base_ik_seq，PiPer，原固定500输入。
- 退出码0；475成功、25几何拒绝、0失败。全部500条状态与 node_periodic_full 相同。
- 成功117560帧；698.697226 s；overall_fps=168.255999。两个worker累计服务时间对应node_fps=87.022275，不是总体并行吞吐。
- 共享服务器、不同GPU及跨时段负载使147.27->168.26不能解读为局部修复带来的确定加速。也没有据此声称整条pipeline达到168FPS。
- 正式代码/配置哈希在运行期间记录并在结束后复核未变化。记录node_interval_gpu9_source_hashes.json。
- 独立MuJoCo FK核对475条、117560帧、235120有效手帧；有限值、资产限位、关节顺序、与基线帧数检查通过。

| 指标 | experiment_004_system | 上轮node_periodic_full | 本次node_interval_gpu9 |
|---|---:|---:|---:|
| 几何步长>30度帧边 | 6121 | 1191 | 1236 |
| 几何步长>90度帧边 | 1038 | 23 | 0 |
| 存在>90度的episode | 297 | 17 | 0 |
| 位置误差中位数/mm | 1.909 | 1.782 | 1.791 |
| 位置误差P95/mm | 46.065 | 57.950 | 56.849 |
| 位置误差P99/mm | 199.828 | 221.492 | 215.508 |
| 位置误差最大/mm | 661.019 | 952.663 | 585.779 |
| 方向误差中位数/度 | 32.499 | 9.296 | 9.526 |
| 方向误差P95/度 | 155.616 | 52.006 | 52.820 |

本次最大几何步长89.19146度；245条episode仍有>30度的帧边。未宣称视觉跳变全部消失。相对最初基线，33273个手帧位置误差增加超过2mm，2809个手帧方向误差增加超过10度。局部修复的2mm/10度预算只约束该修复入口参考，不是整个优化链相对旧系统的保证。不同完整运行有流式批次/数值差异，不能把相邻两列差值都归因于新增修复。

中间状态捕获：capture_resume_gpu9，四个问题episode、8组手臂，保存seed/refinement/repair阶段。cuRobo选中候选的平均位置误差与独立MuJoCo FK相差均小于0.00002mm，确认大误差不是两个模型坐标系不一致造成的统计假象。
- 部分初始IK已有明显跟踪偏差；refinement又可能在降低整段平均目标的同时恶化少数帧。
- dry_hands/15左臂第150帧：seed位置误差131.54mm，refinement623.34mm，距seed大跳变1帧。
- tie_and_untie_shoelace/8右臂第103帧：8.14->77.58mm，距seed大跳变1帧。
- knead_slime/6左臂第96帧：290.06->412.84mm，距seed大跳变1帧。
这些是坏分支附近平滑/优化导致位置退化的证据线索，不是所有误差的单一因果证明。四片段捕获的batch与完整500运行不同，其数值不可混入全量表。

单独关闭LM seed的对照：capture_no_lm_gpu9，四条均运行完成，但最终位置P95全部变差，部分达到数百mm，并重新出现>90度跳变。未合入正式代码。此试验说明不能直接移除当前LM阶段，不等同于测试或否定cuRobo0.9.0的LM步接受修复。未升级第三方依赖或驱动。

可视化：visual_review_gpu9_cpu，四个episode的26帧局部左右对照MP4/四帧PNG。左node_periodic_full，右node_interval_gpu9，原18FPS；生产render/composite以Mesa llvmpipe软件渲染，不占GPU。已检查四张连续帧图：
- knead_slime/6、tie_and_untie_shoelace/8的局部姿态变化减轻；
- dry_hands/15所选窗口差异有限，仍有明显姿态变化；
- clip_unclip_papers/1仍有机械臂大面积近镜头遮挡，基座筛选不等于整个机械臂遮挡已解决。
仅为局部抽查，不代表已逐帧人工看完475条视频。

验证：CPU test_base_ik_seq.py + test_refinement_fk.py为25 passed/6 GPU skipped；随后在GPU9实际补跑test_refinement_fk.py为7 passed。所有本次测试进程正常退出，未留下长期GPU测试任务。

下一步仍应优先处理分支连接与refinement逐帧位姿保护，保持批量IK吞吐；不继续用更强腕部惩罚或全段平均误差掩盖局部退化。本轮速度达标，严重翻转计数显著改善，但中等跳变与末端误差尾部尚未完全解决。


## 剩余中等跳变：候选姿态范围与全局选路试验（2026-11-15，未合入生产）

本轮正式算法/配置哈希仍与 node_interval_gpu9_source_hashes.json 一致；没有改动生产推理。继续使用用户授权共享的GPU9进行小样本候选求解，其余测试为CPU离线。上一轮正式节点168.26 overall FPS不适用于这里的新候选方法。

1. 对最大跳变靠前的12个episode进行更强局部SLSQP试验（18组触发手臂）。固定原base、冻结额外2mm/10度预算，30度步长上限，半径6/12/18帧。
   - >30度帧边174->142，仅4组最大步长下降超过1度。
   - 各组任务elapsed总和1072.13秒，不是并行整体墙钟/CPU time。
   - 困难组常超过100秒且最大跳变未变。拒绝作为默认方案。
   - 记录interval30_v1/results.jsonl。

2. 固定已选base，分别测试16个精确目标IK解、实际已达到位姿的替代IK解、小幅方向扰动、以及从已达到位姿向原人手目标过渡的多个位置/方向目标。
   - 不改变81个base候选或原base评分，不增加视频帧、不改变FPS，不对输入时间轴重新采样。
   - 邻帧关节值/插值只作为同一目标帧的候选，仍按该帧原目标验收，单独效果有限。
   - 动态规划先最小化全轨迹最大几何关节步长，再比较平滑与位姿代价。只优先减少>30度计数会留下更大的孤立跳变；只做平方代价又可能重新出现>90度。
   - 初步仅检查末端预算的候选中发现资产限位越界，最严重约30.3度。因此pool_jitter_gpu9及pool_corridor_gpu9早期bottleneck文件不能作为合格输出。本轮最终数据以明确投影/限位过滤、重新FK及独立MuJoCo核验后的结果为准；从未合入这些越界轨迹。

3. 最终小样本候选池：pool_projected_gpu9。
   - 混合实际位姿/朝原目标过渡的位姿/方向微扰与邻帧候选；周期关节按资产元数据处理，非周期关节按资产限位投影；投影后重新做GPU批量FK。
   - 每帧候选必须满足相对node_interval_gpu9输入的额外位置<=2mm、方向<=10度，并在资产限位内。原轨迹始终是回退候选；这里的预算不与前一轮修复串接作为新的全pipeline保证。
   - GPU FK批量验证比逐候选CPU循环快得多；现有脚本包含候选写盘、压缩和多次独立试验，不能拿该耗时推算正式node_bench FPS。
   - compact_paths.json记录去重/筛选后的候选路径；independent_validation.json对最终float32轨迹逐帧用MuJoCo复核，全部8组通过有限值、限位、2mm/10度预算。正常无跳变的组试验中也可能消耗方向预算，若后续集成必须只触发问题区间、要求实际收益，不应替换正常轨迹。

4. 在新路径上只对最大跳变处追加一次半径12帧局部修复，冻结的参考仍为node_interval_gpu9，预算不累加。最终5个触发组再次独立MuJoCo验证通过，见interval30_after_projected/independent_validation.json：

| episode/group | 最大步长原/新 | >30度原/新 | 位置P95原/新mm |
|---|---:|---:|---:|
| insert_remove_bookshelf/11 right | 89.19 / 56.21 | 9 / 10 | 80.58 / 64.88 |
| declutter_desk/8 right | 88.95 / 36.57 | 21 / 12 | 323.69 / 288.45 |
| assemble_disassemble_tiles/3 right | 85.81 / 48.59 | 12 / 14 | 52.10 / 46.40 |
| clip_unclip_papers/1 left | 84.81 / 59.62 | 16 / 10 | 3.26 / 2.48 |
| clip_unclip_papers/1 right | 48.24 / 48.02 | 13 / 12 | 218.05 / 216.43 |

最大额外位置误差1.99750mm，方向9.95726度。不能把最大步长降低等同于所有>30度计数降低，tiles/bookshelf反而增加少量中等步长。也不能把相对误差预算当成绝对末端准确：declutter和clip右臂仍有很大跟踪偏差。

对候选选路阶段的最大跳变帧检查，原人手目标每帧仅移动约0.93–15.17mm、旋转0.67–4.87度，不支持用原输入巨大单帧运动解释这些残余大关节步长。

5. 视觉抽查：visual_review_branch_cpu中的四个26帧局部左右对照，左current node_interval_gpu9，右新候选。使用生产render/composite + Mesa llvmpipe，保持18FPS。已检查declutter、tiles连续帧图：declutter右臂的关节翻转减轻，但夹爪仍离目标较远；tiles外观更连续但仍有明显姿态变化。未宣称全部500视频质量通过。

下一步：将候选生成/预算筛选集中到GPU，局部触发并限制候选量，优先验证这种候选覆盖改善是否在真实节点入口保留>=80FPS；新增逻辑须独立于scheduler/I/O并读取资产关节/TCP/限位。现有大范围候选脚本仅是数值验证，不直接照搬为默认节点实现。目标仍未完全达到，未把生产版本替换为未测速方案。


## 局部分支搜索正式节点集成与 GPU9 复测（2026-11-15）

本轮仅编辑项目 egowhale/action/sequential_ik.py、tests/test_base_ik_seq.py 和本日志。集成前两份代码已保存于实验目录 before_branch_integration，未覆盖其他人的配置、start.sh、depth 或第三方 cuRobo。

- _BranchRepair 在现有局部修复之后触发，阈值45度，每组最多4个区间，半径约2/3秒；固定区间外接缝。
- 在当前达到的位姿与原始目标之间生成位置/方向候选，并加入9度方向微扰；63个目标 x 8个解，加原轨迹共505个候选/帧。cuRobo候选调用固定256行，尾批补齐后切除，current_state连续存储。
- 基于资产关节名、限位、周期关节、TCP与group处理；PiPer/ARX共用实现。GPU批量FK重新计算投影后的候选误差。
- 动态规划先降低区间最大几何关节步长，再比较平滑与位姿代价；至少改善5度才接受。每TCP相对进入修复之前冻结参考的位置额外2mm、方向额外10度预算，且接受前用独立MuJoCo FK检查最终数值。
- 10度是本轮试验参数，不是用户指定的精确阈值；不是绝对方向误差上限。整个pipeline相对历史实验的误差不能据此宣称增加不超过2mm/10度。
- 参数集中于 sequential_ik.py 的 _BRANCH_*、_CONSTRAINED_* 与 _REPAIR_SPEED；不改变81个base、评分/refinement权重、base微调、时间轴及节点I/O。

验证：
- CPU针对性测试27通过、2跳过；PiPer/ARX真实GPU候选尾批、重复缓存调用、限位和FK一致性2通过。
- 扩展CPU回归56通过、8跳过、1失败：tests/test_robots.py::test_arx_matches_legacy_pack_and_seed 所需 tests/fixtures/arx_adapter_golden.npz 在远端缺失。未通过重新生成golden掩盖此问题。
- Ruff与git diff --check通过。
- 同样4条输入、1 worker：关闭新增分支32.7613整体FPS，开启31.0099，4条均成功。最大几何步长：bookshelf/11 86.08->56.21度，clip/1 59.63->51.26度，tiles/3 20.80->21.16度（未触发新搜索），declutter/8 37.22->37.22度（未触发）。未触发条目的微小差异说明重新求解存在数值差异，不能把全部跨run差异归于新搜索。
- 真实500条/2 worker正在 node_branch_full_gpu9 进行，物理GPU9经用户明确授权共享，其他进程保持运行。结果应以最终summary和独立FK分析为准，不用困难4条短测推断完整吞吐。


## GPU9 局部分支集成完整测试结论

测试条件：物理GPU9共享运行，2个本任务worker，cfgs/bench8_sttn_balanced.yaml，batch_size=5184，PiPer，固定500条，18FPS输入。不终止/修改其他任务。仅base_ik节点吞吐，非整个pipeline system/fps。

## 完成状态与性能

475成功、25几何拒绝、0失败；与上一版全部500条状态及成功帧数一致。117560帧，完整墙钟731.556968秒，overall FPS=160.698353。node_fps=83.674800，其分母是worker累计服务时间，不是整体墙钟。上一版node_interval_gpu9整体168.255999，当前约降低4.49%；均超过80FPS目标，共享负载与跨时段不能当严格性能因果。

来源哈希与本次实际运行一致。475条均通过独立MuJoCo FK检查：235120有效手帧，关节顺序/帧数/finite/资产限位通过。没有声称其末端绝对误差全部合格。

## 全量质量

|指标|原experiment_004_system|上一版node_interval_gpu9|本轮node_branch_full_gpu9|
|---|---:|---:|---:|
|几何步长>30度帧边|6121|1236|1122|
|几何步长>90度帧边|1038|0|0|
|最大几何步长/度|179.97|89.19|65.17|
|含>30度帧边的episode|369|245|231|
|位置误差中位/mm|1.909|1.791|1.756|
|位置误差P95/mm|46.065|56.849|64.539|
|位置误差P99/mm|199.828|215.508|229.774|
|位置误差最大/mm|661.019|585.779|568.003|
|方向误差中位/度|32.499|9.526|9.495|
|方向误差P95/度|155.616|52.820|52.760|

周期关节几何步长按相同资产规则统计。当前raw坐标>90度为2443处，主要混有±pi表示切换，不能用它代替可见运动，也不能将此运动学输出直接当真机轨迹。

本轮184个局部区间触发，149个通过约束及收益检查；97条episode接受至少一个区间。不是所有episode都重新扩展多候选。

## 必须保留的限制

- 速度达到目标，关节最大跳变改善；仍有1122条>30度帧边和231条受影响episode，不能宣称跳变彻底解决。
- 位置P95/P99退化，不能用位置中位数改善或方向改善掩盖。
- 全量重新运行不是固定IK输入的纯选路消融。跨run472条base矩阵差值超过1e-5，包含375条本轮没有接受任何新分支的episode。worker分组/执行顺序及数值求解可能造成变化，未逐项证明来源。
- 最大位置P95退化案例 open_close_insert_remove_tupperware/22（37.19->248.92mm）、screw_unscrew_allen_fixture/6（24.22->231.93mm）、insert_remove_drawer/16（3.47->129.73mm）均未接受新分支，且base发生变化。因此不能把所有变化归于新分支，也不能据此免除最终输出退化。
- 局部额外2mm/10度约束参照该次运行进入修复前的冻结轨迹，既不是绝对误差上限，也不是相对历史run的逐帧保证。
- 后续应冻结base与修复输入做选路消融，并给refinement/选base过程补充逐帧位置质量保护的验证；不要继续盲目提高腕部惩罚或全局平滑强度。
- 目前保留为已通过速度测试的优化实验版本；尚未满足“全部跳变与末端偏差解决”的完整目标。未改depth、资产、用户start.sh或第三方库。

## 文件

summary.json / results.jsonl：速度、状态与全部输出。
validated_comparison.json：对原系统的独立FK统计。
validated_vs_interval.json：对上一版的独立FK统计。
cross_run_differences.json：逐episode基座变化、选路触发和退化案例。
audit.json：运行源码校验、状态/帧数一致性、触发计数。
../visual_review_integrated_full_cpu：四个困难片段的26帧前后合成对照，软件渲染、原18FPS，仅局部抽查，不是全部500视频验收。
项目修改日志：/export/ra/heqijun/EgoWhale/docs/ik_optimization_20261115.md。

视觉抽查补充：已查看完整run的declutter_desk/8在f387/389/390/392的合成对照。上一版右臂在f389-390明显翻转，新版这些帧姿态更稳定；但跨run基座不同，不能仅据该图归因给选路，且此抽查不构成全episode或末端精度验收。四条26帧对照均已生成并保持18FPS。

## 按用户更新后的验收标准结束
用户确认：FPS达到要求且跳变已有改善即可结束。当前overall FPS160.698353，最大几何步长及>30度计数均改善，任务已标记完成，不再追加测试。上述位置误差尾部退化及残余跳变仍保留为结果限制，不因任务结束而删除。
