# Dual-Franka shoelace task

## Task collision geometry

The dual-Franka task enables robot collisions only on `panda_leftfinger` and
`panda_rightfinger`. The G1 Inspire variant enables the thumb and index finger
links on both hands; its arms, torso, legs, palms and other fingers have collisions
disabled. On each thumb distal and index intermediate link, an invisible
16 x 10 x 3 mm box replaces the authored terminal mesh colliders. The pad faces
are calibrated parallel with a nominal 2.8 mm closed gap on both hands.
Both hands use the same finger joint targets. Their pad boxes and surface landmarks
are mirrored in wrist coordinates. The right-body offsets compensate the source
USD's asymmetric thumb joint frames and index link offsets; raw link-local numbers
are therefore not obtained by simply negating the left offsets.
These task-local USD overrides preserve authored mass, inertia and visual geometry.
G1's instanced geometry is made editable before applying the overrides.

G1 uses straight fingers by default: thumb intermediate/distal and index intermediate
joint targets are zero, their mimic constraints are removed, and their limits are
restricted to +/-0.001 degrees. Only root joints define the pinch pose; the binary
action moves each index root through 0.25 rad while holding the thumb root targets.
This changes the simulated mechanism from the coupled physical Inspire hand.
The source USD is unchanged. Pad contact and loaded-pull retention still require
validation; a parallel nominal pose alone does not establish a stable grasp.

Both tasks use the shared `shoelace_constants.FINGER_MU` (40) for the grasping
finger collision shapes. G1 hand drives use stiffness 50 N·m/rad and retain the
asset's damping 0.2 N·m·s/rad. These rotational gains cannot be compared directly
with Franka's prismatic finger gains in N/m.

The shoe, G1 pedestal, cable geometry and cable self-collision retain their existing
settings. Disabling unused robot colliders reduces the candidate pairs used to
preallocate Newton ADMM contact storage. It also means disabled robot parts can
pass through the scene, so these tasks do not train whole-arm collision avoidance.

Existing checkpoints retain compatible observations and actions, but should be
re-evaluated under the simplified contact model. To restore the asset's complete
robot collision geometry, set `spawn.collision_props = None` on both
`cfg.scene.robot_left` and `cfg.scene.robot_right`. For G1, the custom
`spawn_g1_shoelace` spawner also replaces terminal colliders and joint constraints;
restoring the coupled hand requires restoring its matching targets and TCP calibration.

The internal ADMM collision pipeline uses 1,048,575 triangle-pair and contact-reduction
slots, the largest capacity supported by Newton with `latest` contact matching.
The outer collision pipeline and the internal reduction hash table still scale
with the environment count. Monitor overflow warnings when increasing parallelism;
this fixed internal capacity is not a guarantee against overflow.

## Reward / metrics 中文速览

以下对应当前 `IsaacContrib-Shoelace-DualFranka` 的默认配置；训练命令覆盖参数后，以该次运行的配置为准。
任务主线是：**靠近鞋带尾端 → 精细对准并闭合 → 抓稳 → 双手向外拉 → 满足完成条件**。
这些奖励同时计算，并非依次切换的硬阶段。Reward 决定策略学什么，metrics 用于看清实际做到了什么。
TCP 指夹爪的控制参考点；尾端位置取对应鞋带末端 3 个自由段的中心均值。

G1 的 TCP 使用名义闭合姿态下两指面中点相对手腕的固定偏移，以左手标定为基准镜像到右手。
当前 wrist-local 偏移为左手 `(0.193129098, -0.063929696, 0.029342429)` m、
右手 `(0.193129098, 0.063929696, 0.029342429)` m。
左右手开合角度一致；双臂初始关节角按镜像轴符号生成，reset 时右臂镜像左臂的随机扰动。
这保证机器人初始姿态对称，但鞋带尾端本身不对称，左右 TCP 到尾端的距离不保证相等。
标准 IK、`tails_to_tcp` 观测和粗接近奖励都使用这个固定点，不再使用实时中点动作补偿，
也不在闭合时切换接近参考或执行 handoff。Dual-Franka 的粗接近仍使用实时 `capture_distance`。
两种机器人实际抓取、预抓取和成功判定仍依据实时指面、接触和相对速度；固定 TCP 对准不代表已抓住。
旧 G1 策略需重新训练，动作和观测维度虽不变，但 TCP 语义已变化。
对称夹垫、右手目标和镜像 reset 改变了旧 G1 checkpoint 的环境定义；复现旧结果应使用旧版本。

### 6 个 reward 分别做什么

| 配置项 | 默认权重 | 作用与给分方式 |
| --- | --- | --- |
| `dense_task` | `10.0` | 主进度奖励：靠近占 30%、获得抓握占 30%、向外拉占 40%。靠近/抓握按前后分数差给分；拉动只奖有效抓握下的新向外纪录。 |
| `pregrasp` | `1.0` | 补足“快碰到但还没抓住”的引导：15 mm 尺度的精细位置对准占 75%，距离小于 10 mm 时的实际闭合占 25%。按前后差给分，双臂平均后的势能上限为 1。 |
| `grasp_hold` | `1.0` | 奖励持续抓稳。理想双手抓握初期每秒 +1，只有一手时每秒 +0.25；每回合用完 2 秒的“质量加权保持时间”后，降到原给分速率的 20%。松开再抓不会刷新预算。 |
| `success` | `5.0` | 满足下述完整成功条件时一次性 +5，并结束回合；单纯超时不给这项奖励。 |
| `arm_action_rate` | `-0.001` | 惩罚相邻步机械臂动作之差的平方和，抑制动作突变；不包含夹爪开合，也不是除以时间后的加速度。 |
| `arm_action_magnitude` | `-0.001` | 惩罚机械臂动作本身的平方和，抑制过大指令；使用笛卡尔缩放前的动作，不包含夹爪开合。 |

`dense_task` 的设计要点：

- **靠近**：Franka 使用实时捕获线段到尾端的距离，G1 使用固定 TCP 到尾端的距离，尺度均为 80 mm；
  **抓握**：用接触、实际闭合和低滑动来评分。
  两者改善给正分，远离或丢失抓握会扣回对应势能；静止且滤波稳定后，不再给进度分。
- **鼓励合作**：靠近、抓握、拉动各自有 50%、70%、80% 的份额用于双手合作加成，其余按单手分数均值给分。
  因此一只手先做对也有收益，两只手一起做好收益更高。
- **拉动只奖新纪录**：左手朝负 X、右手朝正 X。单手拉动需该手当前原始/滤波抓握分均 ≥ 0.2，
  双手加成要求两手都满足。无抓握的移动也会更新纪录，所以事后抓住、松开重抓、反复内外摆动不能补领旧距离的奖励。
- **尺度不是终点**：每手尺度为 `max((0.18 m - 初始两尾 X 间距) / 2, 0.01 m)`。
  理想双手各拉满一个尺度约得 +4，超过尺度继续创新纪录仍有奖励；最终成功另行判定。

这里的抓握质量 `G ∈ [0, 1]` 是“两指接触 + 实际闭合 + 尾端相对 TCP 低滑速”的组合，采用 0.10 s 滤波。
默认接触宽度 0.5 mm，容忍最多 1 mm 的接触求解穿透，滑速尺度为 0.08 m/s。
它是接触代理分数，不是力传感器测出的抓持力；仅发出闭合指令不会自动算抓住。
`pregrasp` 则只是位置引导：近处空夹也可能拿到少量进度分，但不等于物理抓握或成功。
`dense_task` 和 `pregrasp` 在 reset 后的第一个有效样本只建立基线，不给初始化进度分。

### 怎样才算成功

成功要求**协作拉动历史达标，同时当前几何达标，且相关状态数值有效**：

1. **确实双手拉过**：两条尾端在双手有效抓握时，各累计至少 **25 mm** 的新向外纪录。
   计入一次移动要求该步区间前后两端，两手的原始和滤波抓握分都 ≥ 0.2；这个要求比 dense 拉动奖励更严格。
2. **两尾到位**：相对固定锚点中点，左尾 X ≤ **−90 mm**，右尾 X ≥ **+90 mm**。
   这是最终位置要求，不是相对 reset 的位移；总间距达到 180 mm 仍不能代替两侧各自到位。
3. **结喉区域清空到阈值**：以锚点中点为球心、半径 **25 mm** 的区域内，每侧鞋带的自由段中心数均 **≤ 15**。
   固定锚点不计数；这是局部几何判据，并非严格的拓扑解结判定。

协作拉动历史达标后会保留到 reset，因此最终几何达标时允许已经松手。
默认回合最长 **10 s**；提前成功就结束，超时也结束。

### Metrics 看什么

下表名称均省略 TensorBoard 前缀 `Metrics/shoelace/`，`left/right` 按机械臂侧命名。

| 指标 | 含义 / 怎么读 |
| --- | --- |
| `approach_distance_m` | 双手粗接近距离的均值 [m]：Franka 使用捕获线段，G1 使用固定 TCP。 |
| `grasp_left`, `grasp_right` | 每侧滤波后的抓握质量 [0, 1]；看哪只手没抓稳。 |
| `grasp_both` | 每个环境内两手质量的 soft-AND，再取平均；看是否在同一环境里同时抓住，不能由前两条均值直接推出来。 |
| `pull_left_displacement_m`, `pull_right_displacement_m` | 相对 reset 后首个有效样本的向外位移 [m]；正为向外、负为向内。被动移动也计入，不证明是抓着拉的。 |
| `pull_x_separation_m` | 两尾当前 X 方向的绝对间距 [m]；看整体展开程度，单独不能判成功。 |
| `pull_left_score`, `pull_right_score` | 旧版“抓握门控的拉动进度”诊断分数 [0, 1]。静止理想抓握也可约为 0.5；不是距离，也不是当前新纪录拉动奖励。 |
| `loaded_pull_left_m`, `loaded_pull_right_m` | 在双手有效抓握区间内累计的新向外纪录 [m]；成功要求每个环境的两侧各 ≥ 0.025 m，松手后累计值保留。 |
| `bilateral_pull_completed` | 当前已有两侧 loaded pull 均达标的环境比例；只代表协作拉动阶段完成。 |
| `throat_left_segments`, `throat_right_segments` | 每侧仍在结喉球内的自由段中心数均值；实际成功逐环境检查两侧是否各 ≤ 15。 |
| `geometry_success` | 当前满足“两尾到位 + 每侧结喉计数达标 + 鞋带状态数值有效”的环境比例；不检查协作拉动历史。 |
| `success_rate` | 每个已结束过回合的环境，取其最近一次回合的完整成功结果，再求平均；不是全部历史回合的累计成功率，首次完成前为 0。 |
| `valid_fraction` | dense reward 所检查输入无 NaN/Inf 的环境比例，正常为 1；不是有效抓握率。 |

**读日志时的几个区别：**

- `Episode_Reward/<上述6项之一>` 记录该项**已加权回合奖励和 ÷ 配置回合时长**，默认除以 10 s，提前结束也一样。
  因此成功回合的 `Episode_Reward/success` 是 0.5，实际成功事件仍给 +5；日志再平均后取决于成功比例。
  RewardManager 对每项乘 `weight × step_dt`；进度项和 success 在函数内除过 `step_dt`，保持奖励和动作惩罚按时间累计。
- Metrics 先在环境维度取均值，再由 RSL-RL 对 rollout 内的各步取均值；位移不是回合最大位移，不同环境的正负位移可能相互抵消。
  dense 阶段指标排除无效样本，成功项的 6 个诊断指标对全部环境取均值；当前多 GPU 日志来自 rank 0 本地环境。
- `dense_task.weight=0` 会停掉该项产生的 metrics，包括 `success_rate`；
  `geometry_success`、`throat_*_segments`、`loaded_pull_*_m` 和 `bilateral_pull_completed` 由成功终止项产生，仍会记录。

看训练时可依次排查：**距离有没有下降 → 两手是否同时抓稳 → 两侧 loaded pull 是否增长 → 几何是否达标 → success rate 是否提高**。
位移涨但 loaded pull 不涨，应检查是否被动移动、只有单手抓住，或抓握没有覆盖整个计分区间；
`geometry_success` 高但成功率低，则需要同时核对协作拉动历史，并注意两者分别是当前状态与最近完成回合的统计。

代码入口：[默认配置](shoelace_env_cfg.py) · [奖励与阶段指标](mdp/rewards.py) ·
[成功判定与协作指标](mdp/terminations.py) · [几何量定义](mdp/utils.py)。下面保留详细英文说明与迁移说明。

## Newton solver / coupler 配置支持与 PR 跟踪

核查日期：**2026-09-18**。官方 `develop` 为 `93d7bef70`，在审 PR #6733 的 head 为 `d6c3284c6`。
以下是该次核查的快照；两项在该 develop 版本中均尚未合入。Shoelace 使用 MJWarp 模拟机器人和鞋、
VBD 模拟鞋带、ADMM 处理跨 solver 接触。当前任务分支已有所需设置，不需要等通用接口 PR 合入才能运行。

| Feature | 为什么 shoelace 需要 | 通用接口需要改动的文件 | 与已有 PR 的重合 | 本分支对应 commit ID |
| --- | --- | --- | --- | --- |
| **ADMM 内部碰撞容量配置**：`contact_max_triangle_pairs`、`contact_reduction_hashtable_size_factor` | 大量并行环境产生密集跨 solver 接触，需要调整三角形对缓存和接触归并哈希表容量；外层 collision cfg 无法控制内部容量。主要解决扩容问题，小规模运行未必需要调整。 | [coupler_cfg.py](../../../../isaaclab_contrib/isaaclab_contrib/coupling/coupler_cfg.py)：声明字段；[coupler.py](../../../../isaaclab_contrib/isaaclab_contrib/coupling/coupler.py)：校验并应用配置；[test_coupler.py](../../../../isaaclab_contrib/test/coupling/test_coupler.py)、[test_coupler_runtime.py](../../../../isaaclab_contrib/test/coupling/test_coupler_runtime.py)：验证参数和实际管线容量。**这些已实现。** | 未检索到直接重合的在审 PR，可独立提交。当前适配仍通过重建 Newton 私有内部碰撞管线应用容量设置。 | `cd7b2088ac567f5f2e85428137673375adfd99ae`（简称 `cd7b2088a`），**Scale ADMM contact capacity for Shoelace**。 |
| **通用 VBD 配置暴露**：`rigid_avbd_alpha`、`rigid_contact_hard`、`rigid_body_contact_buffer_size` | 调节鞋带链关节/接触的误差修正、接触模式和每刚体接触容量，替代当前任务的自定义 cfg 子类。属于配置能力补齐；尚未证明三个任务取值都不可缺少。 | [vbd_manager_cfg.py](../../../../isaaclab_newton/isaaclab_newton/physics/vbd_manager_cfg.py)：补字段；[test_vbd_core.py](../../../../isaaclab_newton/test/physics/test_vbd_core.py)：补参数传递和默认行为验证；任务接入时更新 [shoelace_env_cfg.py](shoelace_env_cfg.py)，改用通用配置。现有 manager 已支持按名称传参。**通用接口尚未实现。** | **[PR #6733 — Expose VBD rigid contact controls](https://github.com/isaac-sim/IsaacLab/pull/6733) 部分覆盖**：包含 hard-contact 开关和每刚体接触容量；提供的是 `rigid_avbd_contact_alpha`，没有全局 `rigid_avbd_alpha` 或 `rigid_compliant_alm`。需适配 VBD 已迁入 `isaaclab_newton` 的目录结构。 | `0424ed9a8bb189fb40ccda1632c01f628dd21fdc`（简称 `0424ed9a8`），**Add dual-Franka shoelace RL task**。该提交引入 `ShoelaceVBDSolverCfg` 的任务内设置，**并非通用 VBD 配置暴露已完成的提交**。 |

两个通用 PR 都应更新 [using-vbd-solver.rst](../../../../../docs/source/overview/core-concepts/physical-backends/newton/using-vbd-solver.rst)，
并分别添加 `isaaclab_contrib` / `isaaclab_newton` 的 changelog fragment；ADMM 文档和 fragment 已包含在 `cd7b2088a` 中。
ADMM 随环境数扩容的规则继续放在任务的 [shoelace_env.py](shoelace_env.py)，由
[test_shoelace_physics.py](../../../test/contrib/test_shoelace_physics.py) 验证。
`cd7b2088a` 还包含任务接入和训练默认值变更，拆通用 PR 时应按文件选取。

VBD 当前任务值为 `rigid_avbd_alpha=0.0`、`rigid_contact_hard=True`、`rigid_body_contact_buffer_size=256`。
其中 `rigid_contact_hard=True` 已是当前 Newton 默认值；Newton 1.6 已弃用该旧接触模式开关，
需协调 `rigid_compliant_alm` 的配置与迁移验证，参见 [Newton SolverVBD 文档](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverVBD.html)。
建议 ADMM 容量支持独立提交，VBD 与 #6733 协调补齐，并保留通用配置的既有默认行为。

## Detailed design and migration notes

For multi-GPU training, `--num_envs` is the number of environments on each GPU.
RSL-RL collects 8 steps per environment before each PPO update. With 8 GPUs and
1024 environments per GPU, each iteration collects 65,536 transitions in total.
Override the rollout length with, for example, `agent.num_steps_per_env=32`.

The PPO actor learns its Gaussian exploration standard deviation in log space (`std_type="log"`), with
initial standard deviation 1.0. This changes the exploration parameterization, not the action interface or
deterministic mean-action evaluation. When loading a checkpoint trained with scalar standard deviation,
use its saved agent configuration or set `agent.actor.distribution_cfg.std_type=scalar`; scalar and log
parameterizations use different checkpoint parameter names.

The outer Newton collision pipeline reserves at least `max(1_000_000, 8192 * num_envs)`
triangle pairs per process. The internal ADMM pipeline keeps 1,000,000 triangle pairs
and scales its contact-reduction hashtable factor to reserve at least 2048 slots per
environment. At 1024 environments per GPU, the factor is 2.097152 and the hashtable
has 2,097,152 slots. Larger explicit budgets and factors are preserved.
Override the internal budget with `env.sim.physics.solver_cfg.contact_max_triangle_pairs`
and its table factor with `env.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor`.
These affect memory allocation at startup; they do not change the solver timestep or contact material.
Contact matching remains enabled. Newton 1.6 requires the internal triangle-pair
budget to stay below `2**20` in this mode; increase the hashtable factor for table warnings.

`IsaacContrib-Shoelace-DualFranka` exposes the standalone Newton shoelace scene as a manager-based RL
environment. The initial interface intentionally includes only dual-arm Cartesian actions, binary gripper
actions, proprioceptive and tail observations, compact finger-tail signed-distance history, randomized resets,
dense task and pregrasp progress rewards, grasp retention and success rewards, small arm action penalties,
and episode timeouts.
The compact contact-related policy
input contains:

- four finger-tail signed surface distances, clipped to +/-2 mm and stacked over the current and previous two
  policy steps (12 values, oldest to newest); positive means separation, zero means touching, and negative means
  penetration;
- two positive actual-minus-target gripper closure residuals; and
- two free-tail speed magnitudes relative to the controlling TCPs.

The signed-distance history is derived directly from Newton collision candidates because coupled solvers do not
expose the standard contact-force sensor. A pair without a collision candidate uses the positive 2 mm cap.

The dense reward covers approaching, grasping, and pulling the tails, with independent credit and a bilateral
bonus in each phase. Let `H(a, b) = a * b / (a + b - a * b + 1e-6)` be the Hamacher soft-AND,
`A` the per-arm approach proximity (capture segment for Franka, fixed TCP for G1),
`G` the filtered per-arm grasp quality, and `d` each tail's
outward X displacement from the first valid post-reset sample. The left arm pulls toward negative X, and the
right arm toward positive X. Positions are relative to the fixed cable seam midpoint, so translating the shoe
or the whole environment does not create progress. The default is:

```python
A = 1 - tanh(approach_distance / 0.08)  # metres; capture segment for Franka, fixed TCP for G1
approach = 0.5 * A.mean(dim=-1) + 0.5 * H(A[:, 0], A[:, 1])
grasp = 0.3 * G.mean(dim=-1) + 0.7 * H(G[:, 0], G[:, 1])
acquisition_potential = 0.3 * approach + 0.3 * grasp
scale = max((0.18 - initial_x_separation) / 2, 0.01)  # per environment, metres
P = maximum(d / scale, 0)
B = minimum(P, 1)  # Hamacher inputs stay in [0, 1].
bilateral = H(B[:, 0], B[:, 1]) + maximum(P.min(dim=-1).values - 1, 0)
records = concatenate((P, bilateral[:, None]), dim=1)
new_records = maximum(records - best_records, 0)
eligible = (raw_grasp >= 0.2) & (G >= 0.2)
pull_increment = 0.2 * (new_records[:, :2] * eligible).mean(dim=-1)
pull_increment += 0.8 * new_records[:, 2] * eligible.all(dim=-1)
best_records = maximum(best_records, records)  # also advance without grasps
reward_rate = (acquisition_potential - previous_acquisition_potential + 0.4 * pull_increment) / step_dt
```

`acquisition_weight=0.6` and `approach_fraction=0.5` set progress weights of 30% approach, 30% grasp, and
40% pull. The manager multiplies the returned rate by `step_dt` and the overall weight (10), giving maximum
acquisition gains of 3 for approach and 3 for grasp. Pull pays approximately 4 for ideal bilateral progress
from zero to one scale, then 4 per additional common scale of new outward progress; its total is not capped.
`bilateral_approach_fraction=0.5` and `bilateral_grasp_fraction=0.7` reserve
half the approach budget and 70% of grasp acquisition for cooperation. Independent credit starts either arm;
bringing the other arm into the same phase earns a larger increment. With ideal scores, one approached arm
earns 0.75 of the approach budget and approaching the other adds 2.25. One grasp earns 0.45 of the acquisition
grasp budget; both grasps earn 3. These are potential gains, not rewards paid on every stationary step.

Grasp credit remains independent of TCP proximity. Each grasp requires both finger surfaces near contact,
actual gripper closure, and low tail-TCP slip. Contact scoring tolerates up to 1 mm of negative signed distance
(`contact_penetration_tolerance=0.001`) to avoid penalizing bounded contact-solver penetration under load:

```python
error = maximum(signed_distance, 0) + maximum(-signed_distance - 0.001, 0)
finger_contact = exp(-(error / 0.0005)**2)
```

Positive gaps and penetration beyond the tolerance still reduce the score. The tolerance is a task-local
calibration for the current coupled solver, not proof of force closure or permission to accept arbitrary
penetration. Changing the cable geometry or solver requires checking it again. Empty closure, a missing
finger contact, and high relative slip remain insufficient. The 0.10 s grasp filter smooths contact flicker;
pull eligibility additionally checks current raw quality so a stale filtered grasp cannot pay for release motion.

`bilateral_pull_fraction=0.8` reserves 80% of pulling for cooperation and leaves 20% available independently.
With `pull_use_high_water_mark=True`, pull starts at zero: acquisition alone earns no pull credit. Only a
new physical record with sufficient grasp quality earns an increment; `pull_grasp_threshold=0.2` controls
eligibility. The bilateral record uses both current physical progresses, not independent historical maxima,
and requires both current grasps. Records advance even during ungrasped motion, preventing retrospective
payment on regrasp. They never reset on release or inward motion. Repeated inward/outward cycles cannot
earn the same pull credit twice; progress below the reset baseline pays nothing. The separation scale is
not a stopping threshold: new records beyond it continue paying until termination. The original Hamacher
shaping is preserved up to that scale; beyond it, the bilateral extension follows the smaller current
per-arm progress, so pulling only the leading arm farther earns independent credit, not an extra bilateral
bonus. This removes premature reward saturation without changing success, grasp gates, or the scale.
Each bilateral fraction can be set to zero to use only the corresponding per-arm mean.

Approach and grasp retain their signed potential differences, including penalties for losing a grasp or
moving away. Physical pull credit is never clawed back merely because contact quality drops. Stationary
states pay no dense reward once the filter settles. Invalid samples preserve history, and the first valid
sample after reset seeds without reward. New-record pull can pay on the first physical cycle but not on
subsequent identical cycles. No discounted optimal-policy invariance is claimed.

The independent `pregrasp` progress term bridges coarse approach and the first physical grasp. It uses the same
three-segment tail centers and TCP positions as the policy's `tails_to_tcp` observation:

```python
alignment = exp(-(tail_tcp_distance / 0.015)**2)
closure_gate = maximum(1 - (tail_tcp_distance / 0.01)**2, 0)**2
closure = clip((0.01 - actual_finger_position) / (0.01 - 0.001), 0, 1)
pregrasp_potential = (0.75 * alignment + 0.25 * closure_gate * closure).mean(dim=-1)
pregrasp_reward = pregrasp_potential - previous_pregrasp_potential  # manager weight = 1
```

The 15 mm Gaussian width adds fine positional guidance inside the broader 80 mm approach scale. The closure
gate is exactly zero at and beyond 10 mm and rises smoothly toward the TCP, so closing far from the tail adds
no credit. Both terms reward each arm independently: ideal alignment contributes at most 0.375 per arm, and
nearby actual closure adds at most 0.125 per arm. Their combined maximum potential is 1 across both arms,
smaller than the existing physical grasp and pull budgets. A close command without actual joint motion does
not change the closure score. The gate does not shrink as the gripper closes.

This is a TCP-centered positional heuristic, not a test of tail orientation, collision-free capture, or physical
grasping. Empty closure inside the gate can earn a small, bounded progress increment; it earns no repeated
pregrasp reward for holding still and does not count as contact-based grasp, retention, or success. Opening or
moving away removes the corresponding potential. Closed state cycles have zero undiscounted pregrasp return;
discounted optimal-policy invariance is not claimed. Invalid distances or finger positions return zero without
advancing history; the first valid sample after a per-environment reset seeds history without reset credit.

The two components can be ablated without rescaling the other: set
`env.rewards.pregrasp.params.closure_weight=0` for fine positional guidance alone, or set `alignment_weight=0`
for gated closure alone. Set `env.rewards.pregrasp.weight=0` to recover the previous reward objective with all
other terms unchanged. The default combines both components as an initial shaping candidate, not a trained
result. Compare near-tail close-command frequency, contact-based grasp quality, and continuous retention
duration; change one component at a time in follow-up experiments. Observation, action, and PPO interfaces
are unchanged, so checkpoints remain loadable but need re-evaluation under the new objective.

The independent `grasp_hold` term rewards time spent retaining physical grasps:

```python
hold = 0.5 * G.mean(dim=-1) + 0.5 * H(G[:, 0], G[:, 1])
full_rate_time = minimum(step_dt * hold, maximum(2.0 - full_rate_time_used, 0))
hold_reward = 1.0 * (0.2 * step_dt * hold + 0.8 * full_rate_time)
full_rate_time_used += full_rate_time
```

Its weight is a maximum reward rate of 1.0 per second: initially, ideal bilateral retention pays 1.0 per second
and one-sided retention pays 0.25. `full_reward_duration=2.0` limits full-rate **quality-weighted** time per
episode, not wall-clock time since reset or first contact. Afterwards, `sustained_reward_fraction=0.2`
preserves a smaller maintenance incentive. Release and regrasp do not renew the budget. A step crossing the
budget boundary is split exactly, independent of policy timestep. Empty closure earns approximately zero.
It owns its filter and budget, so disabling or reordering `dense_task` does not change retention evaluation.
Invalid contact, closure, or slip inputs earn zero and preserve filter/budget state; reset clears only selected
environments. When tuning contact or filter parameters, update both `dense_task.params` and
`grasp_hold.params` to keep their grasp definitions aligned.
Set `env.rewards.grasp_hold.weight=0` for a progress-only ablation.
Set `env.rewards.grasp_hold.params.full_reward_duration=None` to disable the time budget for an ablation.

The `success` reward pays +5 on the cooperative-pull-and-geometry success termination, including success on the timeout
step. Timeout alone pays no success reward. `mdp.shoelace_success_reward` divides the success flag by
`step_dt`, so the manager's timestep multiplication leaves +5 per event. The environment then resets.
Ideal bilateral retention throughout the current 10-second episode contributes at most 3.6 instead of 10;
ideal single-arm retention contributes at most 2.1 instead of 2.5. Completion earns a separate event reward.
These are acquisition and retention tuning budgets, not a guarantee that training will learn to pull.
Evaluate simultaneous approach, continuous grasp duration, gripper switching, outward displacement, and
cooperative task success and geometry-only completion separately when comparing runs.

**Migration:** The observation/action interface is unchanged, but the reward objective and defaults changed.
Existing checkpoints remain loadable; re-evaluate them and use a separate training run because return values
and grasp-score curves are not directly comparable. High-water pull no longer clips records at one scale;
old capped-pull checkpoints need further training to learn from the extended reward. To reproduce the capped
objective, use its original code revision; saved configuration alone does not restore that formula.
To reproduce the pre-loaded-pull objective, set
`contact_penetration_tolerance=0.0` in both dense and hold terms, `dense_task.params.pull_use_high_water_mark=False`,
and `grasp_hold.params.full_reward_duration=None`. These are also independent ablation switches; change one
step at a time when attributing training improvements. Old saved parameter dictionaries retain legacy behavior
because the new callable options default to zero penetration tolerance, legacy pull, and no hold budget.

For migration from older acquisition formulas:
`approach_fraction` now linearly divides acquisition between the cooperative approach and grasp scores;
the previous `H(A, approach_fraction)` saturation was removed. Existing parameter names remain accepted,
but restoring their old values does not reproduce the old reward formula. Re-evaluate or retrain policies,
configure the new bilateral fractions explicitly for custom tasks, and disable `grasp_hold` and `success`
independently for ablations. Historical reward curves are not directly comparable.

Two independent penalties regularize the 12 normalized arm commands before Cartesian scaling. Both select the
`left_arm` and `right_arm` action terms by name and exclude binary gripper commands:

- `arm_action_rate` has weight `-0.001` and sums squared changes from the previous policy step to discourage
  jitter and frequent direction changes. This is a step difference, without division by `step_dt`.
- `arm_action_magnitude` has weight `-0.001` and sums squared commands to discourage unnecessary motion,
  including constant commands that incur no action-change cost.

The reward manager multiplies both penalties by `step_dt`, so the total per-step reward is:

```python
reward = 10.0 * (acquisition_potential - previous_acquisition_potential + 0.4 * pull_increment)
reward += pregrasp_reward + hold_reward + 5.0 * success - step_dt * (
    0.001 * (arm_action - previous_arm_action).square().sum(dim=-1)
    + 0.001 * arm_action.square().sum(dim=-1)
)
```

Both progress terms are zero on their first valid samples as described above; other reward terms still apply.
Action history resets to zero per environment, so the first command is compared with zero. The total reward
therefore includes ongoing maintenance rewards as well as new-record progress credit. These small initial weights have
not been tuned through training. Adjust them independently with `env.rewards.arm_action_rate.weight` and
`env.rewards.arm_action_magnitude.weight`.

The dense and success terms write phase metrics to `extras["log"]` on every policy step. RSL-RL prints their rollout
averages in each training iteration and writes the same tags to TensorBoard. Prefer physical displacement
to the grasp-gated score when checking whether a tail actually moved outward:

| Tag under `Metrics/shoelace/` | Interpretation |
| --- | --- |
| `approach_distance_m` | Mean coarse approach distance across both arms [m]: capture segment for Franka, fixed TCP for G1. |
| `grasp_left`, `grasp_right` | Filtered contact, closure, and low-slip grasp quality for each arm in [0, 1]. |
| `grasp_both` | Hamacher soft-AND of both filtered grasp qualities in [0, 1]. |
| `pull_x_separation_m` | Absolute two-tail X separation [m]; successful per-tail X boundaries imply at least 0.18 m, but this total alone is insufficient. |
| `pull_left_displacement_m`, `pull_right_displacement_m` | Signed outward tail displacement from the episode baseline [m], independent of grasp quality. Positive means outward; negative means inward. |
| `pull_left_score`, `pull_right_score` | Legacy `H(G, 0.5*(1+tanh(d/scale)))` diagnostic in [0, 1]; not a distance, actual new-record reward, or success rate. |
| `success_rate` | Mean success over each environment's most recent completed episode. |
| `valid_fraction` | Fraction of environments whose checked reward inputs contain no NaN/Inf; normally 1.0. |
| `geometry_success` | Current fraction satisfying the geometry-only criterion, regardless of grasp or pulling history. |
| `throat_left_segments`, `throat_right_segments` | Current mean free-capsule center counts in the throat for the cable controlled by each arm; each must be at most 15 at success. |
| `bilateral_pull_completed` | Current fraction whose two tails have each earned at least 0.025 m of loaded outward records this episode; retained after release. |
| `loaded_pull_left_m`, `loaded_pull_right_m` | Mean accumulated new outward record distances [m] earned with valid bilateral grasps; not signed net displacement or a reward. |

Each displacement uses the mean position of the corresponding tail's three free-end segments relative
to the fixed seam midpoint. Left-arm outward motion is negative X; right-arm outward motion is positive X.
The baseline is the first finite reward sample after that environment resets, not the first grasp. Thus
`0.02` means 2 cm farther outward than the baseline, `-0.01` means 1 cm inward, and acquiring a stationary
grasp leaves displacement at zero. Releasing and regrasping do not rebase it. The displacement is neither
clipped nor grasp-gated: passive cable motion also counts, so it does not prove that the gripper caused
the motion. A stationary ideal grasp can still produce a legacy pull **score** near 0.5. In high-water mode,
that score is diagnostic only; the actual pull reward requires a new physical record and is zero at rest.

`finite` is the internal per-environment numerical mask. It checks tail-to-TCP distances, finger-tail signed
distances, relative speeds, closure fractions, tail X separation, and outward tail offsets. If any checked
value is NaN or infinite, that environment's dense reward is zero and its reward history is preserved.
`valid_fraction = finite.float().mean()` reports the fraction passing this check in the current step.
For example, 1023 valid environments out of 1024 gives approximately 0.9990. This checks the numerical
validity of these reward inputs; it does not measure grasp validity, task success, or physical plausibility.

Phase metrics average over valid environments before automatic reset. When no environment is valid, those
metrics are zero and `valid_fraction=0` identifies that case. `success_rate` uses the actual `success` termination
flag, excludes environments that have not finished an episode, and is zero until the first completion. These
continuous grasp scores are contact-based proxies, not discrete stage-completion labels. For example, high
`pull_x_separation_m` with low `grasp_both` indicates separated tails without strong simultaneous grasps. Metrics are
computed by the dense reward term and require its weight to remain nonzero, except for `geometry_success`,
`throat_*_segments`, `bilateral_pull_completed`, and `loaded_pull_*_m`. Those diagnostics come from the success term and
remain available when dense rewards are disabled. They average over all environments before reset;
`geometry_success` is a current-state fraction, not a completed-episode success rate. Loaded distances and
the completed cooperative phase retain their previously earned history until that environment resets.
RSL-RL then averages the per-step values over its rollout window. In the current multi-GPU runner these
custom metrics are logged from rank 0's local environments, not reduced across all GPUs. Displacement
curves are therefore mean signed displacements, not per-episode maximum distances; opposite motions
across environments can cancel in the average.

`grasp_both` averages the per-environment joint grasp quality. It cannot be recovered by combining the logged
averages of `grasp_left` and `grasp_right`: different environments may hold different single tails.

Dashboard migration: use `pull_left_displacement_m` and `pull_right_displacement_m` for physical progress,
and `pull_left_score` and `pull_right_score` only when inspecting the legacy grasp-gated diagnostic. The old
`pull_left` and `pull_right` aliases were removed; update dashboards to the corresponding `*_score` tags.
Do not interpret old curves as meters or compare score and displacement values directly. Existing event files
are not rewritten, and running training processes must restart to pick up the logging change. Reward
calculations and retained metric values are unchanged by the alias removal. The new penetration tolerance
still changes the score's grasp input relative to older reward versions; historical grasp and score curves
require matching reward parameters for comparison.

The reward manager separately logs `Episode_Reward/pregrasp`, `Episode_Reward/grasp_hold`, `Episode_Reward/success`,
`Episode_Reward/arm_action_rate`, and `Episode_Reward/arm_action_magnitude` when episodes reset.
These are episode sums divided by the configured episode duration, not raw event rewards.

Success now uses `mdp.shoelace_bilateral_pull_success`. Its geometric component requires all of the following:

- no more than 15 free capsule centers from **each** arm's cable inside a sphere of radius 0.025 m around
  the fixed seam midpoint;
- the left arm's tail at least 0.09 m toward negative X from the midpoint and the right arm's tail at least
  0.09 m toward positive X; and
- finite cable segment positions and velocities.

The throat region follows `reb/newton_shoelace_demo`. The midpoint is the average of the
left cable's last segment and the right cable's first segment. These two fixed anchors are excluded from
the throat count; the remaining 156 segment centers are counted, 78 on each cable. The region follows shoe translations.
Counts use capsule centers strictly inside the sphere, not capsule-sphere overlap, and are split in robot-arm
order: the left arm controls `shoelace_right`, and the right arm controls `shoelace_left`.
`mdp.utils.untying_metrics(..., per_arm_throat_counts=True)` returns these counts as [N, 2]; its default
retains the legacy total-count [N] output. The per-side limit prevents one cleared cable from compensating
for excessive occupancy on the other. Logged counts are rollout/environment averages and can be fractional;
the success predicate checks each environment's integer counts before reset.
It is a regional geometry heuristic, not a general topological proof of untying: changes to the cable asset,
segment resolution or knot geometry require revalidating the radius and count thresholds. Configure the
per-side count through `env.terminations.success.params.maximum_throat_segments_per_arm`.
The initial 15-per-side limit rejects the audited unilateral trajectory's
19-23 remaining capsules on the ungrasped side, while allowing its pulled side's 12-13; it still needs
calibration on genuine bilateral completions.

The signed X boundaries are configured through `env.terminations.success.params.minimum_tail_outward_distance`.
They are absolute offsets from the fixed midpoint, not displacements from reset. Unlike the old 3D-radius
and total-separation checks, downward motion or extra right-tail travel cannot substitute for placing the
left tail sufficiently far left. The distances do not need to be equal. These final-state requirements and
the loaded-pull history below serve different purposes and must both be satisfied.

The default success configuration no longer contains redundant `tail_success_distance`, `threshold`, or
`maximum_throat_segments` settings: the per-tail X bounds already imply radial distances of at least 0.09 m
and total X separation of at least 0.18 m, and the two capsule limits imply a total no greater than 30.
The new cooperative term accepts only the per-side geometric limits. The pre-existing stateless legacy
helper retains its original interface for explicit historical comparisons.

The dense reward's `success_x_separation` remains a pull-normalization scale, not an additional success
gate. Its default is derived as twice `TAIL_SUCCESS_OUTWARD_DISTANCE` so the initial reward and success
geometry agree without maintaining two independent constants. Runtime overrides of these independent
terms should keep that relation when changing the task target. `minimum_pull_distance` instead measures
earned displacement during bilateral grasps, not final position. The approach, pregrasp, and grasp-hold
terms shape different phases or retention, and their individual parameters remain available for ablations.

The default success term additionally requires each tail to earn at least 0.025 m of new outward X records
while **both** hands have valid grasps. Configure this initial threshold with
`env.terminations.success.params.minimum_pull_distance`. A valid grasp uses the same two-finger contact,
actual closure, and low-slip criterion as the rewards. Dense acquisition, grasp retention, and success
all call the single `mdp.shoelace_grasp_quality` implementation; their filters and episode histories remain independent.
Common defaults are defined once in `_GRASP_PARAMS` and copied into each term's parameter dictionary,
so per-term overrides remain independent.
Both raw and 0.10 s filtered qualities must be at least 0.2 for each hand at both ends of the policy-step
interval; configure the threshold with `env.terminations.success.params.grasp_threshold`.

Each tail's record starts at the first finite post-reset sample and advances even without valid grasps.
Only increments beyond that physical record during qualifying intervals accumulate loaded distance.
Passive movement followed by grasping, release with a still-positive filtered score, or inward/outward
cycles cannot retroactively earn completion credit. Invalid observations break interval eligibility.
The cooperative phase stays completed after both distances reach the threshold, so release before
geometric completion is allowed. Geometry must still hold at the terminal step. The term is independent
of reward history and reward weights; `success_rate`, the success termination, and the +5 event all use
this stricter result. Contact quality remains a proxy, not a measured tension or force-closure guarantee.

Migration: checkpoints keep the same action and observation interfaces, but must be re-evaluated under
the new objective. Historical geometry-only success rates and the earlier `geometry_success` metric are
not comparable to the stricter per-cable-count and signed-X result.
Restart existing training/play processes to load the new term. Reward weights and dense reward formulas
are unchanged. The stateless `mdp.shoelace_success` geometry-only helper and
`mdp.tail_x_separation_success` distance-only helper remain available for explicit legacy evaluation.
To use the former, replace the success term with a `DoneTerm` using only `threshold`, `throat_radius`,
`maximum_throat_segments`, `tail_success_distance`, and `cable_cfgs`; do not pass the cooperative parameters.
Use the saved run's parameter values, including its old total count limit of 52, when reproducing that legacy criterion.
`MAXIMUM_THROAT_SEGMENTS` and `TAIL_SUCCESS_DISTANCE` remain as legacy module constants, but are not used
by the current cooperative success configuration.

The deprecated `maximum_progress_rate`, `approach_weight`, `grasp_weight`, and `task_weight` parameters
have been removed. If only `maximum_progress_rate` was used, delete that key. Configurations using any
of the three old weights must be migrated before loading:

1. For old weights `a` (approach), `g` (grasp), and `p` (task), set
   `acquisition_weight=(a+g)/(a+g+p)` and `approach_fraction=a/(a+g)`; use zero for the latter if `a+g=0`.
   Omitted old weights defaulted to `a=0.15`, `g=0.35`, and `p=1.0`.
2. Multiply the reward manager's overall term weight by `a+g+p` to preserve the potential scale.
3. Delete all four deprecated parameters. The ignored `maximum_progress_rate` has no replacement.

Pass `cable_cfgs` and `robot_cfgs` by keyword in direct calls; removing the old parameters changed their
positional indices. The migration preserves phase budgets, not historical reward trajectories. Current
task defaults already use `acquisition_weight`, `approach_fraction`, and `bilateral_pull_fraction`.

The task uses the assets in `scripts/demos/shoelace/assets`.

## Reset randomization

Each episode independently samples uniform offsets for every selected environment:

- Each of the seven joints on each arm starts within +/-0.02 rad (about 1.15 degrees) of its nominal
  pregrasp position, clamped to the soft joint limits. Finger positions stay at the default open position.
  Joint velocities stay at their defaults, and arm position targets match the sampled positions.
- The shoe moves by up to +/-0.02 m along each of X and Y; its height and orientation stay unchanged.
  Both settled laces, including their fixed anchor segments, receive the same translation. The shoe collider,
  tongue, visual mesh and pinned lace mesh move with the kinematic shoe, which stays fixed during the episode.

These conservative ranges vary the initial approach without changing the knot geometry. They help diversify
training starts; broader generalization still needs evaluation on held-out starts.

The ranges live in `EventsCfg.reset_left_arm`, `reset_right_arm`, and `reset_shoe` in `shoelace_env_cfg.py`.
To restore the previous deterministic starts, keep `reset_scene` and disable the three randomization terms
before constructing the environment:

```python
cfg.events.reset_left_arm = None
cfg.events.reset_right_arm = None
cfg.events.reset_shoe = None
```

`shoe_asset_cfg()` now returns a `RigidObjectCfg` for the kinematic shoe. The pinned mesh prim moved from
`{ENV_REGEX_NS}/ShoelacePinned` to `{ENV_REGEX_NS}/Shoe/ShoelacePinned`; update custom prim-path lookups accordingly.
The scene entity name `shoelace_pinned_visual` is unchanged.

## Explicit proxy inertia

`ShoelaceEnvCfg.cable_inertia_regularization` defaults to `1e-6` [kg*m^2]. Before Newton
finalization, each dynamic cable segment receives `I_effective = I_geometry + regularization * identity`
after capsule mass/radius correction and anchor pinning. Its inverse inertia is updated too. Masses,
fixed anchors, shoe and robot inertias are unchanged. This is an intentional approximation of rotational
dynamics for the current solver budget, not extra damping or a change to the cable's material stiffness.
The default approximates the previously used effective inertia without depending on Newton's automatic repair.

Override it with `env.cable_inertia_regularization=3e-7` appended to the training command below.
Values must be finite and nonnegative. Zero disables only the task-local addition: Newton's normal validation
still applies and may enlarge very small inertias. Recheck passive settling, contact penetration and scripted
grasp/pull behavior when changing this value; previous settled states and policies may behave differently.
The standalone demos expose the same setting as `--cable_inertia_regularization`.

```bash
uv run isaaclab random_agent --task IsaacContrib-Shoelace-DualFranka --num_envs 4 --device cuda:0
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Shoelace-DualFranka \
    --num_envs 4 --max_iterations 1 --device cuda:0
```
