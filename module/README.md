# module 目录说明

该文档说明 `module/` 下各模块职责，以及每个文件顶层函数/类的用途。

## 模块总表

| 模块文件 | 作用 | 顶层符号数量 |
|---|---|---:|
| `GT2NewTrack.py` | Generated Track 图像转换到 NewTrack 风格（线检测、路面重绘、批处理入口）。 | 15 |
| `RRL2NewTrack.py` | RoboRacingLeague 场景转换到 NewTrack 风格（白线/棋盘/路面检测与重绘）。 | 25 |
| `WS2NewTrack.py` | Waveshare 场景转换到 NewTrack 风格（语义掩码、中心线推断、颜色迁移）。 | 34 |
| `__init__.py` | module 包统一导出入口（对外 import 聚合）。 | 0 |
| `action_adapter.py` | 动作空间适配封装（策略动作到环境动作映射）。 | 1 |
| `actor.py` | 策略特征提取器（FiLMFeatureExtractor）。 | 1 |
| `callbacks.py` | 训练回调集合（导出、统计、学习率、最佳模型、恢复、进度）。 | 10 |
| `control.py` | 控制链路 wrapper（高层控制、安全限幅、油门控制、曲率感知油门）。 | 4 |
| `green_vehicle_detect.py` | 绿色车辆检测工具（多方法检测、融合、批处理与可视化）。 | 19 |
| `multi_scene_env.py` | 多场景环境与包装器（场景切换、观测构建、V12/V13 wrapper 链）。 | 8 |
| `obv.py` | 观测处理模块（语义观测包装与 v13 状态构造）。 | 2 |
| `reward.py` | 基础奖励包装器（DonkeyRewardWrapper）。 | 1 |
| `robust_lane_detector.py` | 稳健车道检测器与黄色车道增强器。 | 4 |
| `track.py` | 赛道几何数据结构与管理器。 | 2 |
| `utils.py` | 通用工具函数（配置、随机种子、检查点、数值工具、JSON 写入）。 | 11 |
| `v14_attention.py` | V14 注意力网络组件（CBAM + AttentionCNN）。 | 4 |
| `v14_callbacks.py` | V14 控制相关 TensorBoard 回调。 | 1 |
| `v14_cli.py` | V14 命令行参数解析与程序入口。 | 2 |
| `v14_curriculum.py` | V14 课程学习管理器。 | 1 |
| `v14_dep_generatedtrack_base.py` | V14 底层依赖基座（generated-track 旧版核心能力抽取与兼容层）。 | 41 |
| `v14_dep_sim_core.py` | V14 仿真底层依赖（Sim 扩展 API、NPC 控制、旧训练流程依赖）。 | 14 |
| `v14_distillation.py` | V14 策略蒸馏管理器。 | 1 |
| `v14_env_factory.py` | V14 环境工厂（环境构建与 track cache 查询）。 | 2 |
| `v14_paths.py` | V14 路径/配置文件定位工具。 | 1 |
| `v14_racing_line.py` | V14 racing line 计算器。 | 1 |
| `v14_stage_profiles.py` | V14 分阶段配置与训练参数模板。 | 4 |
| `v14_train.py` | V14 训练主流程（train_v14）。 | 1 |
| `v14_wrapper.py` | V14 主 wrapper（组合 reward/npc/spawn mixin）。 | 1 |
| `v14_wrapper_npc.py` | V14 NPC 运行时 mixin（速度约束、重置、碰撞后处理）。 | 1 |
| `v14_wrapper_reward.py` | V14 奖励 mixin（racing/proactive/dynamic/chaos 奖励与 step 后处理）。 | 1 |
| `v14_wrapper_spawn.py` | V14 出生点/重置 mixin（spawn、teleport、布局逻辑）。 | 1 |
| `wrappers.py` | 通用观测/扰动 wrappers（转置、归一化、黄线、重置扰动、缩放）。 | 6 |

## 函数/类功能索引（按模块）

说明：这里列的是每个 `.py` 文件的顶层函数/类（不展开类内 methods）。

### `GT2NewTrack.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `_denoise_shape` | 函数 | Remove small non-elongated blobs from a binary mask. |
| `detect_gt_road_support` | 函数 | Estimate the drivable road neighborhood and exclude grass/off-road regions. |
| `_apply_road_support` | 函数 | 应用函数：应用  apply road support 配置/变换。 |
| `_split_clusters` | 函数 | 内部辅助函数：处理  split clusters。 |
| `_row_multi_centroid_encode` | 函数 | 内部辅助函数：处理  row multi centroid encode。 |
| `_vertical_continuity_filter` | 函数 | 内部辅助函数：处理  vertical continuity filter。 |
| `_recover_far_thin_segments` | 函数 | Recover faint upper-frame segments without reopening broad near-field noise. |
| `_detect_horizontal_line_supplement` | 函数 | 检测函数：识别  detect horizontal line supplement。 |
| `_detect_white_line_local_contrast` | 函数 | 检测函数：识别  detect white line local contrast。 |
| `detect_white_line` | 函数 | Return binary mask of white edge lines. |
| `detect_yellow_line` | 函数 | Return binary mask of yellow center line. |
| `recolor_lines` | 函数 | Return a copy of img_bgr with line pixels replaced by target colors. |
| `render_road_surface` | 函数 | Transform road/bg to NT cloth texture. Lines are NOT modified. |
| `process_directory` | 函数 | Detect lines and recolor for all images in raw_dir. |
| `main` | 函数 | 程序主入口。 |

### `RRL2NewTrack.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `_mask_u8` | 函数 | 内部辅助函数：处理  mask u8。 |
| `_kernel` | 函数 | 内部辅助函数：处理  kernel。 |
| `_split_row_clusters` | 函数 | Split a sorted array of column indices into clusters by gap. |
| `_row_multi_centroid_encode` | 函数 | Like WS2NewTrack's _row_centroid_encode, but keeps ALL line clusters |
| `_vertical_continuity_filter` | 函数 | Remove columns of mask that don't have vertical continuity. |
| `_filter_rrl_candidate_components` | 函数 | Reject large compact bright blobs before row-wise encoding. |
| `_detect_rrl_horizontal_supplement` | 函数 | Recover rare near-horizontal white lines when the vertical pass is weak. |
| `_recover_rrl_upper_adjacent_segments` | 函数 | Keep short upper-curve segments when they sit right next to a kept line. |
| `_recover_rrl_curve_continuation_segments` | 函数 | Recover short upper/mid arc fragments that continue a kept lower line. |
| `_recover_rrl_candidate_arc_bands` | 函数 | Recover wide-but-thin upper arc bands skipped by row encoding. |
| `_bootstrap_rrl_empty_detection` | 函数 | Bootstrap a seed line when the normal cleanup rejects everything. |
| `detect_rrl_white_lines` | 函数 | Detect ALL white lane lines (near, far, straight, curved). |
| `_rrl_blue_line_mask` | 函数 | Use the detected white-line width directly instead of globally thickening it. |
| `detect_rrl_checker` | 函数 | Detect black/white checkered start/finish line. |
| `detect_rrl_road_surface` | 函数 | Detect the dark asphalt road. Low saturation, wide V range. |
| `detect_rrl_colored_bg` | 函数 | Detect colored background (blue barriers, green areas, etc.). |
| `_lowfreq_value_field` | 函数 | 内部辅助函数：处理  lowfreq value field。 |
| `_paint_flat` | 函数 | 内部辅助函数：处理  paint flat。 |
| `_paint_textured` | 函数 | Paint with target color but preserve scaled high-frequency texture. |
| `_blend_to_proto` | 函数 | 内部辅助函数：处理  blend to proto。 |
| `transform_rrl_to_newtrack` | 函数 | Full style transfer: re-render entire image in new_track style. |
| `transform_rrl_lines_only` | 函数 | Minimal: only recolor white lines to blue, keep everything else. |
| `process_directory` | 函数 | 处理函数：执行 process directory。 |
| `_save_preview` | 函数 | 保存函数：写出  save preview。 |
| `main` | 函数 | 程序主入口。 |

### `WS2NewTrack.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `_natural_key` | 函数 | 内部辅助函数：处理  natural key。 |
| `list_images` | 函数 | 功能函数：实现 list images。 |
| `load_images` | 函数 | 加载函数：读取 load images。 |
| `sample_paths` | 函数 | 功能函数：实现 sample paths。 |
| `_u8` | 函数 | 内部辅助函数：处理  u8。 |
| `_k` | 函数 | 内部辅助函数：处理  k。 |
| `_remove_small` | 函数 | 内部辅助函数：处理  remove small。 |
| `_ws_green_text_mask` | 函数 | 内部辅助函数：处理  ws green text mask。 |
| `_filter_centerline_components` | 函数 | 内部辅助函数：处理  filter centerline components。 |
| `semantic_masks` | 函数 | 功能函数：实现 semantic masks。 |
| `road_support_mask` | 函数 | 功能函数：实现 road support mask。 |
| `overlay_semantics` | 函数 | 功能函数：实现 overlay semantics。 |
| `compute_stats` | 函数 | 功能函数：实现 compute stats。 |
| `_extract_line_mask` | 函数 | 提取函数：抽取  extract line mask。 |
| `_split_row_clusters` | 函数 | 内部辅助函数：处理  split row clusters。 |
| `_row_centroid_encode` | 函数 | 内部辅助函数：处理  row centroid encode。 |
| `_center_band_from_yellow` | 函数 | 内部辅助函数：处理  center band from yellow。 |
| `_center_band_from_seed` | 函数 | 内部辅助函数：处理  center band from seed。 |
| `_white_center_support_band` | 函数 | 内部辅助函数：处理  white center support band。 |
| `_expand_white_seed_to_center_blocks` | 函数 | 内部辅助函数：处理  expand white seed to center blocks。 |
| `_clip_center_fill_width` | 函数 | 裁剪辅助函数：限制  clip center fill width 取值范围。 |
| `_add_center_white_edge_ring` | 函数 | 内部辅助函数：处理  add center white edge ring。 |
| `_get_proto` | 函数 | 查询辅助函数：获取  get proto。 |
| `_get_center_proto` | 函数 | 查询辅助函数：获取  get center proto。 |
| `_sanitize_blue` | 函数 | 内部辅助函数：处理  sanitize blue。 |
| `_resolve_tgt_blue` | 函数 | 内部辅助函数：处理  resolve tgt blue。 |
| `_sanitize_center` | 函数 | 内部辅助函数：处理  sanitize center。 |
| `_resolve_tgt_center` | 函数 | 内部辅助函数：处理  resolve tgt center。 |
| `_blend_to_proto` | 函数 | 内部辅助函数：处理  blend to proto。 |
| `_apply_newtrack_blend` | 函数 | 应用函数：应用  apply newtrack blend 配置/变换。 |
| `extract_ws_yellow_mask` | 函数 | 提取函数：抽取 extract ws yellow mask。 |
| `extract_ws_white_case10_line` | 函数 | Extract the WS white center dashes that are recolored into newtrack's warm |
| `_render_ws_road_surface` | 函数 | Transform WS road surface to NT cloth texture. Lines are NOT modified. |
| `transform_ws_to_newtrack` | 函数 | 转换函数：执行 transform ws to newtrack。 |

### `__init__.py`

- 无顶层函数/类（导出聚合文件）。

### `action_adapter.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `ActionAdapterWrapper` | 类 | V13 3D → 2D action adapter, replacing HighLevelControlWrapper. |

### `actor.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `FiLMFeatureExtractor` | 类 | FiLM-conditioned feature extractor for Dict obs with keys "image" and "state". |

### `callbacks.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `PTHExportCallback` | 类 | 每 save_freq 步自动导出 .zip + .pth。 |
| `CoverageLoggingCallback` | 类 | 周期性记录 mask 通道 coverage 统计到 TensorBoard。 |
| `PerSceneStatsCallback` | 类 | 按 scene_key 汇总 ep_info_buffer，避免混合均值掩盖单场景退化。 |
| `SceneSchedulerLoggingCallback` | 类 | 记录 MultiSceneEnv 的场景采样与动态调权状态。 |
| `AdaptiveLearningRateCallback` | 类 | 根据「平衡分数回落」和「approx_kl 过高」自动降低学习率。 |
| `TrainingMetricsFileLoggerCallback` | 类 | 将训练关键指标周期性写入 JSONL 文件，断线后离线分析可用。 |
| `BestModelCallback` | 类 | 按 scene_key 保存最佳模型（全局 best + 每场景 best + 可选平衡分数 best）。 |
| `CrashRecoveryCallback` | 类 | 监控每个场景的 ep_len 滑动平均值，当检测到严重退化时自动回滚模型到最佳 checkpoint。 |
| `ShortEpisodeLoggerCallback` | 类 | 记录 episode 步数 < threshold 的早终止事件到 JSONL 文件。 |
| `TqdmProgressCallback` | 类 | tqdm 进度条，显示当前步数、全局平均奖励、各场景奖励。 |

### `control.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `HighLevelControlWrapper` | 类 | 高层速度控制器。 |
| `ActionSafetyWrapper` | 类 | 转向执行保护（简化版）。 |
| `ThrottleControlWrapper` | 类 | 全局油门边界裁剪。 |
| `CurvatureAwareThrottleWrapper` | 类 | 曲率感知油门上限（兼容保留）。 |

### `green_vehicle_detect.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `VehicleDetection` | 类 | 单个绿色车辆的检测结果 |
| `DetectionResult` | 类 | 一帧图像的完整检测结果 |
| `_denoise_and_filter` | 函数 | 增强的去噪和过滤函数 |
| `detect_a_improved` | 函数 | A: H[35-85] S[80-255] + 增强过滤 (宽松但有饱和度过滤) |
| `detect_b_improved` | 函数 | B: H[40-80] S[100-255] + 过滤 (中等严格度) ✓ CLEAN |
| `detect_c_improved` | 函数 | C: H[45-75] S[120-255] + 过滤 (较严格) ✓ CLEAN |
| `detect_d_improved` | 函数 | D: A + 形态学 + 增强过滤 |
| `detect_e_improved` | 函数 | E: B + 严格饱和度过滤 (S>115) ✓ CLEAN |
| `detect_f_improved` | 函数 | F: C + 极严格过滤 |
| `detect_g_improved` | 函数 | G: A + ROI (下2/3) + 过滤 |
| `detect_h_improved` | 函数 | H: H[38-82] S[110-255] V[70-200] + 过滤 ✓ CLEAN |
| `detect_i_improved` | 函数 | I: 双阈值 + 颜色纯度 + 过滤 ✓ CLEAN |
| `_get_iou` | 函数 | 计算两个bbox的IOU |
| `_extract_objects_from_mask` | 函数 | 从掩码中提取所有连通域 |
| `_fuse_detections` | 函数 | 融合多个方法的检测结果，使用投票机制 |
| `GreenVehicleDetector` | 类 | 绿色车辆检测器，支持融合检测和简单模式。 |
| `_overlay_mask` | 函数 | 内部辅助函数：处理  overlay mask。 |
| `_build_compare_grid` | 函数 | 生成多方案对比网格图 |
| `_run_batch` | 函数 | 对obstacle目录里的图片批量运行检测 |

### `multi_scene_env.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `_donkey_episode_over_no_checkpoint` | 函数 | 替换 DonkeyUnitySimHandler.determine_episode_over: |
| `_install_custom_episode_over` | 函数 | 在 base_env (DonkeyEnv) 上安装自定义 episode_over 函数。 |
| `_set_handler_max_cte` | 函数 | per-scene 设置 handler.max_cte，用于缩短出轨后无效步数。 |
| `MultiSceneEnv` | 类 | 多场景交替训练环境（单模拟器 + 场景切换方案） |
| `MultiInputObsWrapper` | 类 | Return Dict observation: |
| `_build_v12_wrapper_chain` | 函数 | 构建 V12 wrapper 链并返回 (env, action_safety, high_level, reward_wrapper)。 |
| `MultiSceneEnvV12` | 类 | V12 多场景训练环境。 |
| `MultiSceneEnvV13` | 类 | V13 多场景训练环境。 |

### `obv.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `CanonicalSemanticWrapper` | 类 | V13 专用：6 通道观测 = 原图辅助流 + *2NewTrack 语义流。 |
| `_build_state_v13` | 函数 | 返回 7 维状态向量，完全来自传感器 info + adapter 内部状态，无 TrackGeometryManager 依赖： |

### `reward.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `DonkeyRewardWrapper` | 类 | DonkeyCar 统一奖励包装器。 |

### `robust_lane_detector.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `RobustLaneDetector` | 类 | 鲁棒车道线检测器 V9.3c - per-domain 边沿线检测 + 行质心编码 |
| `RobustYellowLaneEnhancer` | 类 | V9.3c 鲁棒增强器 - per-domain 边沿线检测 |
| `validate_on_samples` | 函数 | 在已有采样图像上验证新检测器 vs V8旧检测器的效果 |
| `_save_comparison` | 函数 | 保存对比图（V8 vs V9） |

### `track.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `SceneGeometry` | 类 | 单条赛道的离线几何信息。 |
| `TrackGeometryManager` | 类 | 加载多赛道 JSON 并提供实时局部几何查询（x-z 平面）。 |

### `utils.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `load_config` | 函数 | 从 Python 文件加载大写变量CONFIG对象（若 donkeycar 可用则复用，否则自建）。 |
| `_get_domain_for_env` | 函数 | 根据 env_id 返回域标识（ws/gt，默认 ws）。 |
| `_seed_everything` | 函数 | 统一 Python / NumPy / PyTorch 随机性。 |
| `_safe_seed_env` | 函数 | best-effort 给 VecEnv/Gym env 设置 seed，失败不阻塞训练。 |
| `_evaluate_recurrent_policy_on_vec_env` | 函数 | 显式传递 LSTM state / episode_start 的评估循环。 |
| `_find_latest_checkpoint` | 函数 | 查找 `{name_prefix}_N_steps.zip` 中步数最大的文件。 |
| `_wrap_pi` | 函数 | 内部辅助函数：处理  wrap pi。 |
| `_clip_float` | 函数 | 裁剪辅助函数：限制  clip float 取值范围。 |
| `_safe_float` | 函数 | 安全地把输入转换为 float，失败时返回默认值。 |
| `_apply_global_seeds` | 函数 | 统一设置 Python/NumPy/PyTorch 随机种子。 |
| `_write_json` | 函数 | 把对象写入 JSON 文件并自动创建目录。 |

### `v14_attention.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `ChannelAttention` | 类 | CBAM Channel Attention: 学习通道间的重要性权重。 |
| `SpatialAttention` | 类 | CBAM Spatial Attention: 学习空间位置的重要性权重。 |
| `CBAMBlock` | 类 | CBAM: Channel Attention → Spatial Attention (串联)。 |
| `AttentionCNN` | 类 | CNN + CBAM 注意力特征提取器。 |

### `v14_callbacks.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `V14ControlTBCallback` | 类 | V14 TensorBoard callback: 记录阶段、奖励组件、避让指标。 |

### `v14_cli.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `parse_args` | 函数 | 解析 V14 命令行参数并返回配置对象。 |
| `main` | 函数 | V14 命令行入口：根据模式调用训练/校准逻辑。 |

### `v14_curriculum.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `CurriculumManagerV14` | 类 | V14 课程管理器: 4阶段升级 + 策略蒸馏触发 + 混合采样控制。 |

### `v14_dep_generatedtrack_base.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `_clip` | 函数 | 裁剪辅助函数：限制  clip 取值范围。 |
| `_safe_float` | 函数 | 安全辅助函数：处理  safe float。 |
| `_parse_float_list` | 函数 | Parse comma-separated float list from CLI string. |
| `_apply_global_seeds` | 函数 | 应用函数：应用  apply global seeds 配置/变换。 |
| `_sha256_file` | 函数 | 内部辅助函数：处理  sha256 file。 |
| `_safe_module_version` | 函数 | 安全辅助函数：处理  safe module version。 |
| `_git_metadata_for_path` | 函数 | 内部辅助函数：处理  git metadata for path。 |
| `_write_json` | 函数 | 内部辅助函数：处理  write json。 |
| `ManualWidthSpawnSampler` | 类 | Sample in-track spawn poses from manual width profile outline. |
| `FixedSuccessGuardState` | 类 | Windowed state for fixed2 success guard. |
| `build_dist_scale_profile` | 函数 | 基于 query 到的 nodes/fine_track 生成仿真内一致尺度档案。 |
| `StageConfig` | 类 | 配置数据类：定义 StageConfig 配置字段。 |
| `_max_npc_count_from_stage` | 函数 | 内部辅助函数：处理  max npc count from stage。 |
| `StageTrainProfile` | 类 | 配置数据类：定义 StageTrainProfile 配置字段。 |
| `StageEvalGate` | 类 | 配置数据类：定义 StageEvalGate 配置字段。 |
| `_collect_cli_overrides` | 函数 | 近似收集显式传入的 CLI 选项（dest 名风格）。 |
| `_cli_overrode` | 函数 | 内部辅助函数：处理  cli overrode。 |
| `_set_model_hparams_from_profile` | 函数 | 设置辅助函数：设置  set model hparams from profile。 |
| `apply_stage_train_profile` | 函数 | 应用函数：应用 apply stage train profile 配置/变换。 |
| `stage_eval_gate_pass` | 函数 | 功能函数：实现 stage eval gate pass。 |
| `GeneratedTrackV10Wrapper` | 类 | V10 wrapper: 单地图 generated_track + spawn校验器 + 尺度感知奖励模式。 |
| `GeneratedTrackV11_1Wrapper` | 类 | V11.1 wrapper: single-car target-controller with anti-no-motion shaping. |
| `GeneratedTrackV11_2Wrapper` | 类 | V11.2: fixed2 mode with manual-width in-track spawn and fixed NPC refresh-on-success. |
| `CurriculumManager` | 类 | 训练外层（learn chunk之后）执行评估与升阶。 |
| `evaluate_stage_on_same_env` | 函数 | 在训练间隙用同一环境做评估（非并发）。适配 RecurrentPPO predict 接口。 |
| `_extract_step_from_ckpt_path` | 函数 | 提取函数：抽取  extract step from ckpt path。 |
| `_checkpoint_eval_sort_key` | 函数 | 排序优先级： |
| `_apply_v11_2_curriculum_overrides` | 函数 | Apply fixed2 mode overrides. Does NOT force stage to 2. |
| `_apply_npc_mode_override` | 函数 | 允许通过CLI强制覆盖 NPC 行为模式（跨阶段保持）。 |
| `eval_checkpoints_mode` | 函数 | 模式4：批量评估checkpoint并挑选当前最佳模型。 |
| `force_reload_scene` | 函数 | 强制切场景：先 exit_scene 再 load_scene，防止落到默认地图。 |
| `create_generated_env_and_npcs` | 函数 | 单地图 generated_track 环境创建（无多地图切换）。 |
| `calibrate_only` | 函数 | 模式1：只做节点查询与比例尺标定。 |
| `reset_stress_mode` | 函数 | 模式2：连续 reset 压测，验证 spawn 稳定性。 |
| `export_spawn_table_mode` | 函数 | 模式5：仅根据 manual width profile 导出赛道内离散出生点表。 |
| `V11ControlTBCallback` | 类 | Collect control-layer diagnostics from env info and write to TensorBoard. |
| `train_mode` | 函数 | 模式3：RecurrentPPO(LSTM) 单地图训练 + 分阶段评估升阶。 |
| `cleanup_envs` | 函数 | 功能函数：实现 cleanup envs。 |
| `parse_args` | 函数 | 解析命令行参数。 |
| `_enforce_v11_2_mode` | 函数 | 内部辅助函数：处理  enforce v11 2 mode。 |
| `main` | 函数 | 程序主入口。 |

### `v14_dep_sim_core.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `SimExtendedAPI` | 类 | 封装 DonkeySim 的隐藏 API: |
| `TrackNodeCache` | 类 | 缓存赛道路径节点坐标 |
| `NPCController` | 类 | NPC障碍车控制器 |
| `MultiMapManager` | 类 | 多地图轮换管理器 |
| `YellowLaneEnhancer` | 类 | 黄色车道线增强 (对齐V8, 含DR和CLAHE) |
| `OvertakeTrainingWrapper` | 类 | V9 超车训练环境包装器 (观测对齐V8 8通道管道) |
| `LightweightCNN` | 类 | 轻量级CNN - 对齐V8 |
| `OvertakeCurriculumCallback` | 类 | 超车训练课程学习（修正版） |
| `MapSwitchCallback` | 类 | 多地图切换回调 |
| `AutoSaveCallback` | 类 | 定期保存模型 + .pth |
| `BestModelCallback` | 类 | 保存最佳模型 |
| `create_env_and_npcs` | 函数 | 创建（或重建）环境 + NPC + 查询节点 |
| `train` | 函数 | 主训练入口 |
| `parse_args` | 函数 | 解析命令行参数。 |

### `v14_distillation.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `PolicyDistillationManager` | 类 | 策略蒸馏防遗忘: 在阶段切换时快照策略网络，后续阶段通过KL散度惩罚防止遗忘。 |

### `v14_env_factory.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `_query_generated_track_cache_from_env` | 函数 | 从环境对象提取 generated_track 节点缓存。 |
| `create_v14_env_and_npcs` | 函数 | 创建 V14 训练环境与 NPC 控制器，并返回训练所需对象。 |

### `v14_paths.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `default_generated_track_profile` | 函数 | Return default manual-width profile path for generated_track. |

### `v14_racing_line.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `RacingLineComputer` | 类 | 从赛道曲率 profile 预计算最优赛车线偏移。 |

### `v14_stage_profiles.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `StageConfigV14` | 类 | 配置数据类：定义 StageConfigV14 配置字段。 |
| `StageTrainProfileV14` | 类 | 配置数据类：定义 StageTrainProfileV14 配置字段。 |
| `StageEvalGateV14` | 类 | 配置数据类：定义 StageEvalGateV14 配置字段。 |
| `apply_stage_train_profile_v14` | 函数 | 按阶段配置覆盖训练超参数并同步到 wrapper/model。 |

### `v14_train.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `train_v14` | 函数 | V14 训练主流程：构建环境、创建模型、执行分阶段训练并保存结果。 |

### `v14_wrapper.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `GeneratedTrackV14Wrapper` | 类 | V14 Wrapper: 继承V11_1的控制层，新增: |

### `v14_wrapper_npc.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `V14NpcRuntimeMixin` | 类 | 核心类：实现 V14NpcRuntimeMixin 相关功能。 |

### `v14_wrapper_reward.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `V14RewardMixin` | 类 | 核心类：实现 V14RewardMixin 相关功能。 |

### `v14_wrapper_spawn.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `V14SpawnResetMixin` | 类 | 核心类：实现 V14SpawnResetMixin 相关功能。 |

### `wrappers.py`

| 符号 | 类型 | 功能 |
|---|---|---|
| `GeneralizationWrapper` | 类 | 泛化性增强包装器 - 在 enable_step 步后对 RGB 通道施加亮度/噪声扰动。 |
| `TransposeWrapper` | 类 | HWC → CHW（PyTorch 格式）。 |
| `NormalizeWrapper` | 类 | V8 8 通道归一化（V9.4 修正 per-channel bounds）： |
| `V9YellowLaneWrapper` | 类 | V9.4 三域对齐增强包装器：per-domain edge detection + 行质心编码 + Coverage Dropout。 |
| `GTResetPerturbWrapper` | 类 | GT 场景 reset 后执行少量随机动作，防止策略"记脚本"固定起点时序。 |
| `RGBResizeWrapper` | 类 | V12 专用：将 DonkeyEnv 原始 uint8 HWC 图像 (H, W, 3) |

## 维护约定

- `v14_*` 文件为 V14 线模块化实现，优先在该组文件扩展新能力。
- `v14_dep_*` 文件为底层兼容依赖层，除非必要不做大范围重写。
- 新增公共符号后，同步更新 `module/__init__.py` 与本文档。
