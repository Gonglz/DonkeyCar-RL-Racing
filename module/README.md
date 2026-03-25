# module 目录说明

该文档说明 `module/` 下各模块职责，以及每个文件的顶层类/函数索引，便于定位功能与调试入口。

## 模块总表

| 模块文件 | 作用 | 顶层函数/类 |
|---|---|---|
| `GT2NewTrack.py` | Generated Track 图像转换到 NewTrack 风格（线检测、路面重绘、批处理入口）。 | `def _denoise_shape`，`def detect_gt_road_support`，`def _apply_road_support`，`def _split_clusters`，`def _row_multi_centroid_encode`，`def _vertical_continuity_filter`，`def _recover_far_thin_segments`，`def _detect_horizontal_line_supplement`，`def _detect_white_line_local_contrast`，`def detect_white_line`，`def detect_yellow_line`，`def recolor_lines`，`def render_road_surface`，`def process_directory`，`def main` |
| `RRL2NewTrack.py` | RoboRacingLeague 场景转换到 NewTrack 风格（白线/棋盘/路面检测与重绘）。 | `def _mask_u8`，`def _kernel`，`def _split_row_clusters`，`def _row_multi_centroid_encode`，`def _vertical_continuity_filter`，`def _filter_rrl_candidate_components`，`def _detect_rrl_horizontal_supplement`，`def _recover_rrl_upper_adjacent_segments`，`def _recover_rrl_curve_continuation_segments`，`def _recover_rrl_candidate_arc_bands`，`def _bootstrap_rrl_empty_detection`，`def detect_rrl_white_lines`，`def _rrl_blue_line_mask`，`def detect_rrl_checker`，`def detect_rrl_road_surface`，`def detect_rrl_colored_bg`，`def _lowfreq_value_field`，`def _paint_flat`，`def _paint_textured`，`def _blend_to_proto`，`def transform_rrl_to_newtrack`，`def transform_rrl_lines_only`，`def process_directory`，`def _save_preview`，`def main` |
| `WS2NewTrack.py` | Waveshare 场景转换到 NewTrack 风格（语义掩码、中心线推断、颜色迁移）。 | `def _natural_key`，`def list_images`，`def load_images`，`def sample_paths`，`def _u8`，`def _k`，`def _remove_small`，`def _ws_green_text_mask`，`def _filter_centerline_components`，`def semantic_masks`，`def road_support_mask`，`def overlay_semantics`，`def compute_stats`，`def _extract_line_mask`，`def _split_row_clusters`，`def _row_centroid_encode`，`def _center_band_from_yellow`，`def _center_band_from_seed`，`def _white_center_support_band`，`def _expand_white_seed_to_center_blocks`，`def _clip_center_fill_width`，`def _add_center_white_edge_ring`，`def _get_proto`，`def _get_center_proto`，`def _sanitize_blue`，`def _resolve_tgt_blue`，`def _sanitize_center`，`def _resolve_tgt_center`，`def _blend_to_proto`，`def _apply_newtrack_blend`，`def extract_ws_yellow_mask`，`def extract_ws_white_case10_line`，`def _render_ws_road_surface`，`def transform_ws_to_newtrack` |
| `__init__.py` | module 包统一导出入口（对外 import 聚合）。 | （无顶层函数/类，导出聚合文件） |
| `action_adapter.py` | 动作空间适配封装（策略动作到环境动作映射）。 | `class ActionAdapterWrapper` |
| `actor.py` | 策略特征提取器（FiLMFeatureExtractor）。 | `class FiLMFeatureExtractor` |
| `callbacks.py` | 训练回调集合（导出、统计、学习率、最佳模型、恢复、进度）。 | `class PTHExportCallback`，`class CoverageLoggingCallback`，`class PerSceneStatsCallback`，`class SceneSchedulerLoggingCallback`，`class AdaptiveLearningRateCallback`，`class TrainingMetricsFileLoggerCallback`，`class BestModelCallback`，`class CrashRecoveryCallback`，`class ShortEpisodeLoggerCallback`，`class TqdmProgressCallback` |
| `control.py` | 控制链路 wrapper（高层控制、安全限幅、油门控制、曲率感知油门）。 | `class HighLevelControlWrapper`，`class ActionSafetyWrapper`，`class ThrottleControlWrapper`，`class CurvatureAwareThrottleWrapper` |
| `green_vehicle_detect.py` | 绿色车辆检测工具（多方法检测、融合、批处理与可视化）。 | `class VehicleDetection`，`class DetectionResult`，`def _denoise_and_filter`，`def detect_a_improved`，`def detect_b_improved`，`def detect_c_improved`，`def detect_d_improved`，`def detect_e_improved`，`def detect_f_improved`，`def detect_g_improved`，`def detect_h_improved`，`def detect_i_improved`，`def _get_iou`，`def _extract_objects_from_mask`，`def _fuse_detections`，`class GreenVehicleDetector`，`def _overlay_mask`，`def _build_compare_grid`，`def _run_batch` |
| `multi_scene_env.py` | 多场景环境与包装器（场景切换、观测构建、V12/V13 wrapper 链）。 | `def _donkey_episode_over_no_checkpoint`，`def _install_custom_episode_over`，`def _set_handler_max_cte`，`class MultiSceneEnv`，`class MultiInputObsWrapper`，`def _build_v12_wrapper_chain`，`class MultiSceneEnvV12`，`class MultiSceneEnvV13` |
| `obv.py` | 观测处理模块（语义观测包装与 v13 状态构造）。 | `class CanonicalSemanticWrapper`，`def _build_state_v13` |
| `reward.py` | 基础奖励包装器（DonkeyRewardWrapper）。 | `class DonkeyRewardWrapper` |
| `robust_lane_detector.py` | 稳健车道检测器与黄色车道增强器。 | `class RobustLaneDetector`，`class RobustYellowLaneEnhancer`，`def validate_on_samples`，`def _save_comparison` |
| `track.py` | 赛道几何数据结构与管理器。 | `class SceneGeometry`，`class TrackGeometryManager` |
| `utils.py` | 通用工具函数（配置、随机种子、检查点、数值工具、JSON 写入）。 | `def load_config`，`def _get_domain_for_env`，`def _seed_everything`，`def _safe_seed_env`，`def _evaluate_recurrent_policy_on_vec_env`，`def _find_latest_checkpoint`，`def _wrap_pi`，`def _clip_float`，`def _safe_float`，`def _apply_global_seeds`，`def _write_json` |
| `v14_attention.py` | V14 注意力网络组件（CBAM + AttentionCNN）。 | `class ChannelAttention`，`class SpatialAttention`，`class CBAMBlock`，`class AttentionCNN` |
| `v14_callbacks.py` | V14 控制相关 TensorBoard 回调。 | `class V14ControlTBCallback` |
| `v14_cli.py` | V14 命令行参数解析与程序入口。 | `def parse_args`，`def main` |
| `v14_curriculum.py` | V14 课程学习管理器。 | `class CurriculumManagerV14` |
| `v14_dep_generatedtrack_base.py` | V14 底层依赖基座（generated-track 旧版核心能力抽取与兼容层）。 | `def _clip`，`def _safe_float`，`def _parse_float_list`，`def _apply_global_seeds`，`def _sha256_file`，`def _safe_module_version`，`def _git_metadata_for_path`，`def _write_json`，`class ManualWidthSpawnSampler`，`class FixedSuccessGuardState`，`def build_dist_scale_profile`，`class StageConfig`，`def _max_npc_count_from_stage`，`class StageTrainProfile`，`class StageEvalGate`，`def _collect_cli_overrides`，`def _cli_overrode`，`def _set_model_hparams_from_profile`，`def apply_stage_train_profile`，`def stage_eval_gate_pass`，`class GeneratedTrackV10Wrapper`，`class GeneratedTrackV11_1Wrapper`，`class GeneratedTrackV11_2Wrapper`，`class CurriculumManager`，`def evaluate_stage_on_same_env`，`def _extract_step_from_ckpt_path`，`def _checkpoint_eval_sort_key`，`def _apply_v11_2_curriculum_overrides`，`def _apply_npc_mode_override`，`def eval_checkpoints_mode`，`def force_reload_scene`，`def create_generated_env_and_npcs`，`def calibrate_only`，`def reset_stress_mode`，`def export_spawn_table_mode`，`class V11ControlTBCallback`，`def train_mode`，`def cleanup_envs`，`def parse_args`，`def _enforce_v11_2_mode`，`def main` |
| `v14_dep_sim_core.py` | V14 仿真底层依赖（Sim 扩展 API、NPC 控制、旧训练流程依赖）。 | `class SimExtendedAPI`，`class TrackNodeCache`，`class NPCController`，`class MultiMapManager`，`class YellowLaneEnhancer`，`class OvertakeTrainingWrapper`，`class LightweightCNN`，`class OvertakeCurriculumCallback`，`class MapSwitchCallback`，`class AutoSaveCallback`，`class BestModelCallback`，`def create_env_and_npcs`，`def train`，`def parse_args` |
| `v14_distillation.py` | V14 策略蒸馏管理器。 | `class PolicyDistillationManager` |
| `v14_env_factory.py` | V14 环境工厂（环境构建与 track cache 查询）。 | `def _query_generated_track_cache_from_env`，`def create_v14_env_and_npcs` |
| `v14_paths.py` | V14 路径/配置文件定位工具。 | `def default_generated_track_profile` |
| `v14_racing_line.py` | V14 racing line 计算器。 | `class RacingLineComputer` |
| `v14_stage_profiles.py` | V14 分阶段配置与训练参数模板。 | `class StageConfigV14`，`class StageTrainProfileV14`，`class StageEvalGateV14`，`def apply_stage_train_profile_v14` |
| `v14_train.py` | V14 训练主流程（train_v14）。 | `def train_v14` |
| `v14_wrapper.py` | V14 主 wrapper（组合 reward/npc/spawn mixin）。 | `class GeneratedTrackV14Wrapper` |
| `v14_wrapper_npc.py` | V14 NPC 运行时 mixin（速度约束、重置、碰撞后处理）。 | `class V14NpcRuntimeMixin` |
| `v14_wrapper_reward.py` | V14 奖励 mixin（racing/proactive/dynamic/chaos 奖励与 step 后处理）。 | `class V14RewardMixin` |
| `v14_wrapper_spawn.py` | V14 出生点/重置 mixin（spawn、teleport、布局逻辑）。 | `class V14SpawnResetMixin` |
| `wrappers.py` | 通用观测/扰动 wrappers（转置、归一化、黄线、重置扰动、缩放）。 | `class GeneralizationWrapper`，`class TransposeWrapper`，`class NormalizeWrapper`，`class V9YellowLaneWrapper`，`class GTResetPerturbWrapper`，`class RGBResizeWrapper` |

## 维护约定

- `v14_*` 文件为 V14 线的模块化实现，优先在这些文件扩展新能力。
- `v14_dep_*` 文件为底层兼容依赖层，除非必要不做大范围改动。
- `__init__.py` 作为统一导出入口，新增公共符号时同步更新导出。
