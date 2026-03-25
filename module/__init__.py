# module package for ppo_waveshare_v12
from .utils import (
    load_config, ENV_DOMAIN_MAP, MONITOR_INFO_KEYS,
    _seed_everything, _safe_seed_env, _find_latest_checkpoint,
    _wrap_pi, _clip_float, _get_domain_for_env,
    _safe_float, _apply_global_seeds, _write_json,
)
from .track import SceneGeometry, TrackGeometryManager
from .reward import DonkeyRewardWrapper, ImprovedRewardWrapperV3, V9DomainRewardWrapper
from .control import (
    HighLevelControlWrapper,
    ActionSafetyWrapper, ThrottleControlWrapper, CurvatureAwareThrottleWrapper,
)
from .action_adapter import ActionAdapterWrapper
from .robust_lane_detector import RobustLaneDetector, RobustYellowLaneEnhancer
from .wrappers import (
    GeneralizationWrapper, TransposeWrapper, NormalizeWrapper,
    V9YellowLaneWrapper, GTResetPerturbWrapper,
    RGBResizeWrapper,
    CanonicalSemanticWrapper,
)
from .callbacks import (
    PTHExportCallback, CoverageLoggingCallback,
    PerSceneStatsCallback, PerDomainStatsCallback,   # PerDomainStatsCallback 为别名
    AdaptiveLearningRateCallback, TrainingMetricsFileLoggerCallback,
    BestModelCallback, DomainAwareBestModelCallback,  # DomainAwareBestModelCallback 为别名
    ShortEpisodeLoggerCallback, CrashRecoveryCallback,
)
from .multi_scene_env import (
    MultiSceneEnv, MultiSceneEnvV12, MultiSceneEnvV13,
    MultiInputObsWrapper,
    _build_v12_wrapper_chain,
    _build_state_v13,
)
# V14 modular components (generated-track line)
from .v14_attention import ChannelAttention, SpatialAttention, CBAMBlock, AttentionCNN
from .v14_racing_line import RacingLineComputer
from .v14_stage_profiles import (
    StageConfigV14, StageTrainProfileV14, StageEvalGateV14,
    STAGES_V14, STAGE_TRAIN_PROFILES_V14, STAGE_EVAL_GATES_V14,
    apply_stage_train_profile_v14,
)
from .v14_distillation import PolicyDistillationManager
from .v14_wrapper import GeneratedTrackV14Wrapper
from .v14_curriculum import CurriculumManagerV14
from .v14_callbacks import V14ControlTBCallback
from .v14_env_factory import _query_generated_track_cache_from_env, create_v14_env_and_npcs
from .v14_train import RecurrentPPO, train_v14
from .v14_cli import parse_args as parse_args_v14, main as main_v14
# All V8/V9 components unified here so v12 has no cross-file imports.
