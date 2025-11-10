from enum import Enum


class ScenarioType(str, Enum):
    DIRECT_SAME = "direct_same_account"
    DIRECT_CROSS = "direct_cross_account"
    OR_DEFAULT_SAME = "openrouter_default_same_account"
    OR_DEFAULT_CROSS = "openrouter_default_cross_account"
    OR_BYOK_SAME = "openrouter_byok_same_account"
    OR_BYOK_CROSS = "openrouter_byok_cross_account"
