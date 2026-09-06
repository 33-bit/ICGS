"""Legacy mutable config export; new code uses instant_policy_original() sections."""
from ip.configs.original import instant_policy_original, to_legacy

config = to_legacy(instant_policy_original())
