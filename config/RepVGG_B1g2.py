from .RepVGG_B0 import module_config
from .groups import repvgg_b_g2_map
module_config.update({
    "a": 2,
    "b": 4,
    "groups": repvgg_b_g2_map
})
