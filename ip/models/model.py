"""Legacy AGI constructor; component math lives in GraphDenoiser."""
from ip.models.denoiser import GraphDenoiser


class AGI(GraphDenoiser):
    def __init__(self, config):
        from ip.composition import build_components
        from ip.configs.original import from_legacy, to_legacy
        from ip.actions import OriginalActionCodec
        resolved = from_legacy(config)
        parts = build_components(resolved)
        codec = OriginalActionCodec(resolved.action, parts['graph'].gripper_node_pos, resolved.runtime.device)
        super().__init__(resolved.graph, resolved.backbone, resolved.runtime, parts, codec)
        self.config = to_legacy(resolved)
        if resolved.runtime.compile_models:
            self.compile_models()
