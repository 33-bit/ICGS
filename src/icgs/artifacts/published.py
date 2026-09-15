"""Hash-bound published profile. No binary/reference checkout is needed at runtime."""

from dataclasses import dataclass, replace
from importlib.resources import files
import hashlib
import json
from pathlib import Path

from icgs.configuration.schema import ExperimentConfig


PUBLISHED_PROFILE_ID = "instant_policy_published_vv19_119fa871"
# Legacy compatibility mirror; packaged profile metadata is runtime authority.
PUBLISHED_SHA256 = (
    "119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5"
)
_PUBLISHED_PROFILE_RESOURCE = "profiles/vv19-119fa871.json"


@dataclass(frozen=True)
class _ResolvedNativeProfile:
    profile_id: str
    artifact_sha256: str
    native_point_count: int
    _base_config: ExperimentConfig

    def config_for(self, *, num_demos: int, device: str) -> ExperimentConfig:
        """Derive the published native config for an explicit demo count/device."""

        config = self._base_config
        return replace(
            config,
            name=self.profile_id,
            runtime=replace(
                config.runtime,
                device=device,
                compile_models=False,
                live_voxel_size=0.01,
                cache_context=False,
            ),
            graph=replace(config.graph, num_demos=num_demos),
            sampling=replace(config.sampling, steps=4),
        ).validate()


def resolve_native_profile(profile_id: str) -> _ResolvedNativeProfile:
    """Resolve the one supported hash-bound native profile without aliases."""

    if profile_id != PUBLISHED_PROFILE_ID:
        raise ValueError(f"unknown native profile: {profile_id!r}")

    from icgs.configuration.defaults import from_legacy

    payload = json.loads(
        files("icgs.artifacts").joinpath(_PUBLISHED_PROFILE_RESOURCE).read_text()
    )
    if payload.get("schema_version") != 1:
        raise ValueError("published native profile schema_version must equal 1")
    if payload.get("profile_id") != profile_id:
        raise ValueError("published native profile_id does not match its registry ID")

    artifact_sha256 = payload.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ValueError("published artifact_sha256 must be a canonical SHA-256")
    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("published preprocessing metadata must be an object")
    native_point_count = preprocessing.get("native_point_count")
    if type(native_point_count) is not int or native_point_count <= 0:
        raise ValueError("published native_point_count must be a positive integer")

    return _ResolvedNativeProfile(
        profile_id=profile_id,
        artifact_sha256=artifact_sha256,
        native_point_count=native_point_count,
        _base_config=from_legacy(payload["legacy_config"]),
    )


def _config_with_diffusion_steps(
    profile: _ResolvedNativeProfile,
    *,
    device: str,
    num_demos: int,
    diffusion_steps: int,
) -> ExperimentConfig:
    config = profile.config_for(num_demos=num_demos, device=device)
    return replace(
        config,
        sampling=replace(config.sampling, steps=diffusion_steps),
    ).validate()


def published_config(*, device="cuda", num_demos=2, diffusion_steps=4):
    """Return the convenience config for the supported published profile."""

    profile = resolve_native_profile(PUBLISHED_PROFILE_ID)
    return _config_with_diffusion_steps(
        profile,
        device=device,
        num_demos=num_demos,
        diffusion_steps=diffusion_steps,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_published_policy(
    checkpoint,
    *,
    native_profile=PUBLISHED_PROFILE_ID,
    config=None,
    device="cuda",
    num_demos=2,
    diffusion_steps=4,
):
    """Strictly load the verified published model; unsupported artifacts fail."""

    import torch
    from icgs.artifacts.checkpoints import load_state_dict_compatible
    from icgs.composition import build_policy

    profile = resolve_native_profile(native_profile)
    if config is None:
        config = _config_with_diffusion_steps(
            profile,
            device=device,
            num_demos=num_demos,
            diffusion_steps=diffusion_steps,
        )

    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        path = path / "model.pt"
    digest = sha256_file(path)
    if digest != profile.artifact_sha256:
        raise ValueError(f"unknown published checkpoint hash: {digest}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    policy = build_policy(config)
    policy.checkpoint_report = load_state_dict_compatible(
        policy.network,
        artifact["state_dict"],
        strict=True,
    )
    policy.eval()
    policy.network.requires_grad_(False)
    policy.artifact_sha256 = digest
    return policy
