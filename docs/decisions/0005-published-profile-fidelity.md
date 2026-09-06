# 0005: Explicit published profile versus historical source behavior

Date: 2026-09-06. Status: accepted under target-fidelity correction authorization
in the user's v5 migration brief; empirical evidence recorded in the active plan.

## Evidence and decision

Published checkpoint hash 119fa871... exposes the original dimensions and DDIM
settings after strict reference load. Its config sets pre_trained_encoder=False:
all scene weights are inside model.pt. Native construction disables auxiliary IO
then strictly loads every inference parameter; no random-filled parameter remains.

The public vv19 utils.subsample_pcd first voxel-downsamples at 0.01, unlike old ICGS
live preprocessing. First-denoiser pre-hooks show vv19 receives no precomputed scene
embeddings and draws action/gripper noise before scene FPS. Old ICGS pre-encodes
demo/live clouds before noise. Same seed therefore does not mean same random draws.

Add explicit runtime profile fields live_voxel_size and cache_context. Historical
source profile retains None/True. Published profile selects 0.01/False, so native
public inference matches reference preparation and initial-noise/encoder order.
No topology/normalization/formula change is inferred from tensor shapes.

## Consequences and proof

The published profile has a hash-bound packaged config, strict load, no sidecar
encoder/config requirement and full no-gradient inference. Historical and published
profiles are named separately, not silently equated. Re-run C3/C4 before claiming
fidelity; initial native differences (up to 0.0443 translation, differing grips)
were failures, not grounds for widening tolerance.
