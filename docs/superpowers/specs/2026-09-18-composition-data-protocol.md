# ICGS Composition Data Protocol

Status: approved design for the executable composition manifest and data gates.

## Goal

Freeze the composition, lineage, asset, seed, and acceptance decisions before
any scaled data generation. The protocol must distinguish a declared program
from an execution-certified program and must never turn skipped execution into
published training data.

## Catalog and split

The catalog is fixed before generation:

- Training: `T01` through `T20` (20 programs).
- Development: `V01` through `V04` (4 programs).
- Test: `P1` through `P4`, `G1` through `G4`, and `R1` through `R4` (12
  compositions).

The canonical skeletons are the entries in
`src/icgs/data/collection/programs.py`. The executable manifest must copy the
IDs and ordered semantic steps from that catalog; filenames and generated
episode IDs must not create new programs.

The split is episode- and lineage-closed. A seed and every descendant remain
in one split. Test compositions are locked before any test episode is
generated. The initial asset-family allocation is deterministic and recorded
by family ID: train 70%, development 15%, test 15%. An asset family cannot
appear in more than one split. A physical anchor retains its original split
when used in a context swap.

## Executable binding contract

Every manifest entry has these fields:

`program_id`, `split`, `family`, `scene_id`, `asset_family_id`,
`asset_version`, `source_lineage_id`, `workspace`, `randomization`,
`seed_demos`, `execution_modes`, `controller`, `predicates`, and
`execution_status`.

`execution_status` is one of `planned`, `generation_authorized`,
`pilot_certified`, or `generation_certified`. `generation_authorized` permits
the bounded full-generation run but does not claim any successful episode.
A `planned` entry may be used for catalog validation,
but no episode under that ID may be published as measured data. The existing
G2 runs for `T06`, `T08`, `T09`, `T11`, `T13`, and `T14` remain historical pilot
references and do not alter the 20/4/12 catalog. They used a looser pilot
predicate/controller identity, so they are not automatically `pilot_certified`
under this stricter approved manifest; each must be rerun or explicitly
re-certified before entering the primary track.

`workspace` is an explicit meter-scale axis-aligned box. Randomization ranges
are explicit closed intervals for object translation, yaw, size scale,
camera pose, light intensity, and waypoint perturbation. Values outside a
declared range invalidate the episode.

## Seeds and execution modes

Each training program declares exactly five stable seed IDs for the initial
seed bank. The seed ID determines the reset, action, and variation seeds and
is persisted in every descendant episode. Programs with two valid expert
routes declare two execution modes; otherwise they declare one scripted
waypoint mode. A descendant episode must record the seed ID, mode ID, resolved
randomization values, and generator revision.

## Controller and predicates

The default controller identity is `rlbench-timed-ik-v1`, with quaternion
hemisphere continuity and joint-branch continuity enabled. Per-program
overrides must be explicit. Predicate tolerances are frozen per binding and
must include the required relation, hold duration, support/release condition,
and unsafe-terminal limits. Force or penetration predicates may only be used
when the exact simulator exposes the measured signal.

## Generation and acceptance

The target for a generation-certified training program is 200 successful
contexts. Failed attempts with valid observations are retained in a quarantine
record with `failure_reason`, `first_failed_interval`, controller status, and
the source manifest digest. Failed or skipped attempts never count as
successful contexts.

An episode is publishable only when:

1. its manifest digest matches the approved composition manifest;
2. its provenance is measured and contains the fixed seed/mode metadata;
3. all required archive checksums and array schemas validate;
4. controller statuses and task predicates meet the binding contract; and
5. the immutable storage commit and readback receipt exist.

The initial implementation may generate only programs with a matching approved
execution receipt. For all other programs it must produce a precise blocker
report rather than synthetic episodes.

## Ownership and agent boundary

Protocol decisions belong to the repository owner. Delegated agents may only
serialize this approved manifest, run validators, execute explicitly named
pilot batches, and collect receipts. They may not change IDs, splits, assets,
seed policy, tolerances, or acceptance rules without a new approved design.
