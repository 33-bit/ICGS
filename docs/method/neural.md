# Planned neural components and forward passes

Status: target specification, not an implemented model. **PR** denotes the archived
[proposal](../proposals/ICGS_proposal_English.tex), Shared Network Blocks through
Evaluator for the Remaining Task. **ID** names
deterministic implementation resolutions. Shapes exclude B unless stated; online
B=1. [Contracts](contracts.md) owns state types and units. Losses/freeze phases are
in [data and training](data-training.md).

## Shared blocks

**PR:** biased Linear MLPs with GELU between layers, linear final layer unless a
sigmoid/softmax is named. Transformers: width 256, 8 heads of width 32, FFN 1024,
pre-LN epsilon 1e-5, dropout 0. No sharing between distinct blocks unless specified.

```text
A_h = softmax(LN(Q) Wq_h (LN(K) Wk_h)^T / sqrt(32) + B_h + key_mask)
U   = Q + concat_h(A_h LN(K) Wv_h) Wo
Block(Q,K) = U + MLP_256_1024_256(LN(U))
B_h(i,j) = MLP_3_32_8((x_i-x_j)/ell0)[h]  # geometry blocks only
```

Invalid keys receive -infinity, padding is never pooled. **ID:** zero invalid
query rows before attention/LayerNorm and after each block. An all-invalid padded
anchor's masked neighbor max must be replaced by zero before projection/attention,
not left at negative infinity (which would produce NaN through LayerNorm).
Reject a sample with no valid geometry keys; mixed
nongeometry tokens ensure other attention sets are nonempty. Masked means divide
by valid count, not padded length. Linear layers use PyTorch default initialization
unless a head-specific override is below. GRU uses reset-after-hidden-affine form:
`n=tanh(Wn v+bn+r*(Un h+cn)); h'=(1-z)*n+z*h`, with sigmoid update/reset gates.

## P03: geometry encoder and observation bridge

**PR:** fixed workspace crop, 5 mm voxel filtering, 2048 points; FPS 128 anchors;
32 nearest neighbors per anchor. Coordinates remain metric; ell0=1 m. Encode:

```text
X0[i] = max_neighbor MLP_6_64_128_256([P[j]/ell0-x[i]/ell0; x[i]/ell0])
X = two_geometry_blocks(X0, x/ell0)       # 128 x 256
```

**ID:** voxel keys are floor(P/0.005); average valid points per voxel; order voxel
keys lexicographically. FPS begins at the lexicographically first point, then
selects maximum nearest-selected squared distance, breaking ties by ordered index.
kNN sorts squared distance then index. Repeated padding points/anchors are invalid
for pooling, losses and decoder output; repeat coordinates only to satisfy fixed
storage shapes. If fewer than 32 valid neighbors exist, repeat with invalid masks;
the max aggregation ignores repeats. These rules are for added components only.
Document device numerical-tie limitations, and test permutations away from ties;
do not claim exact continuous rotation equivariance.

Augment by yaw about calibrated gravity, applying the same transform to cloud,
measured pose and all command targets. Do not independently rotate cloud. Crop is
declared before collection; calibrated gravity is transformed consistently.

**PR decoder:** each valid anchor emits 16 patch points:

```text
P_hat[i,r] = x[i] + 0.10 * tanh(MLP_258_256_128_3([LN(X[i]); grid[r]]))
```

**ID:** grid is row-major Cartesian product of {-1,-1/3,1/3,1} squared. Discard
invalid anchors; retain a valid-point mask before any fixed-size resampling.
At imagined nodes pass world cloud plus predicted world pose to existing public
IP preprocessing. That path already transforms into the end-effector frame.
Do not transform the cloud in the bridge and then again in IP.

Measure paired observed/reconstructed-real actions under identical diffusion seeds:
translation discrepancy, geodesic rotation discrepancy, grip mismatch, local
executed success, and CD. A geometry-only loss cannot certify policy equivalence.
Visibility/disocclusion, sparse clouds and native sampling drift are separate
diagnostics. **FG:** approve action-drift/local-success tolerances on development
pilot data before authorizing imagined-policy rollout. Do not borrow C4's
native-vs-reference tolerance as an autoencoder acceptance threshold.

## P04: physical representation and memory

**PR:** p is translation3 + Rot6(ordered first two columns) + grip1 + gravity3 =13.
The action descriptor u is an SE(3) twist6 + commanded grip1 + log-duration1 =8.

```text
v = MLP_277_256_256([masked_mean(X); p; u_previous])
m1_next = GRU_256(v, m1)
m2_next = GRU_256(m1_next, m2)
S = [X; MLP_13_256_256(p); m1_next; m2_next] + type_embeddings
```

S has 131 rows: 128 geometry, one proprioception and two memory rows. Reset
memories and u_previous are zero. Real updates occur once per executed interval;
imagination clones root memory and uses predicted observations/recorded commands.
**ID:** history construction for an anchor includes the boundary observation once;
the first unroll uses its stored state, not a second update. PhysicalState origin
is enforced when accepting a real-history update.

## P05: segmentation and event encoder

**PR:** retain first/last frame. Mark observed grip changes after two-frame
debounce. Between grip boundaries insert a boundary after cumulative translation
exceeds 0.12 m, rotation exceeds 30 degrees, or 20 intervals. Merge segments shorter
than three intervals into neighboring segments except across protected grip
boundaries. Cap at 30 interactions plus two landmarks per demo; merge shortest
adjacent non-grip segments. If protected events cannot fit, emit overflow, never
truncate the suffix. Stress cap 64 is a separately named configuration.

**ID:** stable grip change at frames t-1,t is marked at t (confirmation, not
backdated); initial grip is frame0. Cumulative motion sums consecutive translation
norms/geodesic angles. Evaluate strict spatial thresholds and >=20 time threshold
at each frame. Boundaries form shared-endpoint [a,b] intervals. Process short
segments shortest-first; choose neighbor with shorter duration, then earlier
start index. For cap merges choose adjacent pair of minimum combined duration,
then earlier start; never erase a protected grip boundary. If no legal merge
exists, overflow. Landmarks are a=b=first/last; one-frame demos cannot supply an
action window and fail the successful-context precondition.

```text
d_l = MLP_269_256_256([masked_mean(X_l); p_l])
xi_ab = Scale_ell0(Log(inv(T_a) @ T_b))                 # 6
e_j = MLP_776_512_256([d_a; d_b; mean_valid(d_a:b); xi_ab; g_a; g_b])
order_j = MLP_2_256_256([j/J_n; 1/J_n])
M_C = two_self_attention_blocks(e + order + landmark_type)
```

**ID:** J_n counts all retained tokens, indexed j=0..J_n-1 including landmarks.
No demo-ID embedding or global concatenation index. Concatenate demos for
attention, shuffle demo order during training, preserve SegmentRef mapping under
permutation. Multi-demo output is permutation-compatible; pooled evaluation is
invariant within numerical tolerance, not required bitwise across attention kernels.

## P06: task tracker and router

**PR:** initialize r=0 at reset, then:

```text
z0 = r_previous + Ws masked_mean(S)
z1 = Block(z0, [S; M_C]); z2 = Block(z1, [S; M_C])
r = GRU_256(z2, r_previous)
alpha = softmax([(Wq r)^T Wk M_j / sqrt(256), null_linear(r)])
[rho_j,nu_j,eligible_j] = sigmoid(MLP_768_256_3([M_j;r;M_j*r]))
```

All attention/head outputs respect event validity. No monotonic alignment clamp.
Historical occurrence may remain true while a current relation is false. The
eligible head predicts prerequisites, not execution success.

For valid non-landmark windows let `v_j=(alpha_j+1e-6)*eligible_j`. Normalize over
valid windows. If their sum is below 1e-6, choose full context with probability1;
otherwise choose full context with probability0.5 and a window with probability
0.5*v_j/sum(v). Invalid windows have v=0 before normalization.

**PR:** window = event plus immediate previous/next interaction in the same demo,
keeping endpoints and all gripper transitions, then uniform filling to native
waypoint count10. Mandatory frames exceeding10 invalidate the window. **ID:**
deduplicate mandatory indices; fill remaining slots by evenly spaced ranks in
unused indices, ties earlier, then chronological sort. Too few unique frames
invalidates a window; no repeated pseudo-demonstration. Full context retains the
native preprocessing algorithm and errors if its count is invalid.

Use an explicit one-demo native configuration for a one-demo window and one/two
demos for full contexts. Sequential owner-bound sessions may share frozen weights
only through an explicit composition adapter with correctly reset graph scratch;
no concurrent calls or mutable graph_config edits. **ID default:** independently
constructed native sessions for D=1/D=2, loaded from identical weights once per
method instance (not per branch). This trades resident memory for clear ownership;
record memory cost. Optimize sharing only with separate equivalence tests.

Reference RNG protocol separates route and diffusion seeds; persist both. Preserve
native sampling RNG order within each call and restore scoped global RNG on error.

## P07: physical dynamics and closed rollout

**PR:** concatenate S with `MLP_8_256_256(u)+action_type`, giving132 rows. Four
self-attention blocks produce Y. Three heads share the trunk but not output MLPs:

```text
[delta_X_i; delta_x_i] = MLP_head_256_256_259(Y_i)
[delta_xi; grip_logit] = MLP_head_256_256_7(Y_proprio)
X_tilde = X + delta_X; x_tilde = x + ell0*delta_x
T_next = T_current @ Exp(Scale_inverse(delta_xi))
g_next = (grip_logit >= 0)
P_next = decode_cloud(X_tilde,x_tilde)
encoded_next = encode_cloud(P_next)
b_next = physical_update(encoded_next, predicted_pose_grip, u_before, current_memory)
q_next = track_task(q_current,b_next,M_C)                # outside physical F
```

**ID:** residual final weights normal std1e-4 and zero biases; apply only geometry
259 outputs and pose6 outputs, not the grip-logit row. Record initialization seed.
Use the same head ID for the entire rollout. Cache P_next before re-encoding; do
not decode re-encoded features a second time for IP. Preserve full common absolute
command, recomputing u per pose hypothesis. FPS/kNN indices and hard grip have no
gradient; continuous encoder/decoder operations may transmit input gradients when
their parameters are frozen. `eval()` and `requires_grad=False` are not `no_grad()`.

## P09: continuation, completion, progress and terminal hazards

**PR:** event keys `K_j=M_j+MLP_4_256_256([alpha_j,rho_j,nu_j,eligible_j])` retain
all valid events, including completed ones. Three separate single-query streams
share three attention block weights, but have no query-to-query attention:

```text
Qv = e_v + Wr*r + MLP_2_256_256([H/Hmax, log1p(H)/log1p(Hmax)])
Qs = e_s + Wr*r; Qphi = e_phi + Wr*r
Qk = three_blocks(Qk, [S; K; r])
logit_k = MLP_k_256_128_1(LN(Qk))
V = sigmoid(logit_v / Tv) if H>0 else 0
S_complete = sigmoid(logit_s / Ts); Phi = logit_phi
```

Hmax512 primary; train separate coverage for1024. **ID:** expose uncalibrated and
calibrated outputs together. Completion/progress must be numerically unchanged by
H when all other inputs remain fixed. Do not force values equal for same-goal demos
whose induced reference behavior differs; check nondecreasing value versus H as a
calibration diagnostic, not an unsupported exact architecture guarantee.

First-terminal predictor input is1288 = five pooled256 vectors + action8:
`[mean S_before; r_before; mean S_after; r_after; mean M_C; u]`.
MLP1288→512→256→3 produces mutually exclusive success/failure/continue logits.
Training after-state is one executed transition; search after-state is its own
prediction. Both require the before-state to be active. Simultaneous first success
and failure labels prefer failure; previously absorbed success stays absorbed.
Inference probabilities are `softmax(raw_event_logits / Te)` using the fitted
event-temperature artifact. D2 cross-entropy consumes raw logits, not divided
logits or probabilities. Every search trace records the temperature identity.

## Numerical validation

**PR:** geometry, rotations, losses, probabilities, search selection/widening/backup
use FP32 or higher. Mixed-precision attention is opt-in only after numerical audit.
Training geodesic arccos clips interior with margin1e-6; reported metric uses exact
angle and returns zero at identity. Test small rotations, near-pi rotations,
nonfinite input rejection and gradient finiteness. Do not change native numerical
quirks under the guise of implementing these new helpers.
