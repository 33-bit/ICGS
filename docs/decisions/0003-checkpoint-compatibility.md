# 0003: Preserve state ownership and diagnose legacy checkpoint loading

Date: 2026-09-06. Status: accepted by the owner with the modularization plan.

## Context

Legacy GraphDiffusion registers self.model and aliases to its submodules. Module
nesting changes can create incompatible keys even when tensor shapes are unchanged.
Published model/encoder assets are not available locally.

## Decision

Preserve original learned parameter owners and leaf names where feasible. Retain
legacy import adapters and Lightning registration aliases. Normalize only explicit
legacy aliases and literal _orig_mod path segments; report every translation,
missing/unexpected key, unequal alias collision and shape mismatch. Strict loading
is default. Explicit legacy deployment non-strict mode reports mismatches, never
silently discards them. Unequal alias collisions and shape mismatches fail.

Loading never rewrites the source artifact. Trusted pickle/config and PyTorch
checkpoints are explicit IO at the outer boundary. Preserve legacy config.pkl
compatibility and record new resolved-composition metadata separately/additively.

## Alternatives and consequences

Broad string replacement or strict=False everywhere can hide scientific errors.
A duplicated legacy model implementation doubles maintenance. Stable ownership
plus a small diagnostic translator avoids those costs. Do not infer actual
published-checkpoint compatibility from generated fixtures alone.

## Compatibility implications

Weights-only load, Lightning optimizer resume, arbitrary whole-object pickle and
benchmark reproduction are separate claims. Real artifact/model/simulator tests
remain explicit integration requirements; unavailable proof must be disclosed.
