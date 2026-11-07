# ADR-0001: Fresh paths for seedless environment resets

## Status

Accepted.

## Context

The environment pre-generates a complete price path at reset. Reusing the
constructor seed for every seedless reset caused repeated episodes unless a
wrapper supplied explicit seeds. That behavior is surprising for Gymnasium
training loops and can create accidental path reuse.

## Decision

`reset(seed=s)` remains deterministic and generates the path associated with
`s`. `reset(seed=None)` now draws a fresh path seed from the environment's
Gymnasium RNG. The first seedless reset still honours a constructor seed as the
RNG-stream seed, but subsequent seedless resets advance that stream.

Evaluation and baseline comparisons should pass explicit episode seeds. Training
wrappers may provide fresh seeds for seedless SB3 resets. Published default
evaluation results continue to use explicit shared seeds `0..n-1`.

## Consequences

- Seedless training episodes no longer repeat silently.
- Explicitly seeded reproducibility is preserved.
- Code that depended on repeated seedless paths must opt into explicit seeds.
