# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `batch::Verifier::verify` and `batch::Verifier::verify_multicore` are faster.
  One proof in each batch has weight one and every other proof has an independent
  uniform weight in `1..=2^128 - 1` (uniform nonzero weights for scalar fields of
  at most 128 bits). The verifier uses variable-time arithmetic, which operates on
  public proof data and on weights that are fresh for each batch and drawn after
  the batch is fixed. Valid batches are always accepted. For scalar fields wider
  than 128 bits, a batch containing an invalid proof is now accepted with
  probability at most `1 / (2^128 - 1)`, previously `1 / (r - 1)` for `verify` and
  `8 / (r - 1)` for `verify_multicore`, where `r` is the scalar field order (about
  `2^-254.9` and `2^-252` for BLS12-381). For fields of at most 128 bits, the bound
  is now `1 / (r - 1)` for both methods.

## [0.2.0] - 2026-09-26
### Added
- The `multicore` feature, which the crate's source already referenced but never
  declared. Without it `groth16::batch::Verifier::verify_multicore` (gated on
  that feature) could not be compiled by any consumer.

### Changed
- MSRV bumped to `1.85`.
- Bumped dependencies to `ff 0.14`, `group 0.14`, `rand_core 0.10`,
  `pairing 0.24`, `bellman 0.15`.

## [0.1.0] - 2024-07-15
Initial release (moved from `bellman::groth16`)
