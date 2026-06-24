# Changelog

All notable changes to FlowMix will be documented in this file.

## 1.0.0 - Initial public release

Initial public release of FlowMix.

### Added

- Continuous WAV mix rendering from ordered setlist manifests.
- Two-track transition rendering for auditioning individual handoffs.
- Profile-driven transition scoring with built-in TOML profiles:
  - `edm`
  - `vocal_trance`
  - `lounge`
  - `jazz`
  - `heart`
  - `cinematic`
- Transition candidate scoring using vocal safety, beat/onset evidence, energy, loudness, key/BPM compatibility, cue depth, and tail trimming.
- Optional Demucs vocal analysis with heuristic fallback.
- Per-transition manifest overrides, including manual timing overrides.
- Transition snippets and detailed JSON reports.
- MIT License.