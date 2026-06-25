"""Transition planning: separate analysis/scoring from render/export."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydub import AudioSegment

from flowmix_profiles import ScoringProfile, load_scoring_profile
from flowmix_audio import (
    MAX_OVERLAP_SEC,
    AudioAnalysis,
    TransitionCandidate,
    analyze_audio,
    validate_loaded_segment_parity,
    validate_wav_input,
    validate_wav_pair_compatibility,
    wav_info,
)
from flowmix_scoring import choose_candidates, profile_search_parameters, score_candidate
from flowmix_reports import ranked_candidate_summary


@dataclass
class TrackSpec:
    path: str
    title: str
    bpm: Optional[float] = None
    key: Optional[str] = None


@dataclass
class TwoTrackMixPlan:
    """Analysis and candidate selection for a two-track mix."""

    track_a_path: Path
    track_b_path: Path
    track_a_wav_info: Dict[str, object]
    track_b_wav_info: Dict[str, object]
    track_a_analysis: AudioAnalysis
    track_b_analysis: AudioAnalysis
    seg_a: AudioSegment
    seg_b: AudioSegment
    candidates: List[TransitionCandidate]
    ranked: list[dict[str, object]]
    recommended_name: Optional[str]
    scoring_profile: Optional[ScoringProfile]
    effective_search_a_sec: float
    effective_search_b_sec: float


@dataclass
class JunctionPlan:
    """Candidate selection and mix timeline mapping for one setlist junction."""

    index: int
    from_title: str
    to_title: str
    track_a_analysis: AudioAnalysis
    track_b_analysis: AudioAnalysis
    candidates: List[TransitionCandidate]
    ranked: list[dict[str, object]]
    selected: TransitionCandidate
    override_mode: Optional[str] = None
    mix_transition_start_sec: float = 0.0
    next_track_source_zero_in_mix_sec: float = 0.0


@dataclass
class SetlistMixPlan:
    """Full setlist transition plan before stitching/render."""

    tracks: List[TrackSpec]
    junctions: List[JunctionPlan] = field(default_factory=list)
    scoring_profile: Optional[ScoringProfile] = None
    wav_infos: List[Dict[str, object]] = field(default_factory=list)
    effective_search_a_sec: float = 35.0
    effective_search_b_sec: float = 35.0
    source_zero_in_mix_sec: List[float] = field(default_factory=list)
    manifest_settings: dict[str, object] = field(default_factory=dict)
    segments: List[AudioSegment] = field(default_factory=list)


def _effective_search_windows(
    search_a_sec: float,
    search_b_sec: float,
    max_trim_a_sec: float,
    b_cue_max_sec: float,
    scoring_profile: Optional[ScoringProfile],
) -> tuple[float, float]:
    profile_trim_sec, profile_b_cue_sec, profile_overlaps, _ = profile_search_parameters(
        max_trim_a_sec, b_cue_max_sec, scoring_profile
    )
    max_overlap_sec = max(profile_overlaps) if profile_overlaps else MAX_OVERLAP_SEC
    effective_search_a_sec = max(float(search_a_sec), float(profile_trim_sec) + max_overlap_sec)
    effective_search_b_sec = max(float(search_b_sec), float(profile_b_cue_sec) + max_overlap_sec)
    return effective_search_a_sec, effective_search_b_sec


def filter_candidates_by_mode(candidates: List[TransitionCandidate], mode: str) -> List[TransitionCandidate]:
    if mode == "all" or not candidates:
        return list(candidates)
    filtered = [c for c in candidates if c.name == mode]
    return filtered if filtered else candidates[:1]


def select_candidate(candidates: List[TransitionCandidate], mode: str, transition_index: int) -> TransitionCandidate:
    if not candidates:
        raise ValueError(f"No valid transition candidates generated for transition #{transition_index}.")
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    if mode in ("recommended", "best"):
        return ranked[0]
    matches = [c for c in candidates if c.name == mode]
    if matches:
        return matches[0]
    fallback = ranked[0]
    print(
        f"Warning: transition #{transition_index} requested mode '{mode}' was not generated; "
        f"falling back to '{fallback.name}'.",
        file=sys.stderr,
    )
    return fallback


def transition_override(settings: dict[str, object], index: int, from_title: str, to_title: str) -> dict[str, object] | None:
    """Return a per-transition override object from manifest settings, if present."""
    overrides = settings.get("transition_overrides") or []
    if not isinstance(overrides, list):
        return None
    for item in overrides:
        if not isinstance(item, dict):
            continue
        if item.get("index") == index:
            return item
        if item.get("from") == from_title and item.get("to") == to_title:
            return item
    return None


def coerce_float_override(item: dict[str, object], key: str, fallback: float) -> float:
    value = item.get(key)
    if value is None:
        return fallback
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Invalid manual transition override value for {key}: {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manual transition override value for {key}: {value!r}") from exc


def apply_manual_transition_override(
    base: TransitionCandidate,
    item: dict[str, object],
    transition_index: int,
    track_a: AudioAnalysis,
    track_b: AudioAnalysis,
) -> TransitionCandidate:
    """Create a candidate using exact transition parameters from the manifest."""
    a_fade_start = coerce_float_override(item, "a_fade_start_sec", base.a_fade_start_sec)
    a_cut = coerce_float_override(item, "a_cut_sec", base.a_cut_sec)
    b_cue = coerce_float_override(item, "b_cue_sec", base.b_cue_sec)
    overlap = coerce_float_override(item, "overlap_sec", base.overlap_sec)
    b_gain = coerce_float_override(item, "b_gain_db", base.b_gain_db)

    if overlap <= 0:
        raise ValueError(f"Transition #{transition_index} manual override overlap_sec must be > 0. Got {overlap}.")
    if a_cut <= a_fade_start:
        raise ValueError(
            f"Transition #{transition_index} manual override requires a_cut_sec > a_fade_start_sec. "
            f"Got {a_cut} <= {a_fade_start}."
        )

    rescored = score_candidate(track_a, track_b, a_cut, b_cue, overlap, "candidate")
    trim_a_tail_sec = max(0.0, track_a.duration_sec - a_cut)
    trim_override = item.get("trim_a_tail_sec")
    if isinstance(trim_override, (int, float, str)):
        trim_a_tail_sec = max(0.0, float(trim_override))

    notes = list(rescored.notes) + [
        "manual transition override",
        f"manual a_fade_start_sec={a_fade_start:.3f}",
        f"manual a_cut_sec={a_cut:.3f}",
        f"manual b_cue_sec={b_cue:.3f}",
        f"manual overlap_sec={overlap:.3f}",
        f"manual b_gain_db={b_gain:+.2f}",
        "component scores recomputed for manual timing",
    ]
    if abs((a_cut - a_fade_start) - overlap) > 0.01:
        notes.append(
            "manual warning: a_cut_sec - a_fade_start_sec differs from overlap_sec; "
            "this is allowed but may not be a standard crossfade shape"
        )

    return TransitionCandidate(
        name=str(item.get("name") or item.get("mode") or "manual"),
        score=rescored.score,
        a_fade_start_sec=a_fade_start,
        a_cut_sec=a_cut,
        b_cue_sec=b_cue,
        overlap_sec=overlap,
        b_gain_db=b_gain,
        trim_a_tail_sec=trim_a_tail_sec,
        vocal_collision_score=rescored.vocal_collision_score,
        beat_alignment_score=rescored.beat_alignment_score,
        energy_score=rescored.energy_score,
        placement_score=rescored.placement_score,
        loudness_score=rescored.loudness_score,
        perceptual_loudness_score=rescored.perceptual_loudness_score,
        compatibility_score=rescored.compatibility_score,
        notes=notes,
        soft_duck_db=coerce_float_override(item, "soft_duck_db", base.soft_duck_db),
        soft_duck_target=str(item.get("soft_duck_target", base.soft_duck_target)),
    )


def validate_setlist_formats(tracks: Sequence[TrackSpec]) -> tuple[List[Dict[str, object]], List[AudioSegment]]:
    """Validate all WAVs and load decoded segments before expensive analysis."""
    infos: List[Dict[str, object]] = []
    segments: List[AudioSegment] = []
    first_info: Optional[Dict[str, object]] = None
    first_seg: Optional[AudioSegment] = None

    for idx, spec in enumerate(tracks, start=1):
        p = validate_wav_input(spec.path, f"Track {idx}")
        info = wav_info(p)
        if first_info is None:
            first_info = info
        else:
            mismatches = [(f, first_info[f], info[f]) for f in ("samplerate", "channels", "subtype") if first_info[f] != info[f]]
            if mismatches:
                details = "\n".join(f"  - {field}: first={a_val}, track {idx}={b_val}" for field, a_val, b_val in mismatches)
                raise ValueError(
                    "All setlist WAV formats must match. Refusing to chain tracks rather than silently resampling.\n"
                    + details
                )
        infos.append(info)

        seg = AudioSegment.from_file(str(p), format="wav")
        if first_seg is None:
            first_seg = seg
        else:
            validate_loaded_segment_parity(first_seg, seg)
        segments.append(seg)

    return infos, segments


def plan_setlist_mix(
    tracks: Sequence[TrackSpec],
    *,
    manifest_settings: dict[str, object] | None = None,
    transition_mode: str = "recommended",
    profile: str = "edm",
    scoring_config: Optional[str] = None,
    search_a_sec: float = 35.0,
    search_b_sec: float = 35.0,
    max_trim_a_sec: float = 8.0,
    b_cue_max_sec: float = 10.0,
    vocal_method: str = "heuristic",
    prefer_mps: bool = True,
    assembled_mix_len_ms: Optional[int] = None,
) -> SetlistMixPlan:
    """Analyze each junction and return a render-ready setlist plan."""
    settings = dict(manifest_settings or {})
    scoring_profile = load_scoring_profile(profile, scoring_config) if transition_mode == "profile" else None
    effective_search_a_sec, effective_search_b_sec = _effective_search_windows(
        search_a_sec, search_b_sec, max_trim_a_sec, b_cue_max_sec, scoring_profile
    )

    wav_infos, segments = validate_setlist_formats(tracks)
    source_zero_in_mix_sec: List[float] = [0.0]
    junctions: List[JunctionPlan] = []
    mix_len_ms = assembled_mix_len_ms if assembled_mix_len_ms is not None else len(segments[0])

    for i in range(len(tracks) - 1):
        a_spec, b_spec = tracks[i], tracks[i + 1]
        a_analysis = analyze_audio(
            a_spec.path,
            role="a",
            window_sec=effective_search_a_sec,
            manual_bpm=a_spec.bpm,
            manual_key=a_spec.key,
            vocal_method=vocal_method,
            prefer_mps=prefer_mps,
        )
        b_analysis = analyze_audio(
            b_spec.path,
            role="b",
            window_sec=effective_search_b_sec,
            manual_bpm=b_spec.bpm,
            manual_key=b_spec.key,
            vocal_method=vocal_method,
            prefer_mps=prefer_mps,
        )
        candidates = choose_candidates(
            a_analysis, b_analysis, max_trim_a_sec, b_cue_max_sec, scoring_profile=scoring_profile
        )
        override_item = transition_override(settings, i + 1, a_spec.title, b_spec.title)
        override_mode = None
        if override_item:
            override_mode = str(
                override_item.get("mode")
                or override_item.get("candidate")
                or override_item.get("transition_mode")
                or "manual"
            )
        selection_mode = override_mode or transition_mode
        base_selection_mode = "recommended" if override_mode == "manual" else selection_mode
        selected = select_candidate(candidates, base_selection_mode, i + 1)
        if override_item and (
            override_mode == "manual"
            or any(k in override_item for k in ("a_fade_start_sec", "a_cut_sec", "b_cue_sec", "overlap_sec", "b_gain_db"))
        ):
            selected = apply_manual_transition_override(selected, override_item, i + 1, a_analysis, b_analysis)

        ranked = ranked_candidate_summary(candidates)
        current_a_zero = source_zero_in_mix_sec[i]
        mix_transition_start_sec = current_a_zero + selected.a_fade_start_sec
        mix_transition_start_ms = int(round(mix_transition_start_sec * 1000))
        if mix_transition_start_ms < 0 or mix_transition_start_ms > mix_len_ms:
            raise ValueError(
                f"Transition #{i+1} maps outside the assembled mix: {mix_transition_start_sec:.3f}s. "
                "This usually means a prior transition mapping failed."
            )

        next_zero = mix_transition_start_sec - selected.b_cue_sec
        if next_zero < -1e-6:
            raise ValueError(
                f"Transition #{i+1} maps Track B source zero before the start of the assembled mix: "
                f"next_zero={next_zero:.3f}s, transition_start={mix_transition_start_sec:.3f}s, "
                f"b_cue={selected.b_cue_sec:.3f}s. Reduce --b-cue-max-sec or use a later Track A transition point."
            )
        source_zero_in_mix_sec.append(next_zero)

        junctions.append(
            JunctionPlan(
                index=i + 1,
                from_title=a_spec.title,
                to_title=b_spec.title,
                track_a_analysis=a_analysis,
                track_b_analysis=b_analysis,
                candidates=candidates,
                ranked=ranked,
                selected=selected,
                override_mode=override_mode,
                mix_transition_start_sec=round(mix_transition_start_sec, 3),
                next_track_source_zero_in_mix_sec=round(next_zero, 3),
            )
        )
        mix_len_ms = mix_transition_start_ms + max(0, len(segments[i + 1]) - int(selected.b_cue_sec * 1000))

    return SetlistMixPlan(
        tracks=list(tracks),
        junctions=junctions,
        scoring_profile=scoring_profile,
        wav_infos=wav_infos,
        effective_search_a_sec=effective_search_a_sec,
        effective_search_b_sec=effective_search_b_sec,
        source_zero_in_mix_sec=source_zero_in_mix_sec,
        manifest_settings=settings,
        segments=segments,
    )


def plan_two_track_mix(
    track_a: str,
    track_b: str,
    *,
    mode: str = "all",
    profile: str = "edm",
    scoring_config: Optional[str] = None,
    search_a_sec: float = 35.0,
    search_b_sec: float = 35.0,
    max_trim_a_sec: float = 8.0,
    b_cue_max_sec: float = 10.0,
    a_bpm: Optional[float] = None,
    b_bpm: Optional[float] = None,
    a_key: Optional[str] = None,
    b_key: Optional[str] = None,
    vocal_method: str = "auto",
    prefer_mps: bool = True,
) -> TwoTrackMixPlan:
    """Analyze both tracks, score candidates, and return a render-ready plan."""
    track_a_path = validate_wav_input(track_a, "Track A")
    track_b_path = validate_wav_input(track_b, "Track B")
    a_wav_info, b_wav_info = validate_wav_pair_compatibility(track_a_path, track_b_path)

    scoring_profile = load_scoring_profile(profile, scoring_config) if mode == "profile" else None
    effective_search_a_sec, effective_search_b_sec = _effective_search_windows(
        search_a_sec, search_b_sec, max_trim_a_sec, b_cue_max_sec, scoring_profile
    )

    seg_a = AudioSegment.from_file(str(track_a_path), format="wav")
    seg_b = AudioSegment.from_file(str(track_b_path), format="wav")
    validate_loaded_segment_parity(seg_a, seg_b)

    a_analysis = analyze_audio(
        str(track_a_path),
        role="a",
        window_sec=effective_search_a_sec,
        manual_bpm=a_bpm,
        manual_key=a_key,
        vocal_method=vocal_method,
        prefer_mps=prefer_mps,
    )
    b_analysis = analyze_audio(
        str(track_b_path),
        role="b",
        window_sec=effective_search_b_sec,
        manual_bpm=b_bpm,
        manual_key=b_key,
        vocal_method=vocal_method,
        prefer_mps=prefer_mps,
    )

    candidates = choose_candidates(
        a_analysis, b_analysis, max_trim_a_sec, b_cue_max_sec, scoring_profile=scoring_profile
    )
    if not candidates:
        raise ValueError(
            "No valid transition candidates were generated. Track A may be too short for the requested "
            "search/overlap settings, or the candidate limits are too restrictive."
        )

    candidates = filter_candidates_by_mode(candidates, mode)
    ranked = ranked_candidate_summary(candidates)
    recommended = ranked[0]["name"] if ranked else None

    return TwoTrackMixPlan(
        track_a_path=track_a_path,
        track_b_path=track_b_path,
        track_a_wav_info=a_wav_info,
        track_b_wav_info=b_wav_info,
        track_a_analysis=a_analysis,
        track_b_analysis=b_analysis,
        seg_a=seg_a,
        seg_b=seg_b,
        candidates=candidates,
        ranked=ranked,
        recommended_name=recommended,
        scoring_profile=scoring_profile,
        effective_search_a_sec=effective_search_a_sec,
        effective_search_b_sec=effective_search_b_sec,
    )
