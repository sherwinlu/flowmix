#!/usr/bin/env python3
"""FlowMix 1.0.1 WAV Setlist Builder.

Builds a continuous WAV mix from an ordered setlist of full-mix WAV masters.
Planning (analysis/scoring/selection) lives in flowmix_plan; this module handles
manifest loading, render/export, and CLI orchestration.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from pydub import AudioSegment

from flowmix_audio import format_timestamp, output_format_suffix, resolve_wav_export_subtype, validate_audio_output
from flowmix_plan import TrackSpec, plan_setlist_mix
from flowmix_rendering import export_audio, render_transition_tail, sec_to_ms
from flowmix_reports import build_setlist_report, candidate_verdict

APP_VERSION = "1.0.1"
APP_NAME = f"FlowMix {APP_VERSION} Setlist Builder"


@dataclass
class TransitionChoice:
    index: int
    from_title: str
    to_title: str
    selected_candidate: str
    score: float
    source_a_fade_start_sec: float
    source_a_cut_sec: float
    source_b_cue_sec: float
    overlap_sec: float
    b_gain_db: float
    takeover_overlap_sec: Optional[float]
    b_fade_in_sec: Optional[float]
    b_entry_gain_db: Optional[float]
    trim_a_tail_sec: float
    mix_transition_start_sec: float
    mix_transition_start_timestamp: str
    next_track_source_zero_in_mix_sec: float
    next_track_source_zero_in_mix_timestamp: str
    ranked_candidates: list[dict[str, object]]
    notes: List[str]


def safe_title_from_path(path: str) -> str:
    return Path(path).stem


def sanitize_filename(text: str, max_len: int = 50) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text).strip().replace(" ", "_")
    return (text[:max_len] or "track").strip("._-") or "track"


def load_setlist(path: str) -> Tuple[List[TrackSpec], dict[str, object]]:
    """Load a JSON manifest or a simple newline-delimited text setlist."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"Setlist file does not exist: {path}")

    settings: dict[str, object] = {}
    tracks_raw = None
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            tracks_raw = data.get("tracks")
            settings = data.get("settings", {}) or {}
        elif isinstance(data, list):
            tracks_raw = data
        else:
            raise ValueError("Setlist JSON must be an object with a 'tracks' list or a list of tracks.")
    else:
        lines = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        tracks_raw = lines

    if not isinstance(tracks_raw, list) or len(tracks_raw) < 2:
        raise ValueError("Setlist must contain at least two tracks.")

    base_dir = p.parent
    tracks: List[TrackSpec] = []
    for i, item in enumerate(tracks_raw, start=1):
        if isinstance(item, str):
            raw_path = item
            title = safe_title_from_path(item)
            bpm = None
            key = None
        elif isinstance(item, dict):
            raw_path = item.get("path") or item.get("file")
            if not raw_path:
                raise ValueError(f"Track #{i} is missing a path/file field.")
            title = item.get("title") or safe_title_from_path(raw_path)
            bpm = item.get("bpm")
            key = item.get("key")
        else:
            raise ValueError(f"Track #{i} must be a path string or an object.")

        track_path = Path(raw_path).expanduser()
        if not track_path.is_absolute():
            track_path = (base_dir / track_path).resolve()
        tracks.append(TrackSpec(path=str(track_path), title=str(title), bpm=float(bpm) if bpm is not None else None, key=key))

    return tracks, settings


def build_continuous_mix(args) -> None:
    if args.transition_mode != "profile" and (args.scoring_config is not None or args.profile != "edm"):
        print(
            "Warning: --profile/--scoring-config are only used with --transition-mode profile; ignoring profile settings for this run.",
            file=sys.stderr,
        )
    tracks, manifest_settings = load_setlist(args.setlist)
    if args.apply_manifest_settings:
        for key, value in manifest_settings.items():
            if hasattr(args, key) and getattr(args, key) in (None, build_parser().get_default(key)):
                setattr(args, key, value)

    out_path = validate_audio_output(args.output)
    base = out_path.with_suffix("")
    fmt_suffix = output_format_suffix(out_path)

    print(f"{APP_NAME}")
    print(f"Loaded setlist with {len(tracks)} tracks.")
    print("Chain mode: preserving source timing/pitch; one selected transition per junction.")

    mix_plan = plan_setlist_mix(
        tracks,
        manifest_settings=manifest_settings,
        transition_mode=args.transition_mode,
        profile=args.profile,
        scoring_config=args.scoring_config,
        search_a_sec=args.search_a_sec,
        search_b_sec=args.search_b_sec,
        max_trim_a_sec=args.max_trim_a_sec,
        b_cue_max_sec=args.b_cue_max_sec,
        vocal_method=args.vocal_method,
        prefer_mps=not args.no_mps,
    )

    infos = mix_plan.wav_infos
    segments = mix_plan.segments
    scoring_profile = mix_plan.scoring_profile
    source_subtype = resolve_wav_export_subtype(infos)
    input_formats = ", ".join(
        f"{Path(cast(str, info['path'])).suffix.lower()} ({info['subtype']})" for info in infos[:3]
    )
    if len(infos) > 3:
        input_formats += ", ..."
    print(f"Setlist decode: {infos[0]['samplerate']} Hz, {infos[0]['channels']} ch — inputs: {input_formats}")
    print(f"Output: {fmt_suffix} ({'MP3 320 kbps' if fmt_suffix == '.mp3' else source_subtype or 'PCM_16'})")
    if scoring_profile is not None:
        print(f"Using scoring profile: {scoring_profile.name} — {scoring_profile.description}")
    if mix_plan.effective_search_a_sec > float(args.search_a_sec):
        print(f"Note: widened Track A analysis window from {args.search_a_sec:.1f}s to {mix_plan.effective_search_a_sec:.1f}s.")
    if mix_plan.effective_search_b_sec > float(args.search_b_sec):
        print(f"Note: widened Track B analysis window from {args.search_b_sec:.1f}s to {mix_plan.effective_search_b_sec:.1f}s.")

    snippet_requests: List[tuple[int, str, float]] = []
    final = segments[0]
    transitions: List[TransitionChoice] = []

    for junction in mix_plan.junctions:
        i = junction.index - 1
        selected = junction.selected
        print(f"\n--- Transition {junction.index}/{len(tracks)-1}: {junction.from_title} → {junction.to_title} ---")
        if junction.override_mode:
            print(f"Using transition override for #{junction.index}: {junction.from_title} → {junction.to_title} = {junction.override_mode}")

        mix_transition_start_sec = junction.mix_transition_start_sec
        mix_transition_start_ms = int(round(mix_transition_start_sec * 1000))
        tail = render_transition_tail(segments[i], segments[i + 1], selected)
        final = final[:mix_transition_start_ms] + tail

        print(
            f"Selected {selected.name}: score={selected.score:.3f}; new song starts in mix at "
            f"{format_timestamp(junction.next_track_source_zero_in_mix_sec + selected.b_cue_sec)} "
            f"({junction.next_track_source_zero_in_mix_sec + selected.b_cue_sec:.2f}s), "
            f"using Track B cue {format_timestamp(selected.b_cue_sec)} ({selected.b_cue_sec:.2f}s); "
            f"overlap {selected.overlap_sec:.1f}s, B gain {selected.b_gain_db:+.1f} dB"
        )

        transitions.append(
            TransitionChoice(
                index=junction.index,
                from_title=junction.from_title,
                to_title=junction.to_title,
                selected_candidate=selected.name,
                score=selected.score,
                source_a_fade_start_sec=selected.a_fade_start_sec,
                source_a_cut_sec=selected.a_cut_sec,
                source_b_cue_sec=selected.b_cue_sec,
                overlap_sec=selected.overlap_sec,
                b_gain_db=selected.b_gain_db,
                takeover_overlap_sec=selected.takeover_overlap_sec,
                b_fade_in_sec=selected.b_fade_in_sec,
                b_entry_gain_db=selected.b_entry_gain_db,
                trim_a_tail_sec=selected.trim_a_tail_sec,
                mix_transition_start_sec=junction.mix_transition_start_sec,
                mix_transition_start_timestamp=format_timestamp(mix_transition_start_sec),
                next_track_source_zero_in_mix_sec=junction.next_track_source_zero_in_mix_sec,
                next_track_source_zero_in_mix_timestamp=format_timestamp(junction.next_track_source_zero_in_mix_sec),
                ranked_candidates=junction.ranked,
                notes=([f"transition override: {junction.override_mode}"] if junction.override_mode else [])
                + [candidate_verdict(selected)]
                + selected.notes,
            )
        )
        if args.make_snippets:
            snippet_requests.append((junction.index, f"{junction.from_title}_to_{junction.to_title}", mix_transition_start_sec))

    final_gain_db = 0.0
    if final.max_dBFS > -1.0:
        final_gain_db = round(-1.0 - final.max_dBFS, 3)
        final = final.apply_gain(final_gain_db)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_audio(final, out_path, source_subtype if fmt_suffix == ".wav" else None)

    snippet_outputs = []
    if args.make_snippets:
        snippet_dir = base.parent / f"{base.name}_transition_snippets"
        snippet_dir.mkdir(parents=True, exist_ok=True)
        for idx, label, start_sec in snippet_requests:
            start_ms = sec_to_ms(max(0.0, start_sec - 12.0))
            end_ms = sec_to_ms(min(len(final) / 1000.0, start_sec + 24.0))
            snip_name = f"transition_{idx:02d}_{sanitize_filename(label)}{fmt_suffix}"
            snip_path = snippet_dir / snip_name
            export_audio(
                cast(AudioSegment, final[start_ms:end_ms]),
                snip_path,
                source_subtype if fmt_suffix == ".wav" else None,
            )
            snippet_outputs.append(str(snip_path))

    report_path = base.with_suffix(".flowmix_1_0_0_setlist_report.json")
    report = build_setlist_report(
        app_name=APP_NAME,
        transition_mode=args.transition_mode,
        profile=getattr(scoring_profile, "name", None),
        scoring_config=args.scoring_config,
        vocal_method=args.vocal_method,
        output_file=str(out_path),
        output_duration_sec=round(len(final) / 1000.0, 3),
        output_duration_timestamp=format_timestamp(len(final) / 1000.0),
        final_gain_db=final_gain_db,
        wav_format_validation={"reference": infos[0], "tracks": infos},
        tracks=[asdict(t) for t in tracks],
        track_source_zero_in_mix_sec=[round(x, 3) for x in mix_plan.source_zero_in_mix_sec],
        transitions=[asdict(t) for t in transitions],
        transition_overrides=manifest_settings.get("transition_overrides"),
        snippet_files=snippet_outputs,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n✅ Wrote continuous mix: {out_path}")
    print(f"   Duration: {format_timestamp(len(final) / 1000.0)} ({len(final)/1000.0:.2f}s)")
    if final_gain_db:
        print(f"   Final peak guard applied: {final_gain_db:+.2f} dB")
    print(f"📝 Wrote setlist report: {report_path}")
    if snippet_outputs:
        print(f"🎧 Wrote {len(snippet_outputs)} transition snippets.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FlowMix 1.0.1 setlist builder")
    p.add_argument("setlist", help="JSON manifest or text file with one audio path per line")
    p.add_argument("-o", "--output", default="flowmix_setlist_1_0_0.wav", help="Output continuous mix (.wav or .mp3; MP3 exports at 320 kbps)")
    p.add_argument("--transition-mode", default="recommended", choices=["recommended", "vocal_safe", "beat_aligned", "quick_cut", "smooth", "profile", "vocal_ducked", "long_blend"], help="Which transition style to select at each junction")
    p.add_argument("--make-snippets", action="store_true", help="Export transition audition snippets from the final continuous mix")
    p.add_argument("--profile", default="edm", help="Built-in scoring profile for --transition-mode profile: edm, vocal_trance, lounge, jazz, heart, cinematic")
    p.add_argument("--scoring-config", default=None, help="Path to a TOML scoring profile; overrides --profile")
    p.add_argument("--vocal-method", default="heuristic", choices=["heuristic", "demucs", "none"], help="Vocal analysis method; 'none' maps to heuristic")
    p.add_argument("--no-mps", action="store_true", help="Disable Apple Silicon MPS for Demucs")
    p.add_argument("--search-a-sec", type=float, default=35.0, help="Seconds of each outgoing track outro to analyze")
    p.add_argument("--search-b-sec", type=float, default=35.0, help="Seconds of each incoming track intro to analyze")
    p.add_argument("--max-trim-a-sec", type=float, default=8.0, help="Maximum seconds allowed to trim from outgoing track tail")
    p.add_argument("--b-cue-max-sec", type=float, default=10.0, help="Latest allowed cue point in incoming track intro")
    p.add_argument("--apply-manifest-settings", action="store_true", help="Apply optional settings object from JSON manifest")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.vocal_method == "none":
        ns.vocal_method = "heuristic"
    try:
        build_continuous_mix(ns)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
