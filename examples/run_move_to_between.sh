#!/usr/bin/env bash
python flowmix_two_tracks.py "EDM1.wav" "EDM2.wav" \
  -o flowmix_two_track_demo.wav \
  --mode profile \
  --profile edm \
  --make-snippets
