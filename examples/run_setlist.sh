#!/usr/bin/env bash
python flowmix_setlist.py examples/setlist_example.json \
  -o flowmix_demo.wav \
  --apply-manifest-settings \
  --make-snippets
