#!/bin/bash
# Fetches every set of sets.conf at its pinned commit into sets/NAME (shallow).
# The sets are third-party data under their own licenses: they are only read,
# never built or executed. Rerunning moves an existing set to its pinned commit.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
grep -v '^#' "$HERE/sets.conf" | while read -r name url commit; do
  [ -n "$name" ] || continue
  dir="$HERE/sets/$name"
  if [ ! -d "$dir/.git" ]; then
    mkdir -p "$dir" && git -C "$dir" init -q && git -C "$dir" remote add origin "$url"
  fi
  if [ "$(git -C "$dir" rev-parse -q --verify HEAD)" != "$commit" ]; then
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" -c advice.detachedHead=false checkout -q "$commit"
  fi
  echo "$name $(git -C "$dir" rev-parse --short=12 HEAD)"
done
