#!/usr/bin/env bash
# Fetches the two eval targets at pinned commits into a directory OUTSIDE this repo.
# Nothing fetched here is ever committed. Usage: scripts/fetch_eval_repos.sh [DEST]
set -euo pipefail

DEST="${1:-${TMPDIR:-/tmp}/codesage-eval}"
CODESAGE_SHA=d8633d582013e947eb2c70f86a8b2f34eba33459
HUMANIZE_URL=https://github.com/python-humanize/humanize
HUMANIZE_SHA=785e5dcc0d0308ad0dff3f6cc0faa7085ad0375b

REPO_ROOT="$(git rev-parse --show-toplevel)"
rm -rf "$DEST/codesage" "$DEST/humanize"
mkdir -p "$DEST/codesage" "$DEST/humanize"

# CodeSage's own history is the source: pinned to the commit before this feature, not HEAD,
# so the retrieval corpus does not change underneath the dataset.
git -C "$REPO_ROOT" archive "$CODESAGE_SHA" | tar -x -C "$DEST/codesage"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git -C "$work" init -q
git -C "$work" fetch -q --depth 1 "$HUMANIZE_URL" "$HUMANIZE_SHA"
[ "$(git -C "$work" rev-parse FETCH_HEAD)" = "$HUMANIZE_SHA" ] || {
  echo "humanize: fetched commit does not match $HUMANIZE_SHA" >&2
  exit 1
}
git -C "$work" archive FETCH_HEAD | tar -x -C "$DEST/humanize"

echo "Fetched to $DEST (index the src/ directories: $DEST/codesage/src, $DEST/humanize/src)"
