#!/usr/bin/env bash
#
# Unified release flow.
#
# Usage:
#   scripts/release.sh <version> [--notes "release notes markdown"]
#   scripts/release.sh 1.1.2
#   scripts/release.sh 1.1.2 --notes "Fix Windows matplotlib import."
#
# What it does, in order:
#   1. Verifies the working tree is clean and tests pass.
#   2. Bumps the version across build.py, build.sh, installer.nsi,
#      ImageViewer.spec, and image_viewer.py.
#   3. Commits, tags v<version>, pushes to origin (the tag push triggers
#      the GitHub Actions workflow that builds Windows, Linux, and macOS
#      installers).
#   4. Creates the GitHub release with the supplied (or auto-generated)
#      notes. The CI workflow attaches the .exe, .deb, and .dmg to this
#      release as each build job finishes.
#   5. Optionally watches CI to completion and reports status.
#
# All three platform builds run in CI — no local build step is required,
# and this script is portable across any host with Python, git, and gh.
#
# Timing (measured 2026-05-03):
#   Local steps:        ~30 s (version bump, commit, push, release create)
#   GitHub Actions:     ~5 min on a cold cache
#                         (52s test, 3m 44s Windows, 2m 40s Linux,
#                          macOS expected ~3-5 min — runs in parallel
#                          with Windows and Linux)
#                       ~3-4 min on a warm cache.
#   Total wall-clock:   ~5-6 min from invocation to all 3 binaries on
#                       the release page.

set -euo pipefail

# ----------------------------------------------------------------------
# Args + sanity
# ----------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <version> [--notes \"markdown\"]" >&2
    exit 1
fi

VERSION=$1
shift

NOTES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --notes)
            NOTES=$2
            shift 2
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Version must be MAJOR.MINOR.PATCH (got: $VERSION)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------
# 1. Pre-flight: clean tree, tests pass, gh authed
# ----------------------------------------------------------------------
step "Pre-flight checks"

if ! git diff-index --quiet HEAD --; then
    fail "Working tree has uncommitted changes. Commit or stash them first."
fi

if ! command -v gh >/dev/null 2>&1; then
    fail "GitHub CLI 'gh' not found. Install with: brew install gh"
fi

if ! gh auth status >/dev/null 2>&1; then
    fail "GitHub CLI not authenticated. Run: gh auth login"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    warn "You are on '$CURRENT_BRANCH', not 'main'."
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    fail "Tag v$VERSION already exists. Pick a different version."
fi

step "Running test suite"
python3 -m pytest tests/ -q || fail "Tests failed — aborting release."

# ----------------------------------------------------------------------
# 2. Bump version across all 5 files
# ----------------------------------------------------------------------
step "Bumping version to $VERSION"

# build.py:           VERSION     = "1.1.0"
# build.sh:           VERSION="1.1.0"
# installer.nsi:      !define VERSION   "1.1.0"
# image_viewer.py:    __version__ = "1.1.0"
# ImageViewer.spec:   'CFBundleShortVersionString': '1.1.0',
#                     'CFBundleVersion': '1.1.0',
#
# We use perl -i (BSD/GNU compatible) instead of sed -i (which differs).

perl -i -pe 's/^(VERSION\s*=\s*")[^"]+(")/${1}'"$VERSION"'${2}/' build.py
perl -i -pe 's/^(\s*VERSION=")[^"]+(")/${1}'"$VERSION"'${2}/' build.sh
perl -i -pe 's/^(!define VERSION\s+")[^"]+(")/${1}'"$VERSION"'${2}/' installer.nsi
perl -i -pe 's/^(__version__\s*=\s*")[^"]+(")/${1}'"$VERSION"'${2}/' image_viewer.py
perl -i -pe "s/('CFBundleShortVersionString':\s*')[^']+(')/\${1}$VERSION\${2}/; s/('CFBundleVersion':\s*')[^']+(')/\${1}$VERSION\${2}/" ImageViewer.spec

# Verify the bumps actually happened.
echo "Verifying bumps:"
grep -nE "VERSION|__version__|CFBundle(Short)?Version" \
    build.py build.sh installer.nsi image_viewer.py ImageViewer.spec \
    | grep -i "$VERSION" || fail "Version bump verification failed."

# ----------------------------------------------------------------------
# 3. Commit, tag, push
# ----------------------------------------------------------------------
step "Committing and tagging v$VERSION"

git add build.py build.sh installer.nsi image_viewer.py ImageViewer.spec
git commit -m "Bump version to $VERSION"
git tag -a "v$VERSION" -m "Release $VERSION"

step "Pushing to origin (the tag push triggers CI for Win + Linux + macOS)"
git push origin "$CURRENT_BRANCH"
git push origin "v$VERSION"

# ----------------------------------------------------------------------
# 4. Create the GitHub release immediately
#
# Doing this right after the tag push (and before any build job finishes)
# guarantees softprops/action-gh-release in CI uploads to *this* release
# rather than auto-creating one with default name and notes.
# ----------------------------------------------------------------------
step "Creating GitHub release v$VERSION"

if [[ -z "$NOTES" ]]; then
    NOTES_ARG=(--generate-notes)
else
    NOTES_ARG=(--notes "$NOTES")
fi

gh release create "v$VERSION" \
    --title "v$VERSION" \
    "${NOTES_ARG[@]}" \
    --latest

cat <<EOF

Release page: https://github.com/yakirma/ImageViewer/releases/tag/v$VERSION

CI is now building all three installers in parallel. Measured timing:
  - Cold cache:  ~5 min total
  - Warm cache:  ~3-4 min total

Each platform's binary will appear on the release page as soon as its
build job finishes:
  - Linux  .deb   (~2-3 min after tag push)
  - macOS  .dmg   (~3-5 min)
  - Windows .exe  (~3-4 min)

Watch progress at:
  https://github.com/yakirma/ImageViewer/actions

EOF

# ----------------------------------------------------------------------
# 5. Optionally watch CI
# ----------------------------------------------------------------------
read -rp "Watch CI to completion now? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    sleep 3 # give GitHub a moment to register the run
    RUN_ID=$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')
    if [[ -n "$RUN_ID" ]]; then
        gh run watch "$RUN_ID" || warn "Watch ended with a non-zero status — check the Actions tab."
    else
        warn "Could not locate the CI run; check the Actions tab."
    fi
fi

step "Done."
