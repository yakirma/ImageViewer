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
#      the GitHub Actions workflow that builds Windows + Linux installers).
#   4. Builds the macOS DMG locally (runs in parallel with CI).
#   5. Creates the GitHub release with the DMG attached and release notes.
#      The CI workflow then attaches the .exe and .deb to the same release.
#   6. Optionally watches CI to completion and reports status.
#
# Timing (measured on the first dry run, 2026-05-03):
#   Local steps:        ~30 s (version bump, commit, push)
#   Local DMG build:    ~3 min
#   GitHub Actions:     ~5 min on a cold cache (52s test, 3m 44s Windows,
#                       2m 40s Linux — last two run in parallel).
#                       ~3-4 min on a warm cache.
#   Total wall-clock:   ~5-8 min until the release page has all 3 binaries
#                       (DMG ready locally same-time as CI finishes).

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

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "WARNING: this script builds the macOS DMG locally and assumes a Mac host."
    echo "         On other platforms, skip the DMG step and let CI do everything."
fi

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

step "Pushing to origin (this triggers the CI build for Windows + Linux)"
git push origin "$CURRENT_BRANCH"
git push origin "v$VERSION"

cat <<EOF

CI is now running. Measured wall-clock time:
  - Cold cache (first run / changed deps):  ~5 minutes
  - Warm cache (typical):                    ~3-4 minutes
  - Test job runs first (~1 min), then Windows + Linux builds run in parallel.

Watch progress at:
  https://github.com/yakirma/ImageViewer/actions

EOF

# ----------------------------------------------------------------------
# 4. Build macOS DMG locally (parallel with CI)
# ----------------------------------------------------------------------
if [[ "$(uname -s)" == "Darwin" ]]; then
    step "Building macOS DMG locally (~3 min)"
    python3 build.py
    DMG_PATH="dist/ImageViewer_Installer.dmg"
    if [[ ! -f "$DMG_PATH" ]]; then
        fail "DMG build failed: $DMG_PATH not found."
    fi
    DMG_SIZE=$(du -h "$DMG_PATH" | cut -f1)
    echo "DMG built: $DMG_PATH ($DMG_SIZE)"
else
    warn "Not on macOS — skipping local DMG build. CI will build Win+Linux only."
    DMG_PATH=""
fi

# ----------------------------------------------------------------------
# 5. Create the GitHub release
# ----------------------------------------------------------------------
step "Creating GitHub release v$VERSION"

if [[ -z "$NOTES" ]]; then
    # Auto-generate notes from commits since the previous tag.
    NOTES_ARG=(--generate-notes)
else
    NOTES_ARG=(--notes "$NOTES")
fi

if [[ -n "$DMG_PATH" ]]; then
    gh release create "v$VERSION" "$DMG_PATH" \
        --title "v$VERSION" \
        "${NOTES_ARG[@]}" \
        --latest
else
    gh release create "v$VERSION" \
        --title "v$VERSION" \
        "${NOTES_ARG[@]}" \
        --latest
fi

cat <<EOF

Release page: https://github.com/yakirma/ImageViewer/releases/tag/v$VERSION

The CI workflow will attach ImageViewer_Setup.exe and ImageViewer_Linux.deb
to this release as soon as the Windows and Linux build jobs finish.

EOF

# ----------------------------------------------------------------------
# 6. Optionally watch CI
# ----------------------------------------------------------------------
read -rp "Watch CI runs to completion now? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    # Find the run ID for the tag push and stream it.
    sleep 3 # give GitHub a moment to register the run
    RUN_ID=$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')
    if [[ -n "$RUN_ID" ]]; then
        gh run watch "$RUN_ID" || warn "Watch ended with a non-zero status."
    else
        warn "Could not locate the CI run; check the Actions tab."
    fi
fi

step "Done."
