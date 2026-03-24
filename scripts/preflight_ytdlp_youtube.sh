#!/usr/bin/env bash
# Preflight: Deno + EJS from GitHub + android,web client (no browser cookies).
# Run from repo root: bash scripts/preflight_ytdlp_youtube.sh [optional_video_url]
#
# If YouTube/GitHub time out (e.g. mainland network), set a local SOCKS proxy for yt-dlp + Deno:
#   export https_proxy=socks5h://127.0.0.1:13659
#   export http_proxy=socks5h://127.0.0.1:13659
# Or only for this script: MUSICCAPS_YTDLP_PROXY=socks5h://127.0.0.1:13659 bash scripts/preflight_ytdlp_youtube.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p .yt_dlp_smoke

export PATH="/usr/local/bin:/opt/homebrew/bin:${PATH}"

PROXY_ARGS=()
_p="${MUSICCAPS_YTDLP_PROXY:-}"
if [[ -z "${_p}" && -n "${https_proxy:-}" ]]; then _p="${https_proxy}"; fi
if [[ -z "${_p}" && -n "${HTTPS_PROXY:-}" ]]; then _p="${HTTPS_PROXY}"; fi
if [[ -n "${_p}" ]]; then
  PROXY_ARGS=(--proxy "${_p}")
  echo "using yt-dlp --proxy ${_p}"
fi
# Deno may fetch EJS deps; ensure standard proxy env if user only set MUSICCAPS_YTDLP_PROXY
if [[ -n "${MUSICCAPS_YTDLP_PROXY:-}" ]]; then
  export https_proxy="${https_proxy:-${MUSICCAPS_YTDLP_PROXY}}"
  export http_proxy="${http_proxy:-${MUSICCAPS_YTDLP_PROXY}}"
fi

if ! command -v deno >/dev/null 2>&1; then
  echo "error: deno not on PATH. Install with: brew install deno" >&2
  exit 1
fi

if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "error: yt-dlp not on PATH." >&2
  exit 1
fi

echo "deno: $(command -v deno) ($(deno --version | head -1))"
echo "yt-dlp: $(command -v yt-dlp) ($(yt-dlp --version))"

VIDEO="${1:-https://www.youtube.com/watch?v=jNQXAC9IVRw}"
OUT=".yt_dlp_smoke/preflight.%(ext)s"

echo "Downloading test audio -> ${OUT}"
# Longer timeouts / retries help SOCKS proxies (IncompleteRead on player JS, etc.)
yt-dlp "${PROXY_ARGS[@]:-}" --remote-components ejs:github --no-playlist \
  --socket-timeout 120 --retries 10 --fragment-retries 10 \
  --extractor-args "youtube:player_client=android,web" \
  -f "bestaudio/best" \
  -o "$OUT" \
  "$VIDEO"

echo "ok: wrote matching file(s) under .yt_dlp_smoke/"
