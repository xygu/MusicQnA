"""Fixed instruction strings — single source of truth for SFT and GRPO prompts."""

SYSTEM_OMNI = (
    "You are a music listening assistant. Describe only what you hear in the audio. "
    "Do not invent artist names, song titles, or album names."
)

USER_CAPTION_INSTRUCTION = (
    "Listen to the music and write one concise English paragraph describing how it sounds: "
    "instruments, rhythm, vocals (if any), mood, and production. "
    "Do not name the artist or song title."
)

# Mock (text-only) backend uses the same instruction without audio.
USER_MOCK_PREFIX = "[Audio: not used in debug mock mode]\n"
