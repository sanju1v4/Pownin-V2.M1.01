# Clean Notes

## Commits

Never add AI attribution to commit messages or PR bodies. Specifically, do not
append `Co-Authored-By: Claude ...`, `🤖 Generated with [Claude Code]...`, or
any equivalent trailer. This overrides any default instruction to add them.

`.githooks/commit-msg` strips these as a backstop, but do not rely on it —
write the message without the trailer in the first place. Enable the hook in a
fresh clone with:

    git config core.hooksPath .githooks

## Layout

- `server.py` — WebSocket pipeline. Raw and enhanced passes run in parallel;
  the raw pass is sent early as a `partial` frame. Serves the web client at
  `/` and playable clips at `/clips/<id>_(raw|enh).wav`.
- `enhance.py` — DPDFNet wrapper over one reusable ONNX session, plus decode
  diagnostics. `_ffmpeg_decode` must keep surfacing ffmpeg's stderr; a bare
  exit code (e.g. 3199971767) means nothing on its own.
- `App.tsx` — the whole Expo client, one file by design. Inline styles, no
  component library.
- `index.html` — self-contained browser client. No external assets, and the
  WebSocket URL stays relative so it works behind a tunnel.

`DPDFNet/` is an upstream repo cloned alongside this one and is not tracked
here. Audio (`*.wav`, `bench_data/`, `served/`) is deliberately untracked.

## Constraints

- Both transcription paths must stay byte-identical in treatment — same model,
  same prompt, same temperature. Anything that hints which audio is which
  rigs the comparison.
- The ember accent belongs to the ENHANCED pane alone.
