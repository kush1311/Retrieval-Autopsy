# docs/

Screenshots and recordings embedded in the top-level [README](../README.md).

Committed rather than linked to an external host: an image host that expires takes the
README's only visual with it, and a repository that cannot show what it built is back to
asking the reader to take its word for it.

| file | tab | what it shows |
|---|---|---|
| `space.gif` | SPACE | a query crossing the embedding space, both retrieval legs firing, the surviving chunks reaching the model, the answer returning |
| `rag.png` | RAG | nine pipeline stages with real timings, the lexical/semantic competition, a grounded answer |
| `eval.png` | EVAL | ten trap probes plotted where their questions land, coloured by outcome |

All three were cut from one screen recording of the live inspector at `teach` pace, running
`provider=groq` against the real 449-chunk index. Nothing is mocked or staged.

## Regenerating them

The source recording is **not committed** — `docs/*.mp4` is gitignored. Git keeps every
version of a binary forever, so a single re-record would permanently double the clone size
for a file nobody reads.

Record the inspector, drop the video in here, and cut it with ffmpeg:

```bash
pip install imageio-ffmpeg     # bundles a static ffmpeg; no system install
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")

# the GIF: palette-based, which is what keeps a 17s capture under a megabyte
"$FF" -ss <start> -t <seconds> -i Recording.mp4 \
  -vf "fps=12,scale=900:-1:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=128:stats_mode=diff[p];\
[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 space.gif

# a still
"$FF" -ss <seconds> -i Recording.mp4 -frames:v 1 -vf "scale=1440:-1:flags=lanczos" rag.png
```

Keep the GIF under ~2 MB and each still under ~500 KB. A README that takes ten seconds to
load is a README nobody scrolls.
