# docs/

Recordings and long-form documentation for [Retrieval Autopsy](../README.md).

| file | tab | what it shows |
|---|---|---|
| `space.gif` | SPACE | a query crossing the embedding space — both retrieval legs firing, the surviving chunks reaching the model, the answer returning |
| `rag.gif` | RAG | nine pipeline stages executing with real timings, the lexical/semantic competition, candidates struck through as they lose |
| `eval.gif` | EVAL | ten trap probes plotted where their questions land, coloured by outcome as the suite runs |

All three recordings come from one capture of the live inspector at `teach` pace, running
`provider=groq` against the real 449-chunk index. Nothing is mocked or staged. Browser
chrome and the taskbar are cropped out — the crop is `crop=1280:576:0:104`, which also keeps
a bookmarks bar and a system tray out of a public repository.

Encoded at **15fps, native 1280px, the full 256-colour table, and `sierra2_4a` error
diffusion** — no `scale` filter at all.

Two cheaper settings were tried and rejected. Ordered (`bayer`) dithering at 144 colours is
about a third the size but leaves visible speckle through body text and cross-hatches the
soft gradients behind the point cloud. Downscaling to 960px then cost roughly as much
sharpness again as the palette did. The README displays these at `width="960"`, so the
1280px source is a 1.33x oversample and stays crisp on high-DPI screens.

The result is ~14 MB across the three, which is permanent weight in every clone. That is a
deliberate trade: these recordings are the only way a reader sees the thing this repository
is about, and a grainy one undersells it.

Committed rather than linked to an external host: an image host that expires takes the
README's only visual with it, and a repository that cannot show what it built is back to
asking the reader to take its word for it.

## Regenerating them

Source recordings are **not committed** — `docs/*.mp4` is gitignored. Git keeps every
version of a binary forever, so one re-record would permanently enlarge every clone.

Record the inspector, drop the video in here, and cut it:

```bash
pip install imageio-ffmpeg     # bundles a static ffmpeg; no system install
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")

# Find the segment boundaries first — a contact sheet of the whole capture, one frame
# per second, is faster than scrubbing:
"$FF" -i Recording.mp4 -vf "fps=1,scale=252:-1,tile=10x8" -frames:v 1 sheet.png

# Then cut. Palette-based encoding is what keeps 20s under two megabytes; a naive
# `-f gif` of the same clip is roughly 5x larger.
"$FF" -ss <start> -to <end> -i Recording.mp4 \
  -vf "crop=1280:576:0:104,fps=11,scale=860:-1:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=144:stats_mode=diff[p];\
[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  -loop 0 space.gif
```

Keep each GIF under ~3 MB. A README that takes ten seconds to load is a README nobody
scrolls, and the weight is permanent once committed.
