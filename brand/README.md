# The mark

A hex sticker for the Interdisciplinary Index.

**`ii-hex.svg` is the mark.** The gate's two posts *are* the twin I,
so the letters and the picture are one object rather than letters with a picture
around them. A torii is a threshold — something you pass through — which is what
the tool measures: work crossing out of the field it came from.

The lintel is split **cool | warm**, the same pair the Knowledge Flow chart uses
for what a researcher cites against what cites them. The gate is where the two
meet. It is the only colour in the mark.

| File | Use |
|---|---|
| `ii-hex.svg` | **primary** |
| `ii-hex-mono.svg` | one ink — print, embroidery, stamps |
| `ii-hex-dark.svg` | dark ground |
| `ii-hex@1024.png`, `ii-hex@256.png` | raster |

Geometry is the R-community sticker proportion: pointy-top, width √3 : height 2.
At two inches tall that is the standard 51 × 59 mm sticker.

Palette is the app's own: ink `#14161A`, paper `#FAFAFB`, cool `#2A78D6`,
warm `#EB6834`.

Regenerate every variant and raster from the primary:

```bash
python3 - <<'PY'
s = open('ii-hex.svg').read()
open('ii-hex-mono.svg','w').write(
    s.replace('fill="#2A78D6"','fill="#14161A"').replace('fill="#EB6834"','fill="#14161A"'))
open('ii-hex-dark.svg','w').write(          # tokens, so neither swap eats the other
    s.replace('#FAFAFB','@P@').replace('#14161A','@I@')
     .replace('@P@','#14161A').replace('@I@','#FAFAFB'))
PY
for f in ii-hex ii-hex-mono ii-hex-dark; do
  qlmanage -t -s 1024 -o . "$f.svg" >/dev/null 2>&1 && mv "$f.svg.png" "$f@1024.png"
done
```
