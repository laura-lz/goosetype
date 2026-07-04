# GooseType

GooseType is a generative font engine that turns a folder of goose photos into various fonts. The same concept can be reapplied to other types of photos. 

The goal is for the following pipeline:

```text
raw goose photos
-> segmentation and cutouts
-> geometric feature extraction
-> glyph target comparison
-> 1-2 goose glyph composition
-> browser rendering
-> optional static font export
```

## Design Notes

GooseType treats each goose as a glyph candidate with measurable properties:

- aspect ratio and bounding box
- center of mass
- orientation / slant
- area and perimeter
- boldness and thinness
- curvature proxy
- left/right balance

Style presets work by changing matching weights. For example, an italic-ish family rewards slanted geese, while an abstract family tolerates looser glyph fit.
