# Labelling my own photos (LabelMe)

Labels for `data/self_batch_2/` (61 phone photos), used only for final testing.
The rules match the ADE20K label merge in `configs/labels.yaml`, so scores are comparable.

## Start

In the VS Code terminal (connected to the VM):

```bash
scripts/label_server.sh
```

Then open **http://localhost:6080/vnc.html?autoconnect=1&resize=scale** in your browser.
VS Code forwards the port automatically. If the page doesn't load, add port `6080` in the
**PORTS** tab. Press Ctrl+C in the terminal to stop.

Labels save automatically to `data/labels/self_batch_2/` (one `.json` per photo), so you can
stop and resume any time. The image list shows "N of 61 annotated".

## If it looks frozen

The virtual screen has no window manager, so a pop-up window can open off-screen or behind
LabelMe and wait there invisibly. The launcher turns off the label pop-up, which was the usual
cause. If it still happens, press **Esc** (closes most pop-ups) and try again. If the page shows
"Disconnected", reload it: LabelMe keeps running and nothing is lost.

## Labels

| Label | Draw it on | Becomes |
|---|---|---|
| `floor` | indoor floor, rugs/carpet, sidewalks, footpaths, paved paths | floor |
| `ignore` | stairs, steps, escalators, grass, dirt/earth, sand, fields, and anything you can't tell | left out of scoring |
| `not_floor` | things standing *on* a floor region that the floor outline swallowed (a bench, a person, a sign base) | not floor |

Everything you don't draw on counts as **not floor**: walls, buildings, sky, cars, and
**road and crosswalks** (the target user is a pedestrian, same as ADE20K).

Overlaps: `not_floor` is cut out of `floor`, and `ignore` wins over both.

## Per-photo checkboxes (Flags panel, top right)

- `no_floor`: tick for a photo with no floor at all. A photo with no `floor` shape and no
  tick is reported as unfinished, so nothing gets counted as floor-free by accident.
- `exclude`: tick to leave a photo out (blurry, duplicate, accidental).

## Workflow per photo

1. **Select the label first**: click `floor` (or `not_floor` / `ignore`) in the **Label List** panel
   on the right. Every shape you draw gets the highlighted label; there is no pop-up asking for it.
   Then pick **AI-Segment** in the left toolbar. The model menu at the top should say *SAM 2 Base+*.
   The first click on each photo takes a few seconds; later clicks are instant.
2. **Click** on the floor. Add more clicks on floor parts it missed. **Shift+click** on things
   it wrongly included. Or **drag** a box around the floor area.
3. Press **Enter** (or Ctrl+click the last point) to accept. If a shape got the wrong label,
   select it, press **Delete**, pick the right label and redraw it. (Avoid double-clicking a shape
   to rename it: that opens the label pop-up, which can land off-screen.)
4. Check the edges, zoomed in. To fix one, use **Select**, then drag the polygon's points.
   Plain **Polygon** (click around the outline, then click the first point to close) is fine for
   small areas the AI gets wrong.
5. For `ignore` or `not_floor` shapes, click that label in the Label List first, then draw; tick a flag if one applies, then **Next**.

## Edge cases

- **Curbs**: the flat sidewalk is floor; the vertical curb face and the road are not.
- **Rugs and mats** on a floor: floor (same as ADE20K).
- **People, bags, furniture legs** standing on floor: carve out large ones with `not_floor`;
  don't chase tiny gaps.
- **Shadows, puddles, reflections, wet patches** on a floor: still floor.
- **Glass doors, windows and glass railings**: not floor, and neither is anything seen *through*
  them (another walkway across an atrium, a lower level). The floor shape stops at the base of the
  glass. Floor seen directly, with no glass in front, is still floor even if far away; if you can't
  tell whether glass is in the way, use `ignore`.
- **Far-away floor** you can't make out clearly: `ignore`.
- **Ramps** you can walk on: floor. **Stairs**: ignore.

## After labelling

```bash
.venv/bin/python scripts/labelme_to_masks.py
```

This writes masks to `data/self_floor/masks/self_batch_2/` and the photo list to
`splits/self_batch_2.txt`. It lists any photo that still needs attention. Check the overlays in
`data/self_floor/preview/self_batch_2/` (green = floor, red = ignore) before evaluating.
