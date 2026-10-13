# Multi-camera setup (PDA_TRACK)

```
camN feed --ffmpeg--> MediaMTX camN ----------------------------> Inline Frame (raw)
                          |
                          +--> counter.py (YOLO + tracking + line count)
                                   |-- annotated video --> MediaMTX camN_det --> Inline Frame (detections)
                                   '-- counts (HTTP POST) --> Web Dev "count" --> [default]Cameras/camN/* tags
```

Do the parts in order. Test after each one.

## Part 1. MediaMTX

1. Open `mediamtx.yml`, replace the `paths:` section with `mediamtx_paths.yml`.
2. cam1 already has the Ocean City feed. For cam2/cam3 put another `.m3u8` URL from opencctv.org,
   or point them at a local H.264 `.mp4` (looped) so you can test without more feeds.
   Use forward slashes in the path: `C:/Users/austi/Videos/traffic.mp4`.
3. Restart `mediamtx.exe`. Open `http://localhost:8888/cam1/` and check it plays.

## Part 2. Tags (UDT + 3 cameras)

1. Designer -> Tag Browser -> the `default` provider.
2. Use the kebab menu (three dots) -> **Import Tags**, pick `ignition/tags_import.json`,
   target = provider root. This creates the `CameraUDT` type and `[default]Cameras/cam1..cam3`.
3. Check: expand `Cameras/cam1`. It should hold ObjectCount, CountIn, CountOut, Cars, Trucks,
   Buses, Motorcycles, LastUpdate, Online.

If the import is refused, build it by hand: Tag Browser -> **UDT Definitions** -> add a type named
`CameraUDT` with those nine **Memory** tags (Int4 for the counts, DateTime for LastUpdate,
Boolean for Online). Then create a folder `Cameras` and add cam1..cam3 as instances of it.

The old `[default]Camera` folder can stay or be deleted. Nothing uses it anymore.

## Part 3. Web Dev endpoint

1. Web Dev -> `count` -> HTTP Method **doPost**.
2. Select all, delete, paste `ignition/webdev_doPost_count.py`. Ctrl+S.
3. Test:
   ```powershell
   Invoke-RestMethod -Method Post -Uri "http://localhost:8088/system/webdev/PDA_TRACK/count" -ContentType "application/json" -Body '{"camera":"cam1","count":3,"delta":{"CountIn":2,"Cars":2}}'
   ```
   Expect `ok: True`. `Cameras/cam1/ObjectCount` = 3, `CountIn` = 2, `Cars` = 2.
   Run it again: ObjectCount stays 3, but CountIn goes to 4 (deltas accumulate).
4. Unknown camera test: use `"camera":"nope"`. Expect `ok: False`.

## Part 4. Python service

```powershell
pip install -r requirements.txt
python counter.py --cam cam1 --show      # one camera with a debug window
python run_all.py                        # all cameras in cameras.json
```

- Needs: MediaMTX running, ffmpeg on PATH (already installed), project saved.
- Check `http://localhost:8888/cam1_det/` in a browser: the video with boxes and the yellow line.
- `cameras.json`: `line` is two points, each [x, y] from 0 to 1. `[[0,0.6],[1,0.6]]` is a horizontal
  line 60% of the way down. Move it to where traffic crosses. Swap the two points to flip in/out.
- `process_fps` (default 8) is how many frames per second each camera is analysed and published.
  Lower it if CPU is maxed. Higher helps tracking fast vehicles.

## Part 5. Perspective view

### 5a. View custom properties
Click the empty canvas outside the root container so the **view** properties show. Add custom props:

| name | type | value |
|---|---|---|
| `cam` | string | `cam1` |
| `det` | boolean | false |
| `mediaHost` | string | `localhost` (use the server's IP/hostname if viewers are on other PCs) |

### 5b. Inline Frame source
Bind `props.src` (Expression binding):
```
"http://" + {view.custom.mediaHost} + ":8888/" + {view.custom.cam} + if({view.custom.det}, "_det", "") + "/"
```

### 5c. Switch cameras
- **Dropdown**: `props.options` =
  ```json
  [{"label":"Ocean City","value":"cam1"},{"label":"Camera 2","value":"cam2"},{"label":"Camera 3","value":"cam3"}]
  ```
  Bind `props.value` to `view.custom.cam` as a **Property** binding with **Bidirectional** ticked.
- **Next camera button**: component Events -> `onActionPerformed` -> Add -> Script:
  ```python
  cams = ["cam1", "cam2", "cam3"]
  i = cams.index(self.view.custom.cam)
  self.view.custom.cam = cams[(i + 1) % len(cams)]
  ```
- **Toggle Switch** ("Show detections"): bind `props.selected` to `view.custom.det`, Bidirectional.

### 5d. Live stats (labels / LED)
Expression bindings on `props.text`:
```
"In view now: " + tag("[default]Cameras/" + {view.custom.cam} + "/ObjectCount")
"In: " + tag("[default]Cameras/" + {view.custom.cam} + "/CountIn") + "   Out: " + tag("[default]Cameras/" + {view.custom.cam} + "/CountOut")
"Cars " + tag("[default]Cameras/" + {view.custom.cam} + "/Cars") + " | Trucks " + tag("[default]Cameras/" + {view.custom.cam} + "/Trucks") + " | Buses " + tag("[default]Cameras/" + {view.custom.cam} + "/Buses")
```
Online indicator, on a shape or label `props.style.backgroundColor`:
```
if(tag("[default]Cameras/" + {view.custom.cam} + "/Online"), "#2ecc71", "#e74c3c")
```
(If `tag()` isn't in your expression function list, use a Tag binding with an indirect path instead.)

### 5e. Reset-totals button
```python
base = "[default]Cameras/%s/" % self.view.custom.cam
names = ["CountIn", "CountOut", "Cars", "Trucks", "Buses", "Motorcycles"]
system.tag.writeBlocking([base + n for n in names], [0] * len(names))
```

### 5f. Optional: auto-cycle every 10 seconds
Add custom props `auto` (boolean, false) and `shown` (string) with this Expression binding on `shown`:
```
if({view.custom.auto}, "cam" + (floor(toMillis(now(10000)) / 10000) % 3 + 1), {view.custom.cam})
```
Then use `view.custom.shown` instead of `view.custom.cam` in the Inline Frame expression and the stats labels,
and add a toggle bound to `auto`. Change `3` to your camera count.

### 5g. Optional: thumbnail grid
New view `CameraTile` (params: `cam`, `name`) with a small Inline Frame
(`"http://localhost:8888/" + {view.params.cam} + "/"`), a label with the count, and on click:
`system.perspective.sendMessage("selectCam", {"cam": self.view.params.cam}, scope="page")`.
On the main view add a message handler (view -> Scripting -> Message Handlers) named `selectCam`:
`self.view.custom.cam = payload['cam']`.
Put the tiles in a **Flex Repeater** with `instances` = one entry per camera. Keep tiles on the raw
streams. Several detection streams at once is heavy.

## Part 6. Health ping every 5 minutes

Scripting -> Gateway Events -> Timer -> new script, Fixed Delay `300000`, paste `ignition/timer_health.py`.
It sets each `Cameras/<cam>/Online` from how recently counter.py posted, and logs when a camera drops or recovers.
To test quickly: stop `counter.py`, wait ~2+ minutes plus a timer tick, and `Online` should go false.

### Alarm on offline
UDT Definitions -> `CameraUDT` -> `Online` -> Alarms -> add: mode Equality, setpoint `0` (false),
priority High, label "Camera offline". Then attach an alarm pipeline (email/SMS) if you want notifications.

### History and traffic charts
1. Gateway -> Config -> modules: install **Historian Core** and **SQL Historian** (both were on your download list),
   and add a database connection (Config -> Databases) plus a historian provider.
   Menu names vary slightly in 8.3.
2. In `CameraUDT`, on `ObjectCount`, `CountIn`, `CountOut`, `Cars`, `Trucks`, `Buses`, `Motorcycles`:
   enable History, pick the provider, sample mode "On Change".
3. Add a **Time Series Chart** (or **Easy Chart**) to a view and point it at those tags.
   Totals are cumulative, so for "vehicles per hour" chart the difference (aggregate "Difference"
   or use the Tag Historian's `Range`/delta settings).
