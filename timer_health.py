# Designer: Scripting -> Gateway Events -> Timer -> new timer script
# Delay = 300000 ms (every 5 minutes).
#
# A camera is Online only if counter.py has posted recently. That proves the
# whole chain: camera/feed -> MediaMTX -> detection -> Ignition.
#
# TIP: while testing, set the delay to 10000 and MAX_AGE_SECONDS to 20, then put them back.

MAX_AGE_SECONDS = 120    # counter.py posts every second, so 2 min of silence = down

log = system.util.getLogger("CameraHealth")
now = system.date.now()

for result in system.tag.browse("[default]Cameras").getResults():
    cam = str(result['name'])
    base = "[default]Cameras/%s/" % cam
    if not system.tag.exists(base + "Online"):
        continue                          # not a camera folder (e.g. Alarm Metrics)

    last = system.tag.readBlocking([base + "LastUpdate"])[0].value
    online = last is not None and system.date.secondsBetween(last, now) <= MAX_AGE_SECONDS

    previous = system.tag.readBlocking([base + "Online"])[0].value
    system.tag.writeBlocking([base + "Online"], [online])

    if previous and not online:
        log.warn("%s went OFFLINE (last update: %s)" % (cam, last))
    elif online and not previous:
        log.info("%s is back ONLINE" % cam)
