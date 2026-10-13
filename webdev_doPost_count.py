# Designer: Web Dev -> count -> HTTP Method "doPost" (Enabled). Replaces the old script.
#
# Body sent by counter.py once per second per camera:
#   {"camera": "cam1", "count": 4, "delta": {"CountIn": 2, "Cars": 2, "Trucks": 1}}
#
# "count" is vehicles in view right now (overwrites ObjectCount).
# "delta" values are ADDED to the running totals, so restarting the Python
# service never resets or double-counts anything.
#
# The reply lists any tag that could not be written, e.g.
#   {"ok": false, "failed": ["[default]Cameras/cam1/Motorcyles"]}

def doPost(request, session):
    data = request['postData']
    base = "[default]Cameras/%s/" % data['camera']

    if not system.tag.exists(base + "ObjectCount"):
        return {'json': {'ok': False, 'error': 'unknown camera: %s' % data['camera']}}

    paths = [base + "ObjectCount", base + "LastUpdate"]
    values = [data['count'], system.date.now()]

    delta = dict((k, v) for k, v in data.get('delta', {}).items() if v)
    if delta:
        names = list(delta.keys())
        current = system.tag.readBlocking([base + n for n in names])
        for name, qv in zip(names, current):
            paths.append(base + name)
            values.append((qv.value or 0) + delta[name])

    results = system.tag.writeBlocking(paths, values)
    failed = [p for p, r in zip(paths, results) if not r.isGood()]
    return {'json': {'ok': not failed, 'failed': failed}}
