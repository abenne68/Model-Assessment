"""Line-crossing counter for tracked objects (pure Python, no dependencies).

Points are normalised (0..1) so the same line works at any resolution.
"""


class LineCounter:
    """Counts each tracked object at most once, when its centre crosses a line.

    The line runs from A to B. Crossing from the negative side of A->B to the
    positive side is "in"; the opposite direction is "out". Swap A and B in
    cameras.json to flip which direction is which.
    """

    PRUNE_EVERY = 300      # frames between clean-ups
    STALE_AFTER = 150      # frames without a sighting before a track is forgotten

    def __init__(self, a, b):
        self.ax, self.ay = a
        self.bx, self.by = b
        self.dx, self.dy = self.bx - self.ax, self.by - self.ay
        self.length_sq = self.dx * self.dx + self.dy * self.dy
        if self.length_sq == 0:
            raise ValueError("line endpoints must differ")
        self.last = {}         # track_id -> (point, frame_index)
        self.counted = set()   # track ids already counted
        self.frame_index = 0

    def _side(self, p):
        return self.dx * (p[1] - self.ay) - self.dy * (p[0] - self.ax)

    def update(self, track_id, point):
        """Feed one tracked point. Returns 'in', 'out' or None."""
        prev = self.last.get(track_id)
        self.last[track_id] = (point, self.frame_index)
        if prev is None or track_id in self.counted:
            return None
        prev_point = prev[0]
        s0, s1 = self._side(prev_point), self._side(point)
        if (s0 < 0) == (s1 < 0):
            return None
        # where the motion segment meets the infinite line, and is that on the segment A-B?
        u = s0 / (s0 - s1)
        ix = prev_point[0] + u * (point[0] - prev_point[0])
        iy = prev_point[1] + u * (point[1] - prev_point[1])
        t = ((ix - self.ax) * self.dx + (iy - self.ay) * self.dy) / self.length_sq
        if not 0.0 <= t <= 1.0:
            return None
        self.counted.add(track_id)
        return "in" if s0 < 0 else "out"

    def end_frame(self):
        """Call once per processed frame; forgets tracks that disappeared."""
        self.frame_index += 1
        if self.frame_index % self.PRUNE_EVERY:
            return
        cutoff = self.frame_index - self.STALE_AFTER
        stale = [tid for tid, (_, seen) in self.last.items() if seen < cutoff]
        for tid in stale:
            del self.last[tid]
            self.counted.discard(tid)
