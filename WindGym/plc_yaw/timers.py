"""Scan-clock timer semantics. Repeated calls at one timestamp do not add time.

Inputs/PT are retained on calls without arguments, as in the supplied ST.
The model uses live PT and millisecond integer time, with no startup falling edge.
These choices require target-runtime trace validation before claiming PLC parity.
"""

import math


def milliseconds(seconds):
    """Positive REAL-to-TIME approximation, nearest ms, halves rounded upward."""
    return math.floor(seconds * 1000 + 0.5)


class Edge:
    def __init__(self, falling=False):
        self.previous = False
        self.falling = falling

    def __call__(self, signal):
        result = (
            self.previous and not signal
            if self.falling
            else signal and not self.previous
        )
        self.previous = bool(signal)
        return bool(result)


class TON:
    def __init__(self):
        self.signal = False
        self.pt = 0
        self.started = None
        self.q = False
        self.et = 0

    def __call__(self, now, signal=None, pt=None):
        if signal is not None:
            self.signal = bool(signal)
        if pt is not None:
            self.pt = max(0, pt)
        if not self.signal:
            self.started = None
            self.et = 0
            self.q = False
        else:
            if self.started is None:
                self.started = now
            self.et = min(max(0, now - self.started), self.pt)
            self.q = self.et >= self.pt
        return self.q


class TP:
    def __init__(self):
        self.signal = False
        self.previous = False
        self.pt = 0
        self.started = None
        self.q = False
        self.et = 0

    def __call__(self, now, signal=None, pt=None):
        if signal is not None:
            self.signal = bool(signal)
        if pt is not None:
            self.pt = max(0, pt)
        edge = self.signal and not self.previous
        self.previous = self.signal
        if self.started is not None:
            self.et = min(max(0, now - self.started), self.pt)
            if now - self.started >= self.pt:
                self.started = None
                self.q = False
        elif edge and self.pt > 0:
            self.started = now
            self.et = 0
            self.q = True
        elif not self.signal:
            self.et = 0
        return self.q
