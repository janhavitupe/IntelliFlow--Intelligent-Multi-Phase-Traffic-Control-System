"""
yolo_generator.py

PLACEHOLDER - Not yet implemented.

Future traffic source backed by YOLO vehicle detection + OpenCV.
Will convert detected vehicles from camera feeds into spawn tuples using
the same BaseTrafficSource interface, so the scheduler needs no changes.
"""
from .base_source import BaseTrafficSource


class YoloTrafficSource(BaseTrafficSource):
    """
    Placeholder for YOLO/OpenCV-based vehicle detection.

    TODO: Integrate ultralytics YOLO, process camera frames, count and
    classify vehicles per approach/lane, and emit Vehicle spawns.
    """

    def __init__(self):
        self.model = None

    def generate_spawns(self, time: float):
        raise NotImplementedError("YoloTrafficSource is not implemented yet.")

    def reset(self):
        pass
