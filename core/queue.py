"""
queue.py

A dedicated FIFO Queue for vehicles.

The Lane delegates all vehicle management to this class. The Queue
handles enqueue/dequeue operations, waiting-time bookkeeping, queue
length tracking, and will later support priority (ambulance) handling.
"""
from collections import deque

from .enums import Priority
from .vehicle import Vehicle


class Queue:
    """
    A FIFO queue of Vehicle objects stored inside a Lane.

    Keeps waiting statistics so the Analytics layer can aggregate them
    without scanning individual vehicles every time.
    """

    def __init__(self):
        # deque supports fast appends/pops from both ends.
        self._vehicles = deque()
        self._owner_lane = None            # set via bind_lane()
        self._total_waiting_time = 0.0
        self._max_queue_length = 0
        self._max_waiting_time = 0.0
        self._total_vehicles_served = 0
        self._total_enqueued = 0

    # ---------- Binding ----------

    def bind_lane(self, lane):
        """
        Store a reference to the Lane this queue belongs to.
        Used to populate Vehicle.current_lane on enqueue.
        """
        self._owner_lane = lane

    # ---------- Query ----------

    @property
    def length(self) -> int:
        return len(self._vehicles)

    @property
    def is_empty(self) -> bool:
        return len(self._vehicles) == 0

    @property
    def max_queue_length(self) -> int:
        return self._max_queue_length

    @property
    def max_waiting_time(self) -> float:
        return self._max_waiting_time

    @property
    def total_waiting_time(self) -> float:
        return self._total_waiting_time

    @property
    def total_vehicles_served(self) -> int:
        return self._total_vehicles_served

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued

    @property
    def average_waiting_time(self) -> float:
        """Average waiting time across all vehicles currently in the queue."""
        if self.length == 0:
            return 0.0
        return self._total_waiting_time / self.length

    # ---------- Mutation ----------

    def enqueue(self, vehicle: Vehicle):
        """Append a vehicle to the tail of the queue."""
        vehicle.current_lane = self._owner_lane
        self._vehicles.append(vehicle)
        self._total_enqueued += 1
        self._max_queue_length = max(self._max_queue_length, self.length)

    def dequeue(self) -> Vehicle:
        """Remove and return the vehicle at the head of the queue."""
        if self.is_empty:
            raise IndexError("Cannot dequeue from an empty Queue.")
        vehicle = self._vehicles.popleft()
        self._total_vehicles_served += 1
        self._total_waiting_time = max(0.0, self._total_waiting_time - vehicle.waiting_time)
        vehicle.current_lane = None
        return vehicle

    def peek(self) -> Vehicle:
        """Return the vehicle at the head without removing it."""
        return self._vehicles[0]

    def update_waiting_time(self, delta=1.0):
        """
        Increment waiting time for every vehicle still in the queue and
        accumulate into the queue-level waiting statistic.
        """
        if self.is_empty:
            return
        wait_delta = delta * self.length
        self._total_waiting_time += wait_delta
        for vehicle in self._vehicles:
            vehicle.update_waiting_time(delta)
            self._max_waiting_time = max(self._max_waiting_time, vehicle.waiting_time)

    # ---------- Future priority support ----------

    def count_priority(self, priority: Priority) -> int:
        """Return the number of vehicles with the given priority."""
        return sum(1 for v in self._vehicles if v.priority == priority)

    def has_emergency_vehicle(self) -> bool:
        """True if at least one HIGH-priority vehicle is queued."""
        return self.count_priority(Priority.HIGH) > 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter(self._vehicles)

    def __repr__(self) -> str:
        return f"Queue(length={self.length}, served={self._total_vehicles_served})"
