"""
services package

Contains pluggable service components that model physical vehicle
discharge behavior at the intersection. The scheduler remains unaware of
service rates; the Simulation layer uses the ServiceModel to determine
how many vehicles can leave each active movement per tick.
"""
from .service_model import ServiceModel

__all__ = ["ServiceModel"]
