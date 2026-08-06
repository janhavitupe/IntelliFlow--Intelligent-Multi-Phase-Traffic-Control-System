"""
main.py

Entry point for the Smart Traffic Management System simulator.

Kept intentionally minimal - it creates a Simulation and runs it.
All simulation logic lives in the Simulation class (simulation.py).
"""
from simulation import Simulation


def main():
    simulation = Simulation(
        yellow_duration=2.0,
        green_duration=12.0,
        tick_interval=0.5,
        max_ticks=100,   # Set to None for an indefinite run (Ctrl+C to stop)
        live=True,
    )
    simulation.run()


if __name__ == "__main__":
    main()

