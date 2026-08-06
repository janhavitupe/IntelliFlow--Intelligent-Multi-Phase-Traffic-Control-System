"""
scheduler package

Contains the TrafficScheduler, which is responsible for scheduling
traffic phases (not directly controlling signal hardware). It delegates
phase-selection decisions to a pluggable BaseStrategy.
"""
