#!/usr/bin/env python3
"""
Example script showing how to use the separated scenario generation and simulation components.
"""

import os
import sys
from manager.world_manager import WorldManager
from base.rt_communicator import RTCommunicator

def main():
    # Example 1: Generate scenarios first
    print("=== Step 1: Generate Scenarios ===")
    os.system("python sample_scenario.py --num-tasks 2 --num-pedestrians 3 --output data/example_scenarios.json")
    
    # Example 2: Run simulation with generated scenario
    print("\n=== Step 2: Run Simulation with Generated Scenario ===")
    
    # Initialize communicator (you may need to adjust this based on your setup)
    communicator = RTCommunicator()  # Adjust initialization as needed
    
    # Create world manager with scenario data
    # Use control_mode='human' for human control, 'llm' for LLM control
    world_manager = WorldManager(
        communicator=communicator,
        max_steps=50,
        agent_path="data/config.json",
        scenario_path="data/example_scenarios.json",
        control_mode='llm'  # Change to 'human' for human control mode
    )
    
    # Spawn agent and pedestrians
    world_manager.spawn_agent()
    
    # Run simulation
    world_manager.run()
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()
