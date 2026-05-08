import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_stats_from_file(latest_file):
    print(f"Reading {latest_file}...")
    ea = EventAccumulator(latest_file)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    if not tags:
        print("No scalar tags found.")
        return
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            last_event = events[-1]
            print(f"{tag}: {last_event.value:.4f} (step {last_event.step})")

if __name__ == "__main__":
    # Checking the large event file
    get_stats_from_file(r'c:\Users\faiza\OneDrive\Desktop\Tesla Model\runs\ssl\events.out.tfevents.1778180428.SFAR.36440.0')
