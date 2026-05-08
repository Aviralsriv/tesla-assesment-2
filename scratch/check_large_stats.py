import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_stats_from_file(file_path):
    print(f"Reading {file_path}...")
    ea = EventAccumulator(file_path)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            last_event = events[-1]
            print(f"{tag}: {last_event.value:.4f} (step {last_event.step})")

if __name__ == "__main__":
    # Path to the large event file found earlier
    file_path = r'c:\Users\faiza\OneDrive\Desktop\Tesla Model\runs\ssl\events.out.tfevents.1778180428.SFAR.36440.0'
    get_stats_from_file(file_path)
