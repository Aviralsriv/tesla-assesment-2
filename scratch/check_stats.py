import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_stats(log_dir):
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if 'tfevents' in f:
                event_files.append(os.path.join(root, f))
    
    if not event_files:
        print("No event files found.")
        return

    # Sort by modification time
    event_files.sort(key=os.path.getmtime, reverse=True)
    
    latest_file = event_files[0]
    print(f"Reading {latest_file}...")
    
    ea = EventAccumulator(latest_file)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            last_event = events[-1]
            print(f"{tag}: {last_event.value:.4f} (step {last_event.step})")

if __name__ == "__main__":
    get_latest_stats(r'c:\Users\faiza\OneDrive\Desktop\Tesla Model\runs\ssl')
