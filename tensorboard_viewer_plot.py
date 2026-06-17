#!/usr/bin/env python3
"""
TensorBoard Log Viewer
Reads TensorBoard logs, lets you select a run, and plots graphs.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_logs(log_dir):
    """
    Load all TensorBoard logs from a directory.
    
    Args:
        log_dir: Path to folder containing TensorBoard event files
        
    Returns:
        Dictionary: {run_name: DataFrame}
    """
    runs = {}
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return runs
    
    print(f"Found {len(event_files)} event files")
    
    for event_file in event_files:
        # Extract run name from folder structure
        rel_path = os.path.relpath(event_file, log_dir)
        run_name = os.path.dirname(rel_path)
        if not run_name:
            run_name = "default"
        
        print(f"Loading: {run_name}")
        
        try:
            # Load the event file
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Get scalar tags (loss, accuracy, etc.)
            tags = ea.Tags()['scalars']
            
            if not tags:
                print(f"  No scalar data in {run_name}")
                continue
            
            # Collect data
            data = []
            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    data.append({
                        'step': event.step,
                        'tag': tag,
                        'value': event.value,
                        'wall_time': event.wall_time
                    })
            
            runs[run_name] = pd.DataFrame(data)
            print(f"  Loaded {len(data)} data points for {len(tags)} tags")
            
        except Exception as e:
            print(f"Error loading {event_file}: {e}")
    
    return runs

def select_run(runs):
    """Let user select which run to visualize."""
    print("\n" + "="*50)
    print("Available Runs:")
    print("="*50)
    
    run_names = list(runs.keys())
    for i, name in enumerate(run_names):
        count = len(runs[name])
        tags = runs[name]['tag'].nunique()
        print(f"  {i}: {name} ({count} rows, {tags} metrics)")
    
    while True:
        try:
            choice = input("\nEnter the number of the run to visualize: ")
            choice = int(choice)
            if 0 <= choice < len(run_names):
                return run_names[choice]
            else:
                print(f"Please enter a number between 0 and {len(run_names)-1}")
        except ValueError:
            print("Please enter a valid number")

def plot_run(df, run_name):
    """Plot all metrics from a selected run."""
    if df.empty:
        print("No data to plot")
        return
    
    tags = df['tag'].unique()
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 4*len(tags)))
    if len(tags) == 1:
        axes = [axes]
    
    for ax, tag in zip(axes, tags):
        tag_data = df[df['tag'] == tag]
        ax.plot(tag_data['step'], tag_data['value'])
        ax.set_xlabel('Step')
        ax.set_ylabel(tag)
        ax.set_title(f'{run_name} - {tag}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_name = run_name.replace('/', '_').replace('\\', '_')
    filename = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to: {filename}")
    
    plt.show()

def main():
    print("="*60)
    print("TENSORBOARD LOG VIEWER")
    print("="*60)
    
    log_dir = input("Enter path to TensorBoard log directory: ").strip()
    
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return
    
    print(f"Loading logs from: {log_dir}")
    
    runs = load_tensorboard_logs(log_dir)
    
    if not runs:
        print("No runs found. Check your log directory.")
        return
    
    print(f"\nLoaded {len(runs)} runs")
    
    run_name = select_run(runs)
    print(f"\nSelected: {run_name}")
    
    plot_run(runs[run_name], run_name)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
