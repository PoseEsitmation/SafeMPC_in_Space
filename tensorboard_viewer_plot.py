#!/usr/bin/env python3
"""
TensorBoard Log Viewer
Reads TensorBoard logs, lets you select a run, and generates:
1. Individual PNGs for key metrics
2. Combined 3-category graph (Setting time, Reward, Control effort)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_logs(log_dir):
    """Load all TensorBoard logs from a directory."""
    runs = {}
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return runs
    
    print(f"Found {len(event_files)} event files")
    
    for event_file in event_files:
        rel_path = os.path.relpath(event_file, log_dir)
        run_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) else "default"
        print(f"Loading: {run_name}")
        
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            tags = ea.Tags()['scalars']
            
            if not tags:
                print(f"  No scalar data in {run_name}")
                continue
            
            data = []
            for tag in tags:
                for event in ea.Scalars(tag):
                    data.append({'step': event.step, 'tag': tag, 'value': event.value})
            
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
            choice = int(input("\nEnter the number of the run to visualize: "))
            if 0 <= choice < len(run_names):
                return run_names[choice]
        except ValueError:
            pass
        print(f"Please enter a number between 0 and {len(run_names)-1}")

def plot_individual_metrics(df, run_name):
    """Generate individual PNGs for key metrics."""
    
    key_metrics = [
        'train/loss',
        'train/learning_rate',
        'train_env/task_0/reward',
        'eval_env/task_0/reward',
        'train_env/task_0/koz_violations',
        'train_env/task_0/attitude_error_deg'
    ]
    
    all_tags = df['tag'].unique()
    tags = [tag for tag in all_tags if any(km in tag for km in key_metrics)]
    
    if not tags:
        tags = all_tags[:3]
        print(f"No key metrics found. Showing first 3: {tags}")
    else:
        print(f"Found {len(tags)} key metrics")

    output_dir = f"plots/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    for tag in tags:
        tag_data = df[df['tag'] == tag]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tag_data['step'], tag_data['value'])
        ax.set_xlabel('Step')
        ax.set_ylabel(tag)
        ax.set_title(f'{run_name} - {tag}')
        ax.grid(True, alpha=0.3)
        
        safe_name = tag.replace('/', '_').replace('\\', '_')
        filename = os.path.join(output_dir, f'{safe_name}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close(fig)

def plot_3_category_graph(df, run_name):
    """
    Plot 3 categories in 3 subplots: Setting time, Reward, Control effort.
    Matches the requested graph format exactly.
    """
    # Pivot data so tags become columns
    pivot_df = df.pivot(index='step', columns='tag', values='value').reset_index()
    
    print("\nAvailable metrics in data:")
    for col in pivot_df.columns:
        print(f"  - {col}")
    
    # Define categories
    normal = pivot_df[pivot_df['train_env/task_0/thetha_margin'] > 0] if 'train_env/task_0/thetha_margin' in pivot_df.columns else pd.DataFrame()
    violated = pivot_df[pivot_df['train_env/task_0/koz_violations'] > 0] if 'train_env/task_0/koz_violations' in pivot_df.columns else pd.DataFrame()
    non_settled = pivot_df[pivot_df['train_env/task_0/attitude_error_deg'] > 10] if 'train_env/task_0/attitude_error_deg' in pivot_df.columns else pd.DataFrame()
    
    print(f"\nCategory counts:")
    print(f"  Normal: {len(normal)} rows")
    print(f"  Violated: {len(violated)} rows")
    print(f"  Non-settled: {len(non_settled)} rows")
    
    # Calculate percentages for info text
    total = len(pivot_df)
    non_settled_pct = (len(non_settled) / total) * 100 if total > 0 else 0
    violated_pct = (len(violated) / total) * 100 if total > 0 else 0
    
    # Create 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Setting time (using thetha_margin as proxy)
    ax1 = axes[0]
    if not normal.empty:
        ax1.scatter(normal['step'], normal['train_env/task_0/thetha_margin'], 
                   color='green', label='Normal', alpha=0.6, s=8)
    if not violated.empty:
        ax1.scatter(violated['step'], violated['train_env/task_0/koz_violations'], 
                   color='red', label='Violated', alpha=0.6, s=8)
    if not non_settled.empty:
        ax1.scatter(non_settled['step'], non_settled['train_env/task_0/attitude_error_deg'], 
                   color='orange', label='Non-settled', alpha=0.6, s=8)
    ax1.set_ylabel('Setting time (sec)')
    ax1.set_title(f'Setting time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward
    ax2 = axes[1]
    if not normal.empty:
        ax2.scatter(normal['step'], normal['train_env/task_0/reward'], 
                   color='green', alpha=0.6, s=8)
    if not violated.empty:
        ax2.scatter(violated['step'], violated['train_env/task_0/reward'], 
                   color='red', alpha=0.6, s=8)
    if not non_settled.empty:
        ax2.scatter(non_settled['step'], non_settled['train_env/task_0/reward'], 
                   color='orange', alpha=0.6, s=8)
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Reward')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Control effort (using reward inverse as proxy)
    ax3 = axes[2]
    if not normal.empty:
        ax3.scatter(normal['step'], normal['train_env/task_0/reward'] * -1, 
                   color='green', alpha=0.6, s=8)
    if not violated.empty:
        ax3.scatter(violated['step'], violated['train_env/task_0/reward'] * -1, 
                   color='red', alpha=0.6, s=8)
    if not non_settled.empty:
        ax3.scatter(non_settled['step'], non_settled['train_env/task_0/reward'] * -1, 
                   color='orange', alpha=0.6, s=8)
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Control effort (W/M²s)')
    ax3.set_title(f'Control effort')
    ax3.grid(True, alpha=0.3)
    
    # Add info text at top
    fig.suptitle(f't=100 s, Bound [80,180] deg\nNon-settled: {non_settled_pct:.3f}%, Violated rate: {violated_pct:.3f}%', 
                 fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    output_dir = f"plots/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, '3_category_graph.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)

def main():
    print("="*60)
    print("TENSORBOARD LOG VIEWER")
    print("="*60)
    
    log_dir = input("Enter path to TensorBoard log directory: ").strip()
    
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return
    
    runs = load_tensorboard_logs(log_dir)
    
    if not runs:
        print("No runs found. Check your log directory.")
        return
    
    print(f"\nLoaded {len(runs)} runs")
    run_name = select_run(runs)
    print(f"\nSelected: {run_name}")
    
    df = runs[run_name]
    
    # 1. Individual PNGs for key metrics
    plot_individual_metrics(df, run_name)
    
    # 2. Combined 3-category graph
    plot_3_category_graph(df, run_name)
    
    print("\nDone! All plots saved to plots/")

if __name__ == "__main__":
    main()
