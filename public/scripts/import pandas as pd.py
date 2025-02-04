import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy import stats

# Define the output directory
output_dir = r"C:\Users\c23017555\OneDrive - Cardiff University\Documents\PhD\EXperiment 1\analysis\Fin_analysis\gaze\output"
os.makedirs(output_dir, exist_ok=True)

# Define paths
ai_advisor_path = r"C:\Users\c23017555\OneDrive - Cardiff University\Documents\PhD\EXperiment 1\analysis\Fin_analysis\gaze\condition_aiadvisor_data_group_a_rows.csv"
no_advisor_path = r"C:\Users\c23017555\OneDrive - Cardiff University\Documents\PhD\EXperiment 1\analysis\Fin_analysis\gaze\condition_noadvisor_data_group_a_rows.csv"
backdrop_path = r"C:\Users\c23017555\OneDrive - Cardiff University\Documents\PhD\EXperiment 1\analysis\Fin_analysis\gaze\backdrop.png"

try:
    # Check if files exist
    if not os.path.exists(ai_advisor_path):
        print(f"Error: AI Advisor file not found at path: {ai_advisor_path}")
    elif not os.path.exists(no_advisor_path):
        print(f"Error: No Advisor file not found at path: {no_advisor_path}")
    elif not os.path.exists(backdrop_path):
        print(f"Error: Backdrop image not found at path: {backdrop_path}")
    else:
        # Read data
        ai_data = pd.read_csv(ai_advisor_path)
        no_advisor_data = pd.read_csv(no_advisor_path)
        backdrop = imread(backdrop_path)
        
        # Add condition labels
        ai_data['source'] = 'AI Advisor'
        no_advisor_data['source'] = 'No Advisor'
        
        # Combine datasets
        eye_data = pd.concat([ai_data, no_advisor_data], ignore_index=True)
        
        # Filter for trial numbers between 0 and 4
        eye_data = eye_data[eye_data['trial_number'].isin(range(0, 5))]
        
        # Create heatmaps for each participant-trial combination
        for source in ['AI Advisor', 'No Advisor']:
            source_data = eye_data[eye_data['source'] == source]
            unique_combinations = source_data.groupby(['participant_id', 'trial_number'])
            
            for (participant_id, trial_number), group_data in unique_combinations:
                # Create figure with 16:9 aspect ratio and black background
                plt.figure(figsize=(16, 9), facecolor='black')
                
                # Define screen dimensions
                screen_width = 1920
                screen_height = 1080
                
                # Create black background
                plt.gca().set_facecolor('black')
                
                # Create copies of the data
                x_normalized = group_data['x'].copy()
                y_normalized = group_data['y'].copy()
                
                # Print original ranges for debugging
                print(f"\nOriginal ranges - {source}, Participant {participant_id}, Trial {trial_number}:")
                print(f"X range: {x_normalized.min():.2f} to {x_normalized.max():.2f}")
                print(f"Y range: {y_normalized.min():.2f} to {y_normalized.max():.2f}")
                
                # Simple min-max normalization to screen coordinates
                x_normalized = (x_normalized - x_normalized.min()) / (x_normalized.max() - x_normalized.min()) * screen_width
                y_normalized = (y_normalized - y_normalized.min()) / (y_normalized.max() - y_normalized.min()) * screen_height
                
                # Print normalized ranges
                print(f"Normalized ranges:")
                print(f"X range: {x_normalized.min():.2f} to {x_normalized.max():.2f}")
                print(f"Y range: {y_normalized.min():.2f} to {y_normalized.max():.2f}")
                
                # Create and plot heatmap with adjusted bins
                heatmap, xedges, yedges = np.histogram2d(
                    x_normalized,
                    y_normalized,
                    bins=[60, 34],  # Adjusted to maintain 16:9 ratio
                    range=[[0, screen_width], [0, screen_height]]
                )
                
                # Plot heatmap
                ax = sns.heatmap(
                    heatmap.T,
                    cmap='hot',
                    cbar_kws={'label': 'Gaze Frequency'},
                    alpha=0.7,
                    zorder=2,
                    robust=True,
                    vmin=0,
                    vmax=np.percentile(heatmap, 95),
                    mask=(heatmap.T == 0)
                )
                
                # Force the plot to match screen dimensions
                ax.set_aspect('equal')
                plt.xlim(0, len(heatmap))
                plt.ylim(len(heatmap[0]), 0)
                
                # Style the plot for dark background
                plt.title(f'Eye Tracking Heatmap - {source}\nParticipant {participant_id}, Trial {trial_number}', 
                         color='white', pad=20)
                plt.xlabel('X Coordinate', color='white')
                plt.ylabel('Y Coordinate', color='white')
                
                # Make tick labels white
                ax.tick_params(colors='white')
                
                # Save heatmap with black background
                heatmap_file = os.path.join(output_dir, f'heatmap_{source.replace(" ", "_")}_participant_{participant_id}_trial_{trial_number}.png')
                plt.savefig(heatmap_file, 
                          dpi=300, 
                          bbox_inches='tight', 
                          facecolor='black',
                          edgecolor='none')
                plt.close()
        
        # Statistical Analysis
        with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write("Eye Tracking Analysis Summary\n")
            f.write("============================\n\n")
            
            # Separate analysis for each condition
            for source in ['AI Advisor', 'No Advisor']:
                source_data = eye_data[eye_data['source'] == source]
                f.write(f"\n{source} Analysis\n")
                f.write("="* (len(source) + 9) + "\n")
                
                # Basic Summary
                f.write("\n1. Basic Summary\n")
                f.write("--------------\n")
                f.write(f"Total number of gaze points: {len(source_data)}\n")
                f.write(f"Number of unique participants: {source_data['participant_id'].nunique()}\n")
                f.write(f"Number of unique trials: {source_data['trial_number'].nunique()}\n")
                
                # Coordinate Analysis
                f.write("\n2. Coordinate Analysis\n")
                f.write("--------------------\n")
                for coord in ['x', 'y']:
                    f.write(f"\n{coord.upper()}-coordinate:\n")
                    f.write(f"  Mean: {source_data[coord].mean():.2f}\n")
                    f.write(f"  Median: {source_data[coord].median():.2f}\n")
                    f.write(f"  Std Dev: {source_data[coord].std():.2f}\n")
                    f.write(f"  Range: {source_data[coord].min():.2f} to {source_data[coord].max():.2f}\n")
            
            # Comparative Analysis
            f.write("\n\nComparative Analysis between AI Advisor and No Advisor\n")
            f.write("================================================\n\n")
            
            # T-tests for X and Y coordinates
            for coord in ['x', 'y']:
                ai_coord = eye_data[eye_data['source'] == 'AI Advisor'][coord]
                no_advisor_coord = eye_data[eye_data['source'] == 'No Advisor'][coord]
                t_stat, p_val = stats.ttest_ind(ai_coord, no_advisor_coord)
                
                f.write(f"{coord.upper()}-coordinate comparison:\n")
                f.write(f"  T-statistic: {t_stat:.4f}\n")
                f.write(f"  P-value: {p_val:.4f}\n")
                f.write(f"  Significant difference: {'Yes' if p_val < 0.05 else 'No'}\n\n")
            
            # Gaze point count comparison
            ai_points = len(eye_data[eye_data['source'] == 'AI Advisor'])
            no_advisor_points = len(eye_data[eye_data['source'] == 'No Advisor'])
            total_points = len(eye_data)
            
            f.write("Gaze Point Distribution:\n")
            f.write(f"  AI Advisor: {ai_points} points ({(ai_points/total_points)*100:.1f}%)\n")
            f.write(f"  No Advisor: {no_advisor_points} points ({(no_advisor_points/total_points)*100:.1f}%)\n")
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        print(f"Detailed statistical analysis saved to: {os.path.join(output_dir, 'analysis_summary.txt')}")

except Exception as e:
    print(f"Error processing data: {str(e)}")
