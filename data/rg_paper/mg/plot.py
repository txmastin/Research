import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy.integrate import simpson

def calculate_metrics(errors_dict, reservoir_labels):
    """Calculate performance metrics for the critical network compared to the controls."""
    metrics = {}
    
    # Convert errors to accuracies
    accuracies = {
        res_type: 1 - np.array(errors_dict[res_type]) for res_type in errors_dict
    }
    
    # Calculate metrics for the critical network
    critical_acc = np.mean(accuracies["critical"], axis=0)  # Mean accuracy across trials
    metrics["final_accuracy_critical"] = critical_acc[-1]
    metrics["mean_accuracy_critical"] = np.mean(critical_acc)
    metrics["stability_critical"] = np.std(critical_acc)
    
    # Compare to the control networks
    for res_type in ["control1", "control2"]:
        control_acc = np.mean(accuracies[res_type], axis=0)
        
        # Final accuracy improvement
        final_acc_improvement = (
            (critical_acc[-1] - control_acc[-1]) / control_acc[-1] * 100
        )
        
        # Mean accuracy improvement
        mean_acc_improvement = (
            (np.mean(critical_acc) - np.mean(control_acc)) / np.mean(control_acc) * 100
        )
        
        # Stability comparison
        stability_diff = np.std(critical_acc) - np.std(control_acc)
        
        # Store metrics
        metrics[f"final_accuracy_improvement_{res_type}"] = final_acc_improvement
        metrics[f"mean_accuracy_improvement_{res_type}"] = mean_acc_improvement
        metrics[f"stability_difference_{res_type}"] = stability_diff
    
    return metrics

def load_data(file_pattern):
    """Load multiple files matching the pattern and return a list of numpy arrays."""
    files = sorted([f for f in os.listdir() if f.startswith(file_pattern)])
    all_data = [np.loadtxt(f) for f in files]
    return all_data

def plot_average_error(errors_dict, renorm_dict, reservoir_labels, line_styles):
    """Plot the average error over training steps for each reservoir type."""
    plt.figure(figsize=(10, 6))
    
    # Plot original reservoirs
    for res_type, label in reservoir_labels.items():
        avg_errors = np.mean(errors_dict[res_type], axis=0)
        plt.plot(avg_errors, color=line_styles[res_type]['color'], linestyle=line_styles[res_type]['style'], label=f"{label} (Original)")


    # Plot renormalized reservoirs
    for res_type, label in reservoir_labels.items():
        avg_errors_renorm = np.mean(renorm_dict[res_type], axis=0)
        plt.plot(avg_errors_renorm, color=line_styles[res_type]['color'], linestyle=line_styles[res_type]['style'], label=f"{label} (Renormalized)", alpha=0.6)
    
    plt.xlabel("Training Step")
    plt.ylabel("Average Error")
    plt.legend()
    plt.grid(False)
    plt.savefig("average_error_over_training.png")
    plt.show()

def plot_average_error_with_std(errors_dict, renorm_dict, reservoir_labels, line_styles):
    """Plot the average error over training steps with shaded standard deviation."""
    plt.figure(figsize=(10, 6))
    
    # Plot original reservoirs with standard deviation
    for res_type, label in reservoir_labels.items():
        # Calculate mean and std for the original reservoirs
        avg_errors = np.mean(errors_dict[res_type], axis=0)
        std_errors = np.std(errors_dict[res_type], axis=0)
        
        # Plot mean with shaded region for std
        plt.semilogx(avg_errors, color=line_styles[res_type]['color'], linestyle=line_styles[res_type]['style'], label=f"{label} (Original)")
        plt.fill_between(
            range(len(avg_errors)), avg_errors - std_errors, avg_errors + std_errors,
            color=line_styles[res_type]['color'], alpha=0.2
        )
    
    plt.xlabel("Training Step", fontsize=18)
    plt.ylabel("Average Error", fontsize=18)
    plt.legend()
    plt.grid(False)
 
    
    plt.figure(figsize=(10, 6))
    # Plot renormalized reservoirs with standard deviation
    for res_type, label in reservoir_labels.items():
        # Calculate mean and std for the renormalized reservoirs
        avg_errors_renorm = np.mean(renorm_dict[res_type], axis=0)
        std_errors_renorm = np.std(renorm_dict[res_type], axis=0)
        
        # Plot mean with shaded region for std
        plt.semilogx(avg_errors_renorm, color=line_styles[res_type]['color'], linestyle=line_styles[res_type]['style'], alpha=0.6, label=f"{label} (Renormalized)")
        plt.fill_between(
            range(len(avg_errors_renorm)), avg_errors_renorm - std_errors_renorm, avg_errors_renorm + std_errors_renorm,
            color=line_styles[res_type]['color'], alpha=0.1
        )
    
    # Formatting and labels
    plt.xlabel("Training Step", fontsize=18)
    plt.ylabel("Average Error", fontsize=18)
    plt.legend()
    plt.grid(False)
    plt.savefig("average_error_with_std_over_training.png")
    plt.show()



def calculate_final_accuracy(errors_dict):
    """Calculate the average accuracy (1 - error) at the final training step for each type."""
    accuracy = {}
    for res_type, data in errors_dict.items():
        final_errors = [trial[-1] for trial in data]
        accuracy[res_type] = 1 - np.mean(final_errors)
    return accuracy


def calculate_std(errors_dict):
    """Calculate the standard deviation of accuracies for each reservoir type."""
    std_devs = {}
    for res_type, data in errors_dict.items():
        final_errors = [trial[-1] for trial in data]
        std_devs[res_type] = np.std([1 - err for err in final_errors])  # Convert errors to accuracies
    return std_devs

from matplotlib.legend_handler import HandlerLine2D

def plot_accuracy_vs_size(accuracy_before, accuracy_after, reservoir_labels, line_styles, renorm_sizes, std_before, std_after):
    """Scatter plot: accuracy vs reservoir size with adjusted arrows and error bars."""
    plt.figure(figsize=(10, 6))

    # Original reservoir sizes and accuracies
    original_sizes = [1000] * len(accuracy_before)
    original_accuracies = list(accuracy_before.values())

    # Renormalized reservoir sizes and accuracies
    renormalized_sizes = [renorm_sizes[res_type] for res_type in accuracy_after.keys()]
    renormalized_accuracies = list(accuracy_after.values())

    # Plot original points with error bars
    for i, res_type in enumerate(accuracy_before.keys()):
        plt.errorbar(
            original_sizes[i], original_accuracies[i],
            yerr=std_before[res_type], fmt='^', color=line_styles[res_type]['color'],
            markersize=10, capsize=5
        )

    # Plot renormalized points with error bars
    for i, res_type in enumerate(accuracy_after.keys()):
        plt.errorbar(
            renormalized_sizes[i], renormalized_accuracies[i],
            yerr=std_after[res_type], fmt='o', color=line_styles[res_type]['color'],
            markersize=10, capsize=5
        )

        # Calculate the offset for arrow termination
        arrow_dx = renormalized_sizes[i] - original_sizes[i]
        arrow_dy = renormalized_accuracies[i] - original_accuracies[i]
        scale = 0.99  # Adjust the scale to shorten the arrow
        adjusted_end_x = original_sizes[i] + scale * arrow_dx
        adjusted_end_y = original_accuracies[i] + scale * arrow_dy

        # Add an arrow from original to adjusted endpoint
        plt.annotate(
            "",
            xy=(adjusted_end_x, adjusted_end_y),  # Shortened arrow endpoint
            xytext=(original_sizes[i], original_accuracies[i]),  # Arrow start
            arrowprops=dict(
                arrowstyle="->",
                color=line_styles[res_type]['color'],
                lw=2.5
            )
        )

    # Manually create the legend
    manual_legend = [
        plt.Line2D([0], [0], marker='^', color=line_styles["critical"]['color'], label="Critical Spiking (Before Coarse-Graining)", markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='o', color=line_styles["critical"]['color'], label="Critical Spiking (After Coarse-Graining)", markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='^', color=line_styles["control1"]['color'], label="Synchronous Spiking (Before Coarse-Graining)", markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='o', color=line_styles["control1"]['color'], label="Synchronous Spiking (After Coarse-Graining)", markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='^', color=line_styles["control2"]['color'], label="Irregular Spiking (Before Coarse-Graining)", markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='o', color=line_styles["control2"]['color'], label="Irregular Spiking (After Coarse-Graining)", markersize=10, linestyle=''),
    ]
    plt.legend(handles=manual_legend, loc="lower left", fontsize=10)

    # Plot labels and formatting
    plt.xlabel("Reservoir Size", fontsize=18)
    plt.ylabel("Accuracy After Training", fontsize=18)
    plt.savefig("accuracy_vs_size_with_manual_legend.png")
    plt.show()


def old_plot_accuracy_vs_size(accuracy_before, accuracy_after, reservoir_labels, line_styles, renorm_sizes, std_before, std_after):
    """Scatter plot: accuracy vs reservoir size with adjusted arrows and error bars."""
    plt.figure(figsize=(10, 6))

    # Original reservoir sizes and accuracies
    original_sizes = [1000] * len(accuracy_before)
    original_accuracies = list(accuracy_before.values())
    
    # Renormalized reservoir sizes and accuracies
    renormalized_sizes = [renorm_sizes[res_type] for res_type in accuracy_after.keys()]
    renormalized_accuracies = list(accuracy_after.values())
    
    # Plot original points with error bars
    for i, res_type in enumerate(accuracy_before.keys()):
        plt.errorbar(
            original_sizes[i], original_accuracies[i],
            yerr=std_before[res_type], fmt='^', color=line_styles[res_type]['color'], 
            label=f"{reservoir_labels[res_type]} (Before Coarse-Graining)", markersize=10, capsize=5
        )

    # Plot renormalized points with error bars
    for i, res_type in enumerate(accuracy_after.keys()):
        plt.errorbar(
            renormalized_sizes[i], renormalized_accuracies[i],
            yerr=std_after[res_type], fmt='o', color=line_styles[res_type]['color'], 
            label=f"{reservoir_labels[res_type]} (After Coarse-Graining)", markersize=10, capsize=5
        )

        # Calculate the offset for arrow termination
        arrow_dx = renormalized_sizes[i] - original_sizes[i]
        arrow_dy = renormalized_accuracies[i] - original_accuracies[i]
        scale = 0.99  # Adjust the scale to shorten the arrow
        adjusted_end_x = original_sizes[i] + scale * arrow_dx
        adjusted_end_y = original_accuracies[i] + scale * arrow_dy

        # Add an arrow from original to adjusted endpoint
        plt.annotate(
            "", 
            xy=(adjusted_end_x, adjusted_end_y),  # Shortened arrow endpoint
            xytext=(original_sizes[i], original_accuracies[i]),  # Arrow start
            arrowprops=dict(
                arrowstyle="->", 
                color=line_styles[res_type]['color'],
                lw=2.5
            )
        )
    
    # Plot labels and formatting
    plt.xlabel("Reservoir Size", fontsize=18)
    plt.ylabel("Accuracy After Training", fontsize=18)
    plt.legend()
    plt.savefig("accuracy_vs_size_with_arrows_and_error_bars.png")
    plt.show()



def older_plot_accuracy_vs_size(accuracy_before, accuracy_after, reservoir_labels, line_styles, renorm_sizes):
    """Scatter plot: accuracy vs reservoir size with adjusted arrows showing renormalization."""
    plt.figure(figsize=(10, 6))

    # Original reservoir sizes and accuracies
    original_sizes = [1000] * len(accuracy_before)
    original_accuracies = list(accuracy_before.values())

    # Renormalized reservoir sizes and accuracies
    renormalized_sizes = [renorm_sizes[res_type] for res_type in accuracy_after.keys()]
    renormalized_accuracies = list(accuracy_after.values())

    # Plot original points
    for i, res_type in enumerate(accuracy_before.keys()):
        plt.scatter(original_sizes[i], original_accuracies[i], color=line_styles[res_type]['color'],
                    marker='^', label=f"{reservoir_labels[res_type]} (Original)", s=100)

    # Plot renormalized points
    for i, res_type in enumerate(accuracy_after.keys()):
        plt.scatter(renormalized_sizes[i], renormalized_accuracies[i], color=line_styles[res_type]['color'],
                    marker='o', label=f"{reservoir_labels[res_type]} (Renormalized)", s=100)

        # Calculate the offset for arrow termination
        arrow_dx = renormalized_sizes[i] - original_sizes[i]
        arrow_dy = renormalized_accuracies[i] - original_accuracies[i]
        scale = 0.99  # Adjust the scale to shorten the arrow
        adjusted_end_x = original_sizes[i] + scale * arrow_dx
        adjusted_end_y = original_accuracies[i] + scale * arrow_dy

        # Add an arrow from original to adjusted endpoint
        plt.annotate(
            "",
            xy=(adjusted_end_x, adjusted_end_y),  # Shortened arrow endpoint
            xytext=(original_sizes[i], original_accuracies[i]),  # Arrow start
            arrowprops=dict(
                arrowstyle="->",
                color=line_styles[res_type]['color'],
                lw=2.5
            )
        )

    # Plot labels and formatting
    plt.title("Accuracy vs Reservoir Size")
    plt.xlabel("Reservoir Size")
    plt.ylabel("Average Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_size_with_arrows.png")
    plt.show()

if __name__ == "__main__":
    # Reservoir types and their corresponding labels
    reservoir_labels = {
        "control1": "Synchronous Spiking",
        "control2": "Irregular Spiking",
        "critical": "Critical Spiking"
    }
    
    # Line styles for plots
    line_styles = {
        "control1": {"color": "blue", "style": "dotted"},
        "control2": {"color": "black", "style": "dashed"},
        "critical": {"color": "red", "style": "solid"}
    }
    
    # Renormalized reservoir sizes
    renorm_sizes = {
        "critical": 441.6,
        "control1": 451.2,
        "control2": 458.4
    }
    
    # Dictionary to hold error data
    errors_before = {}
    errors_after = {}
    
    # Load data for each reservoir type
    for res_type in reservoir_labels.keys():
        errors_before[res_type] = load_data(f"avg_errors_{res_type}_")
        errors_after[res_type] = load_data(f"avg_errors_{res_type}_renorm_")
    
    std_before = calculate_std(errors_before)
    std_after = calculate_std(errors_after)

    # Step 1: Plot average error over training steps
    plot_average_error_with_std(errors_before, errors_after, reservoir_labels, line_styles)
    
    # Step 2: Calculate final accuracy
    accuracy_before = calculate_final_accuracy(errors_before)
    accuracy_after = calculate_final_accuracy(errors_after)
    
    # Step 3: Plot accuracy vs reservoir size
    plot_accuracy_vs_size(accuracy_before, accuracy_after, reservoir_labels, line_styles, renorm_sizes, std_before, std_after)
    
    # Print the final accuracy values
    print("Final Accuracy Before Renormalization:")
    for res_type, acc in accuracy_before.items():
        print(f"{reservoir_labels[res_type]}: {acc:.4f}")
    
    print("\nFinal Accuracy After Renormalization:")
    for res_type, acc in accuracy_after.items():
        print(f"{reservoir_labels[res_type]}: {acc:.4f}")
    
    metrics_before = calculate_metrics(errors_before, reservoir_labels)
    metrics_after = calculate_metrics(errors_after, reservoir_labels)

    # Print metrics
    print("Metrics Before Renormalization:")
    for key, value in metrics_before.items():
        print(f"{key}: {value:.2f}")

    print("\nMetrics After Renormalization:")
    for key, value in metrics_after.items():
        print(f"{key}: {value:.2f}")

