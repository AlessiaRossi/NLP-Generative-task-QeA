import matplotlib.pyplot as plt
import os
import json
import numpy as np

def plot_training_history(trainer, output_dir):
    """
    Create and save plots of training and validation loss to monitor overfitting
    and model learning trends.
    
    Args:
        trainer: HuggingFace Trainer instance
        output_dir: Directory to save the plots
    
    Returns:
        Plot file paths
    """
    try:
        # Create visualization directory if it doesn't exist
        viz_dir = os.path.join(output_dir, "training_viz")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get training history from log history
        history = trainer.state.log_history
        
        # Extract training loss values
        train_loss = []
        train_epochs = []
        for entry in history:
            if "loss" in entry and "epoch" in entry:
                train_loss.append(entry["loss"])
                train_epochs.append(entry["epoch"])
        
        # Extract evaluation loss values
        eval_loss = []
        eval_epochs = []
        for entry in history:
            if "eval_loss" in entry and "epoch" in entry:
                eval_loss.append(entry["eval_loss"]) 
                eval_epochs.append(entry["epoch"])
        
        # Plot training and evaluation loss
        plt.figure(figsize=(10, 5))
        if train_loss:
            plt.plot(train_epochs, train_loss, 'b-', label='Training Loss')
        if eval_loss:
            plt.plot(eval_epochs, eval_loss, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        loss_plot_path = os.path.join(viz_dir, 'loss_history.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Save raw data for later analysis
        history_path = os.path.join(viz_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_loss': list(zip(train_epochs, train_loss)) if train_loss else [],
                'eval_loss': list(zip(eval_epochs, eval_loss)) if eval_loss else []
            }, f, indent=2)
        
        return loss_plot_path
    except Exception as e:
        print(f"Error plotting training history: {e}")
        return None

def analyze_learning_curve(trainer, output_dir):
    """
    Analyze the learning curve to determine if training was stopped too early
    or if the model is overfitting/underfitting.
    
    Args:
        trainer: HuggingFace Trainer instance
        output_dir: Directory to save the analysis results
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Create analysis directory if it doesn't exist
        analysis_dir = os.path.join(output_dir, "training_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get training history from log history
        history = trainer.state.log_history
        
        # Extract training and evaluation loss values
        eval_losses = []
        for entry in history:
            if "eval_loss" in entry:
                eval_losses.append(entry["eval_loss"])
        
        if not eval_losses:
            return {"error": "No evaluation losses found"}
        
        # Calculate analysis metrics
        min_loss = min(eval_losses)
        min_loss_epoch = eval_losses.index(min_loss) + 1
        last_epoch = len(eval_losses)
        
        # Check if the minimum loss is at the last epoch
        stopped_too_early = min_loss_epoch == last_epoch
        
        # Calculate trend in the latter half of training
        if len(eval_losses) > 3:
            latter_half = eval_losses[len(eval_losses)//2:]
            trend = np.polyfit(range(len(latter_half)), latter_half, 1)[0]
        else:
            trend = 0
        
        # Analyze results
        analysis = {
            "min_loss": min_loss,
            "min_loss_epoch": min_loss_epoch,
            "total_epochs": last_epoch,
            "stopped_too_early": stopped_too_early,
            "latter_half_trend": trend,
            "recommendation": ""
        }
        
        # Generate recommendations
        if stopped_too_early:
            analysis["recommendation"] = "Training may have been stopped too early. Consider increasing the number of epochs."
        elif trend < -0.01:  # Still decreasing
            analysis["recommendation"] = "Loss was still decreasing. Model might benefit from training longer."
        elif trend > 0.01:  # Increasing
            analysis["recommendation"] = "Loss was increasing in latter epochs. Possible overfitting."
        else:
            analysis["recommendation"] = "Training appears to have converged appropriately."
        
        # Save analysis results
        analysis_path = os.path.join(analysis_dir, 'learning_curve_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    except Exception as e:
        print(f"Error analyzing learning curve: {e}")
        return {"error": str(e)}
