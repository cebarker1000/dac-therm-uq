#!/usr/bin/env python3
"""
Comprehensive evaluation of the full surrogate model.
Tests the surrogate on all training points and computes detailed statistics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from train_surrogate_models import FullSurrogateModel
from analysis.uq_wrapper import load_recast_training_data, reconstruct_curve_from_fpca
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SurrogateEvaluator:
    """
    Comprehensive evaluator for the full surrogate model.
    """
    
    def __init__(self, surrogate_model_path="outputs/full_surrogate_model.pkl",
                 training_data_path="outputs/training_data_fpca.npz",
                 original_data_path="outputs/uq_batch_results.npz"):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        surrogate_model_path : str
            Path to the saved surrogate model
        training_data_path : str
            Path to the FPCA training data
        original_data_path : str
            Path to the original simulation results
        """
        print("Loading surrogate model...")
        self.surrogate = FullSurrogateModel.load_model(surrogate_model_path)
        
        print("Loading training data...")
        self.training_data = load_recast_training_data(training_data_path)
        
        print("Loading original simulation data...")
        from analysis.uq_wrapper import load_batch_results
        self.original_data = load_batch_results(original_data_path)
        
        # Extract valid data
        self.valid_mask = ~np.isnan(self.original_data['oside_curves']).any(axis=1)
        self.original_curves = self.original_data['oside_curves'][self.valid_mask]
        self.training_parameters = self.training_data['parameters']
        self.training_fpca_scores = self.training_data['fpca_scores']
        
        print(f"Loaded {len(self.training_parameters)} training samples")
        print(f"Number of FPCA components: {self.surrogate.n_components}")
        print(f"Number of parameters: {self.surrogate.n_parameters}")
    
    def evaluate_on_training_data(self):
        """
        Evaluate the surrogate model on all training points.
        
        Returns:
        --------
        dict
            Dictionary containing all evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATING SURROGATE ON ALL TRAINING POINTS")
        print("="*60)
        
        # Predict using surrogate
        print("Making surrogate predictions...")
        surrogate_curves, surrogate_fpca_scores, surrogate_uncertainties, _ = \
            self.surrogate.predict_temperature_curves(self.training_parameters)
        
        # Compute statistics for FPCA coefficients
        fpca_stats = self._compute_fpca_statistics(
            self.training_fpca_scores, surrogate_fpca_scores
        )
        
        # Compute statistics for reconstructed curves
        curve_stats = self._compute_curve_statistics(
            self.original_curves, surrogate_curves
        )
        
        # Compute uncertainty analysis
        uncertainty_stats = self._compute_uncertainty_statistics(surrogate_uncertainties)
        
        # Compute parameter sensitivity
        sensitivity_stats = self._compute_parameter_sensitivity()
        
        results = {
            'fpca_statistics': fpca_stats,
            'curve_statistics': curve_stats,
            'uncertainty_statistics': uncertainty_stats,
            'sensitivity_statistics': sensitivity_stats,
            'surrogate_curves': surrogate_curves,
            'surrogate_fpca_scores': surrogate_fpca_scores,
            'surrogate_uncertainties': surrogate_uncertainties,
            'original_curves': self.original_curves,
            'training_fpca_scores': self.training_fpca_scores
        }
        
        return results
    
    def _compute_fpca_statistics(self, true_scores, predicted_scores):
        """
        Compute statistics for FPCA coefficient predictions.
        """
        print("Computing FPCA coefficient statistics...")
        
        stats = {}
        for i in range(self.surrogate.n_components):
            true_comp = true_scores[:, i]
            pred_comp = predicted_scores[:, i]
            
            # Basic metrics
            mse = mean_squared_error(true_comp, pred_comp)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_comp, pred_comp)
            r2 = r2_score(true_comp, pred_comp)
            
            # Correlation coefficients
            pearson_corr, pearson_p = pearsonr(true_comp, pred_comp)
            spearman_corr, spearman_p = spearmanr(true_comp, pred_comp)
            
            # Relative errors
            rel_error = np.abs(true_comp - pred_comp) / (np.abs(true_comp) + 1e-10)
            mean_rel_error = np.mean(rel_error)
            max_rel_error = np.max(rel_error)
            
            stats[f'component_{i+1}'] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'mean_relative_error': mean_rel_error,
                'max_relative_error': max_rel_error,
                'true_range': (true_comp.min(), true_comp.max()),
                'predicted_range': (pred_comp.min(), pred_comp.max())
            }
        
        # Overall statistics
        overall_mse = np.mean([stats[f'component_{i+1}']['mse'] for i in range(self.surrogate.n_components)])
        overall_r2 = np.mean([stats[f'component_{i+1}']['r2'] for i in range(self.surrogate.n_components)])
        
        stats['overall'] = {
            'mean_mse': overall_mse,
            'mean_r2': overall_r2,
            'total_rmse': np.sqrt(overall_mse)
        }
        
        return stats
    
    def _compute_curve_statistics(self, true_curves, predicted_curves):
        """
        Compute statistics for reconstructed temperature curves.
        """
        print("Computing curve reconstruction statistics...")
        
        # Per-curve statistics
        curve_errors = []
        curve_r2_scores = []
        curve_pearson_corrs = []
        
        for i in range(len(true_curves)):
            true_curve = true_curves[i]
            pred_curve = predicted_curves[i]
            
            # RMSE for this curve
            rmse = np.sqrt(mean_squared_error(true_curve, pred_curve))
            curve_errors.append(rmse)
            
            # R² for this curve
            r2 = r2_score(true_curve, pred_curve)
            curve_r2_scores.append(r2)
            
            # Pearson correlation for this curve
            corr, _ = pearsonr(true_curve, pred_curve)
            curve_pearson_corrs.append(corr)
        
        # Overall statistics
        overall_rmse = np.mean(curve_errors)
        overall_r2 = np.mean(curve_r2_scores)
        overall_corr = np.mean(curve_pearson_corrs)
        
        # Time-point statistics
        time_point_errors = np.sqrt(np.mean((true_curves - predicted_curves)**2, axis=0))
        
        stats = {
            'per_curve': {
                'rmse_mean': overall_rmse,
                'rmse_std': np.std(curve_errors),
                'rmse_min': np.min(curve_errors),
                'rmse_max': np.max(curve_errors),
                'r2_mean': overall_r2,
                'r2_std': np.std(curve_r2_scores),
                'r2_min': np.min(curve_r2_scores),
                'r2_max': np.max(curve_r2_scores),
                'correlation_mean': overall_corr,
                'correlation_std': np.std(curve_pearson_corrs),
                'correlation_min': np.min(curve_pearson_corrs),
                'correlation_max': np.max(curve_pearson_corrs)
            },
            'time_point_errors': time_point_errors,
            'curve_errors': curve_errors,
            'curve_r2_scores': curve_r2_scores,
            'curve_correlations': curve_pearson_corrs
        }
        
        return stats
    
    def _compute_uncertainty_statistics(self, uncertainties):
        """
        Analyze uncertainty estimates from the surrogate model.
        """
        print("Computing uncertainty statistics...")
        
        # Statistics for each component
        component_stats = {}
        for i in range(self.surrogate.n_components):
            comp_uncertainty = uncertainties[:, i]
            
            component_stats[f'component_{i+1}'] = {
                'mean_uncertainty': np.mean(comp_uncertainty),
                'std_uncertainty': np.std(comp_uncertainty),
                'min_uncertainty': np.min(comp_uncertainty),
                'max_uncertainty': np.max(comp_uncertainty),
                'median_uncertainty': np.median(comp_uncertainty)
            }
        
        # Overall uncertainty statistics
        overall_stats = {
            'mean_total_uncertainty': np.mean(np.sum(uncertainties, axis=1)),
            'std_total_uncertainty': np.std(np.sum(uncertainties, axis=1)),
            'component_correlations': np.corrcoef(uncertainties.T)
        }
        
        return {
            'component_statistics': component_stats,
            'overall_statistics': overall_stats
        }
    
    def _compute_parameter_sensitivity(self):
        """
        Compute parameter sensitivity using surrogate predictions.
        """
        print("Computing parameter sensitivity...")
        
        # Use a subset of parameters for sensitivity analysis
        n_samples = min(100, len(self.training_parameters))
        sample_indices = np.random.choice(len(self.training_parameters), n_samples, replace=False)
        
        base_parameters = self.training_parameters[sample_indices]
        base_predictions, _, _, _ = self.surrogate.predict_temperature_curves(base_parameters)
        
        sensitivity_scores = {}
        
        for i, param_name in enumerate(self.surrogate.parameter_names):
            # Perturb each parameter by 1%
            perturbed_parameters = base_parameters.copy()
            perturbation = base_parameters[:, i] * 0.01
            perturbed_parameters[:, i] += perturbation
            
            perturbed_predictions, _, _, _ = self.surrogate.predict_temperature_curves(perturbed_parameters)
            
            # Compute sensitivity as normalized change in output
            output_change = np.abs(perturbed_predictions - base_predictions)
            normalized_sensitivity = np.mean(output_change, axis=1) / (np.abs(perturbation) + 1e-10)
            
            sensitivity_scores[param_name] = {
                'mean_sensitivity': np.mean(normalized_sensitivity),
                'std_sensitivity': np.std(normalized_sensitivity),
                'max_sensitivity': np.max(normalized_sensitivity)
            }
        
        return sensitivity_scores
    
    def print_summary_statistics(self, results):
        """
        Print a comprehensive summary of evaluation results.
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*60)
        
        # FPCA statistics
        fpca_stats = results['fpca_statistics']
        print("\nFPCA Coefficient Statistics:")
        print("-" * 40)
        for i in range(self.surrogate.n_components):
            comp_stats = fpca_stats[f'component_{i+1}']
            print(f"Component {i+1}:")
            print(f"  R² = {comp_stats['r2']:.4f}")
            print(f"  RMSE = {comp_stats['rmse']:.6f}")
            print(f"  Pearson Corr = {comp_stats['pearson_correlation']:.4f}")
            print(f"  Mean Rel Error = {comp_stats['mean_relative_error']:.4f}")
        
        print(f"\nOverall FPCA: R² = {fpca_stats['overall']['mean_r2']:.4f}, "
              f"RMSE = {fpca_stats['overall']['total_rmse']:.6f}")
        
        # Curve statistics
        curve_stats = results['curve_statistics']
        print("\nCurve Reconstruction Statistics:")
        print("-" * 40)
        print(f"Mean RMSE: {curve_stats['per_curve']['rmse_mean']:.6f} ± {curve_stats['per_curve']['rmse_std']:.6f}")
        print(f"Mean R²: {curve_stats['per_curve']['r2_mean']:.4f} ± {curve_stats['per_curve']['r2_std']:.4f}")
        print(f"Mean Correlation: {curve_stats['per_curve']['correlation_mean']:.4f} ± {curve_stats['per_curve']['correlation_std']:.4f}")
        print(f"RMSE Range: [{curve_stats['per_curve']['rmse_min']:.6f}, {curve_stats['per_curve']['rmse_max']:.6f}]")
        print(f"R² Range: [{curve_stats['per_curve']['r2_min']:.4f}, {curve_stats['per_curve']['r2_max']:.4f}]")
        
        # Uncertainty statistics
        uncertainty_stats = results['uncertainty_statistics']
        print("\nUncertainty Statistics:")
        print("-" * 40)
        for i in range(self.surrogate.n_components):
            comp_unc = uncertainty_stats['component_statistics'][f'component_{i+1}']
            print(f"Component {i+1}: Mean Uncertainty = {comp_unc['mean_uncertainty']:.6f}")
        
        # Parameter sensitivity
        sensitivity_stats = results['sensitivity_statistics']
        print("\nParameter Sensitivity (Top 5):")
        print("-" * 40)
        sensitivities = [(name, stats['mean_sensitivity']) for name, stats in sensitivity_stats.items()]
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        for i, (param_name, sensitivity) in enumerate(sensitivities[:5]):
            print(f"{i+1}. {param_name}: {sensitivity:.6f}")
    
    def create_visualization_plots(self, results, output_dir="outputs/surrogate_evaluation"):
        """
        Create comprehensive visualization plots.
        """
        print(f"\nCreating visualization plots in {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. FPCA coefficient scatter plots
        self._plot_fpca_scatter_plots(results, output_dir)
        
        # 2. Curve reconstruction plots
        self._plot_curve_reconstructions(results, output_dir)
        
        # 3. Error distribution plots
        self._plot_error_distributions(results, output_dir)
        
        # 4. Parameter sensitivity plot
        self._plot_parameter_sensitivity(results, output_dir)
        
        # 5. Uncertainty analysis plots
        self._plot_uncertainty_analysis(results, output_dir)
        
        print("All plots saved successfully!")
    
    def _plot_fpca_scatter_plots(self, results, output_dir):
        """Plot FPCA coefficient predictions vs true values."""
        fpca_stats = results['fpca_statistics']
        true_scores = results['training_fpca_scores']
        pred_scores = results['surrogate_fpca_scores']
        
        fig, axes = plt.subplots(1, self.surrogate.n_components, figsize=(5*self.surrogate.n_components, 5))
        if self.surrogate.n_components == 1:
            axes = [axes]
        
        for i in range(self.surrogate.n_components):
            true_comp = true_scores[:, i]
            pred_comp = pred_scores[:, i]
            r2 = fpca_stats[f'component_{i+1}']['r2']
            
            axes[i].scatter(true_comp, pred_comp, alpha=0.6)
            axes[i].plot([true_comp.min(), true_comp.max()], 
                        [true_comp.min(), true_comp.max()], 'r--', label='Perfect')
            axes[i].set_xlabel('True FPCA Coefficient')
            axes[i].set_ylabel('Predicted FPCA Coefficient')
            axes[i].set_title(f'Component {i+1} (R² = {r2:.4f})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fpca_scatter_plots.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_curve_reconstructions(self, results, output_dir):
        """Plot example curve reconstructions."""
        original_curves = results['original_curves']
        surrogate_curves = results['surrogate_curves']
        
        # Plot first 6 curves
        n_plots = min(6, len(original_curves))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_plots):
            axes[i].plot(original_curves[i], label='Original', linewidth=2)
            axes[i].plot(surrogate_curves[i], '--', label='Surrogate', linewidth=2)
            axes[i].set_title(f'Curve {i+1}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Normalized Temperature')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/curve_reconstructions.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distributions(self, results, output_dir):
        """Plot error distribution histograms."""
        curve_stats = results['curve_statistics']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE distribution
        axes[0, 0].hist(curve_stats['curve_errors'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('RMSE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RMSE Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² distribution
        axes[0, 1].hist(curve_stats['curve_r2_scores'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('R² Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation distribution
        axes[1, 0].hist(curve_stats['curve_correlations'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Pearson Correlation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Correlation Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time-point errors
        axes[1, 1].plot(curve_stats['time_point_errors'])
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Error by Time Point')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_distributions.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, results, output_dir):
        """Plot parameter sensitivity rankings."""
        sensitivity_stats = results['sensitivity_statistics']
        
        param_names = list(sensitivity_stats.keys())
        sensitivities = [sensitivity_stats[name]['mean_sensitivity'] for name in param_names]
        
        # Sort by sensitivity
        sorted_indices = np.argsort(sensitivities)[::-1]
        sorted_names = [param_names[i] for i in sorted_indices]
        sorted_sensitivities = [sensitivities[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(sorted_names)), sorted_sensitivities)
        plt.xlabel('Parameters')
        plt.ylabel('Mean Sensitivity')
        plt.title('Parameter Sensitivity Ranking')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_sensitivities)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sorted_sensitivities)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/parameter_sensitivity.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_analysis(self, results, output_dir):
        """Plot uncertainty analysis."""
        uncertainty_stats = results['uncertainty_statistics']
        surrogate_uncertainties = results['surrogate_uncertainties']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Component uncertainties
        component_means = [uncertainty_stats['component_statistics'][f'component_{i+1}']['mean_uncertainty'] 
                          for i in range(self.surrogate.n_components)]
        component_names = [f'PC{i+1}' for i in range(self.surrogate.n_components)]
        
        axes[0, 0].bar(component_names, component_means)
        axes[0, 0].set_ylabel('Mean Uncertainty')
        axes[0, 0].set_title('Mean Uncertainty by Component')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Uncertainty correlation matrix
        corr_matrix = uncertainty_stats['overall_statistics']['component_correlations']
        im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_xticks(range(self.surrogate.n_components))
        axes[0, 1].set_yticks(range(self.surrogate.n_components))
        axes[0, 1].set_xticklabels(component_names)
        axes[0, 1].set_yticklabels(component_names)
        axes[0, 1].set_title('Uncertainty Correlation Matrix')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Total uncertainty distribution
        total_uncertainty = np.sum(surrogate_uncertainties, axis=1)
        axes[1, 0].hist(total_uncertainty, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Total Uncertainty')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Total Uncertainty Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty vs prediction error
        curve_errors = results['curve_statistics']['curve_errors']
        axes[1, 1].scatter(total_uncertainty, curve_errors, alpha=0.6)
        axes[1, 1].set_xlabel('Total Uncertainty')
        axes[1, 1].set_ylabel('Curve RMSE')
        axes[1, 1].set_title('Uncertainty vs Prediction Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/uncertainty_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, results, output_file="outputs/comprehensive_evaluation_results.npz"):
        """
        Save detailed evaluation results to file.
        """
        print(f"\nSaving detailed results to {output_file}...")
        
        np.savez_compressed(
            output_file,
            # FPCA statistics
            fpca_r2_scores=np.array([results['fpca_statistics'][f'component_{i+1}']['r2'] 
                                    for i in range(self.surrogate.n_components)]),
            fpca_rmse_scores=np.array([results['fpca_statistics'][f'component_{i+1}']['rmse'] 
                                      for i in range(self.surrogate.n_components)]),
            fpca_correlations=np.array([results['fpca_statistics'][f'component_{i+1}']['pearson_correlation'] 
                                       for i in range(self.surrogate.n_components)]),
            
            # Curve statistics
            curve_errors=results['curve_statistics']['curve_errors'],
            curve_r2_scores=results['curve_statistics']['curve_r2_scores'],
            curve_correlations=results['curve_statistics']['curve_correlations'],
            time_point_errors=results['curve_statistics']['time_point_errors'],
            
            # Uncertainty statistics
            surrogate_uncertainties=results['surrogate_uncertainties'],
            
            # Parameter sensitivity
            parameter_names=self.surrogate.parameter_names,
            sensitivity_scores=np.array([results['sensitivity_statistics'][name]['mean_sensitivity'] 
                                        for name in self.surrogate.parameter_names]),
            
            # Raw data for further analysis
            original_curves=results['original_curves'],
            surrogate_curves=results['surrogate_curves'],
            training_parameters=self.training_parameters,
            training_fpca_scores=results['training_fpca_scores'],
            surrogate_fpca_scores=results['surrogate_fpca_scores']
        )
        
        print("Detailed results saved successfully!")

def main():
    """
    Run the comprehensive evaluation.
    """
    print("V2-HeatFlow Comprehensive Surrogate Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = SurrogateEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_on_training_data()
    
    # Print summary
    evaluator.print_summary_statistics(results)
    
    # Create visualizations
    evaluator.create_visualization_plots(results)
    
    # Save detailed results
    evaluator.save_detailed_results(results)
    
    print("\n🎉 Comprehensive evaluation completed!")
    print("Check the outputs/ directory for results and plots.")

if __name__ == "__main__":
    main() 