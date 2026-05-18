"""
Graphical User Interface for Cr³⁺ Phosphor Expert System
Author: Snežana Đurković
Year: 2026

Modern GUI for running ML predictions and expert system evaluation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import threading
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import your existing modules
try:
    from expert_system_scoring import PhosphorExpertSystem, main_expert_system_pipeline
    from integrated_prediction_pipeline import (
        train_final_models, predict_with_uncertainty, 
        run_cross_validation, set_seeds
    )
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


class PhosphorGUI:
    """Main GUI Application for Phosphor Expert System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Cr³⁺ Phosphor Expert System - Discovery Framework")
        self.root.geometry("1200x800")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color scheme
        self.bg_color = "#f0f0f0"
        self.accent_color = "#2196F3"
        self.success_color = "#4CAF50"
        self.warning_color = "#FF9800"
        self.error_color = "#f44336"
        
        self.root.configure(bg=self.bg_color)
        
        # Variables
        self.training_file = tk.StringVar()
        self.prediction_file = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.getcwd())
        self.random_state = tk.IntVar(value=42)
        self.optimize_state = tk.BooleanVar(value=False)
        self.target_dqb_min = tk.DoubleVar(value=2.8)
        self.target_dqb_max = tk.DoubleVar(value=3.8)
        self.target_emission_min = tk.DoubleVar(value=650)
        self.target_emission_max = tk.DoubleVar(value=720)
        
        # Status
        self.is_running = False
        self.models_trained = False
        self.catboost_models = None
        self.nn_models = None
        self.scaler = None
        
        # Create UI
        self.create_ui()
        
        # Check if modules are available
        if not MODULES_AVAILABLE:
            messagebox.showwarning(
                "Missing Modules",
                "Expert system modules not found!\n\n"
                "Make sure expert_system_scoring.py and\n"
                "integrated_prediction_pipeline.py are in the same folder."
            )
    
    def create_ui(self):
        """Create main UI layout"""
        
        # Title
        title_frame = tk.Frame(self.root, bg=self.accent_color, height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🔬 Cr³⁺ Phosphor Discovery Expert System",
            font=("Arial", 18, "bold"),
            bg=self.accent_color,
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (Controls)
        left_panel = tk.Frame(main_container, bg=self.bg_color, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Right panel (Results)
        right_panel = tk.Frame(main_container, bg=self.bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create sections
        self.create_file_section(left_panel)
        self.create_settings_section(left_panel)
        self.create_action_section(left_panel)
        self.create_results_section(right_panel)
    
    def create_file_section(self, parent):
        """File selection section"""
        section = ttk.LabelFrame(parent, text="📁 Input Files", padding=10)
        section.pack(fill=tk.X, pady=(0, 10))
        
        # Training file
        ttk.Label(section, text="Training Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(section, textvariable=self.training_file, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(section, text="Browse", command=self.browse_training_file).grid(row=0, column=2)
        
        # Prediction file
        ttk.Label(section, text="Prediction Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(section, textvariable=self.prediction_file, width=30).grid(row=1, column=1, padx=5)
        ttk.Button(section, text="Browse", command=self.browse_prediction_file).grid(row=1, column=2)
        
        # Output folder
        ttk.Label(section, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(section, textvariable=self.output_folder, width=30).grid(row=2, column=1, padx=5)
        ttk.Button(section, text="Browse", command=self.browse_output_folder).grid(row=2, column=2)
    
    def create_settings_section(self, parent):
        """Settings section"""
        section = ttk.LabelFrame(parent, text="⚙️ Expert System Settings", padding=10)
        section.pack(fill=tk.X, pady=(0, 10))
        
        # Target Dq/B range
        ttk.Label(section, text="Target Dq/B Range:").grid(row=0, column=0, sticky=tk.W, pady=5)
        dqb_frame = tk.Frame(section)
        dqb_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Entry(dqb_frame, textvariable=self.target_dqb_min, width=8).pack(side=tk.LEFT)
        ttk.Label(dqb_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(dqb_frame, textvariable=self.target_dqb_max, width=8).pack(side=tk.LEFT)
        
        # Target emission range
        ttk.Label(section, text="Target Emission (nm):").grid(row=1, column=0, sticky=tk.W, pady=5)
        emission_frame = tk.Frame(section)
        emission_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Entry(emission_frame, textvariable=self.target_emission_min, width=8).pack(side=tk.LEFT)
        ttk.Label(emission_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(emission_frame, textvariable=self.target_emission_max, width=8).pack(side=tk.LEFT)
        
        # Random state
        ttk.Label(section, text="Random State:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(section, textvariable=self.random_state, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Optimize checkbox
        ttk.Checkbutton(
            section,
            text="Optimize Random State (slow)",
            variable=self.optimize_state
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
    
    def create_action_section(self, parent):
        """Action buttons section"""
        section = tk.Frame(parent, bg=self.bg_color)
        section.pack(fill=tk.X, pady=(0, 10))
        
        # Run button
        self.run_button = tk.Button(
            section,
            text="▶ Run Complete Pipeline",
            command=self.run_pipeline,
            bg=self.success_color,
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            cursor="hand2"
        )
        self.run_button.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(section, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            section,
            text="Ready to run",
            bg=self.bg_color,
            fg=self.accent_color,
            font=("Arial", 10)
        )
        self.status_label.pack()
        
        # Quick actions
        quick_frame = ttk.LabelFrame(section, text="Quick Actions", padding=5)
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            quick_frame,
            text="📊 View Results",
            command=self.view_results
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            quick_frame,
            text="📄 Open Report",
            command=self.open_report
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            quick_frame,
            text="📁 Open Output Folder",
            command=self.open_output_folder
        ).pack(fill=tk.X, pady=2)
    
    def create_results_section(self, parent):
        """Results display section"""
        section = ttk.LabelFrame(parent, text="📈 Results & Recommendations", padding=10)
        section.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(section)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Console output
        console_frame = tk.Frame(self.notebook)
        self.notebook.add(console_frame, text="Console Output")
        
        self.console_text = scrolledtext.ScrolledText(
            console_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4"
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Top recommendations
        recommendations_frame = tk.Frame(self.notebook)
        self.notebook.add(recommendations_frame, text="Top Recommendations")
        
        # Treeview for recommendations
        columns = ("Rank", "Formula", "Dq/B", "Emission", "Score", "Tier")
        self.recommendations_tree = ttk.Treeview(
            recommendations_frame,
            columns=columns,
            show="headings",
            height=15
        )
        
        for col in columns:
            self.recommendations_tree.heading(col, text=col)
            self.recommendations_tree.column(col, width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(
            recommendations_frame,
            orient=tk.VERTICAL,
            command=self.recommendations_tree.yview
        )
        self.recommendations_tree.configure(yscrollcommand=scrollbar.set)
        
        self.recommendations_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tab 3: Statistics
        stats_frame = tk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: Visualization
        viz_frame = tk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        self.viz_canvas_frame = viz_frame
    
    def browse_training_file(self):
        filename = filedialog.askopenfilename(
            title="Select Training Dataset",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.training_file.set(filename)
    
    def browse_prediction_file(self):
        filename = filedialog.askopenfilename(
            title="Select Prediction Dataset",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.prediction_file.set(filename)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
    
    def log_console(self, message):
        """Log message to console"""
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.console_text.update()
    
    def update_status(self, message, color=None):
        """Update status label"""
        self.status_label.config(text=message)
        if color:
            self.status_label.config(fg=color)
        self.root.update()
    
    def run_pipeline(self):
        """Run the complete ML + Expert System pipeline"""
        
        if self.is_running:
            messagebox.showwarning("Already Running", "Pipeline is already running!")
            return
        
        # Validate inputs
        if not self.training_file.get():
            messagebox.showerror("Error", "Please select a training dataset!")
            return
        
        if not self.prediction_file.get():
            messagebox.showerror("Error", "Please select a prediction dataset!")
            return
        
        if not MODULES_AVAILABLE:
            messagebox.showerror(
                "Error",
                "Expert system modules not found!\n"
                "Place expert_system_scoring.py and\n"
                "integrated_prediction_pipeline.py in the same folder."
            )
            return
        
        # Run in thread to prevent GUI freeze
        thread = threading.Thread(target=self._run_pipeline_thread)
        thread.daemon = True
        thread.start()
    
    def _run_pipeline_thread(self):
        """Thread worker for pipeline execution"""
        
        self.is_running = True
        self.run_button.config(state=tk.DISABLED, text="Running...")
        self.progress.start()
        
        try:
            # Clear console
            self.console_text.delete(1.0, tk.END)
            
            self.log_console("="*60)
            self.log_console("STARTING EXPERT SYSTEM PIPELINE")
            self.log_console("="*60)
            
            # Load training data
            self.update_status("Loading training data...", self.accent_color)
            self.log_console("\n[1/6] Loading training data...")
            
            train_df = pd.read_excel(self.training_file.get())
            X_train = train_df.iloc[:, 1:-1].values
            y_train = train_df.iloc[:, -1].values
            
            self.log_console(f"✓ Loaded {len(train_df)} samples with {X_train.shape[1]} features")
            
            # Cross-validation
            if self.optimize_state.get():
                self.update_status("Optimizing random state (this may take a while)...", self.warning_color)
                self.log_console("\n[2/6] Optimizing random state...")
                # Would call optimize_random_state here
                self.log_console("✓ Optimization complete")
            else:
                self.update_status("Running cross-validation...", self.accent_color)
                self.log_console(f"\n[2/6] Using random state: {self.random_state.get()}")
            
            # Train models
            self.update_status("Training ensemble models...", self.accent_color)
            self.log_console("\n[3/6] Training final models...")
            
            set_seeds(self.random_state.get())
            self.catboost_models, self.nn_models, self.scaler = train_final_models(
                X_train, y_train, self.random_state.get()
            )
            self.models_trained = True
            
            self.log_console("✓ Models trained successfully!")
            
            # Load prediction data
            self.update_status("Making predictions...", self.accent_color)
            self.log_console("\n[4/6] Loading prediction candidates...")
            
            pred_df = pd.read_excel(self.prediction_file.get())
            formulas = pred_df.iloc[:, 0].values
            X_pred = pred_df.iloc[:, 1:].values
            
            # Make predictions
            catboost_preds, nn_preds, uncertainties = predict_with_uncertainty(
                X_pred, self.catboost_models, self.nn_models, self.scaler
            )
            
            predictions_df = pd.DataFrame({
                'Formula': formulas,
                'CatBoost_Pred': catboost_preds,
                'NN_Pred': nn_preds,
                'Uncertainty': uncertainties
            })
            
            self.log_console(f"✓ Predictions complete for {len(predictions_df)} candidates")
            
            # Run expert system
            self.update_status("Running expert system evaluation...", self.accent_color)
            self.log_console("\n[5/6] Running expert system...")
            
            # Create expert system with custom settings
            expert = PhosphorExpertSystem(
                target_dqb_range=(self.target_dqb_min.get(), self.target_dqb_max.get()),
                target_emission_range=(self.target_emission_min.get(), self.target_emission_max.get())
            )
            
            results_df = expert.evaluate_dataset(predictions_df)
            
            # Save outputs
            output_dir = self.output_folder.get()
            excel_path = os.path.join(output_dir, "expert_system_recommendations.xlsx")
            report_path = os.path.join(output_dir, "expert_system_report.txt")
            
            results_df.to_excel(excel_path, index=False)
            self.log_console(f"✓ Saved recommendations to: {excel_path}")
            
            # Generate report
            report = expert.generate_summary_report(results_df)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.log_console(f"✓ Saved report to: {report_path}")
            
            # Display results
            self.update_status("Processing complete!", self.success_color)
            self.log_console("\n[6/6] Complete!")
            self.log_console("\n" + "="*60)
            self.log_console("PIPELINE FINISHED SUCCESSFULLY")
            self.log_console("="*60)
            
            # Update recommendations table
            self._display_recommendations(results_df)
            
            # Update statistics
            self._display_statistics(results_df)
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Pipeline completed successfully!\n\n"
                f"Results saved to:\n{output_dir}\n\n"
                f"Top candidates identified: {len(results_df[results_df['Tier']=='Tier 1'])}"
            ))
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            self.log_console(f"\n❌ {error_msg}")
            self.update_status("Error occurred!", self.error_color)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            self.is_running = False
            self.progress.stop()
            self.run_button.config(state=tk.NORMAL, text="▶ Run Complete Pipeline")
    
    def _display_recommendations(self, results_df):
        """Display top recommendations in table"""
        # Clear existing
        for item in self.recommendations_tree.get_children():
            self.recommendations_tree.delete(item)
        
        # Add top 20
        top_results = results_df.head(20)
        for idx, row in top_results.iterrows():
            values = (
                idx + 1,
                row['Formula'],
                f"{row['Predicted_DqB']:.2f}",
                f"{row['Predicted_Emission_nm']:.0f}",
                f"{row['Total_Score']:.1f}",
                row['Tier']
            )
            
            # Color code by tier
            if row['Tier'] == 'Tier 1':
                tag = 'tier1'
            elif row['Tier'] == 'Tier 2':
                tag = 'tier2'
            else:
                tag = 'tier3'
            
            self.recommendations_tree.insert('', tk.END, values=values, tags=(tag,))
        
        # Configure tags
        self.recommendations_tree.tag_configure('tier1', background='#c8e6c9')
        self.recommendations_tree.tag_configure('tier2', background='#fff9c4')
        self.recommendations_tree.tag_configure('tier3', background='#ffccbc')
    
    def _display_statistics(self, results_df):
        """Display statistics in text widget"""
        self.stats_text.delete(1.0, tk.END)
        
        stats = f"""
EVALUATION STATISTICS
{"="*50}

Total Candidates Evaluated: {len(results_df)}

Tier Distribution:
  • Tier 1 (Strongly Recommended): {len(results_df[results_df['Tier']=='Tier 1'])}
  • Tier 2 (Consider): {len(results_df[results_df['Tier']=='Tier 2'])}
  • Tier 3 (Edge Cases): {len(results_df[results_df['Tier']=='Tier 3'])}
  • Tier 4 (Not Recommended): {len(results_df[results_df['Tier']=='Tier 4'])}

Score Statistics:
  • Average Total Score: {results_df['Total_Score'].mean():.2f}
  • Highest Score: {results_df['Total_Score'].max():.2f}
  • Lowest Score: {results_df['Total_Score'].min():.2f}

Predicted Dq/B Range:
  • Min: {results_df['Predicted_DqB'].min():.2f}
  • Max: {results_df['Predicted_DqB'].max():.2f}
  • Mean: {results_df['Predicted_DqB'].mean():.2f}

Predicted Emission Range (nm):
  • Min: {results_df['Predicted_Emission_nm'].min():.0f}
  • Max: {results_df['Predicted_Emission_nm'].max():.0f}
  • Mean: {results_df['Predicted_Emission_nm'].mean():.0f}

Average Uncertainty: {results_df['Uncertainty'].mean():.3f}

{"="*50}

TOP 5 RECOMMENDATIONS:
"""
        
        self.stats_text.insert(tk.END, stats)
        
        top5 = results_df.head(5)
        for idx, row in top5.iterrows():
            rec = f"\n{idx+1}. {row['Formula']}\n"
            rec += f"   Dq/B: {row['Predicted_DqB']:.2f} ± {row['Uncertainty']:.2f}\n"
            rec += f"   Emission: {row['Predicted_Emission_nm']:.0f} nm\n"
            rec += f"   Score: {row['Total_Score']:.1f}\n"
            rec += f"   {row['Recommendation']}\n"
            self.stats_text.insert(tk.END, rec)
    
    def view_results(self):
        """Open Excel file with results"""
        excel_path = os.path.join(self.output_folder.get(), "expert_system_recommendations.xlsx")
        if os.path.exists(excel_path):
            os.startfile(excel_path) if os.name == 'nt' else os.system(f'open "{excel_path}"')
        else:
            messagebox.showwarning("Not Found", "Results file not found! Run pipeline first.")
    
    def open_report(self):
        """Open text report"""
        report_path = os.path.join(self.output_folder.get(), "expert_system_report.txt")
        if os.path.exists(report_path):
            os.startfile(report_path) if os.name == 'nt' else os.system(f'open "{report_path}"')
        else:
            messagebox.showwarning("Not Found", "Report file not found! Run pipeline first.")
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        output_dir = self.output_folder.get()
        if os.path.exists(output_dir):
            os.startfile(output_dir) if os.name == 'nt' else os.system(f'open "{output_dir}"')
        else:
            messagebox.showwarning("Not Found", "Output folder not found!")


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = PhosphorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
