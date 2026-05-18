# 🔬 Phosphor-GUI

> **Modern Graphical Interface for Cr³⁺ Phosphor Discovery Expert System**  
> Streamlit web app + Tkinter desktop app for ML-powered materials prediction and synthesis candidate selection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**Phosphor-GUI** provides intuitive graphical interfaces for the Cr³⁺ phosphor discovery expert system, enabling researchers to:

- 🤖 **Run ML predictions** using dual CatBoost + Neural Network models
- 🎯 **Expert evaluation** with physics-informed scoring (Tanabe-Sugano diagrams)
- 📊 **Interactive visualization** of candidates and recommendations
- 🏆 **Tier-based ranking** for optimal synthesis candidate selection
- 📥 **Export results** to Excel/CSV with comprehensive reports

---

## ✨ Features

### 🌐 Streamlit Web App (Recommended)
- **Modern responsive interface** that works in any browser
- **Interactive visualizations** with Plotly (zoom, hover, export)
- **Real-time progress tracking** during ML training
- **Drag-and-drop file upload** for datasets
- **One-click download** of results (Excel/CSV)
- **Multi-tab interface**: Overview, Recommendations, Statistics, Full Results
- **Cloud-ready**: Deploy to Streamlit Cloud for free

### 🖥️ Tkinter Desktop App
- **Classic desktop application** with no browser required
- **File browser dialogs** for dataset selection
- **Console output** with live logging
- **Progress bars** for long-running tasks
- **Quick actions**: Open results, reports, output folder
- **Cross-platform**: Windows, macOS, Linux

---

## 🚀 Quick Start

### Option 1: Streamlit Web App (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/KirkaSSS/Phosphor-GUI.git
cd Phosphor-GUI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app
streamlit run streamlit_app.py

# 4. Open in browser (auto-opens at http://localhost:8501)
```

### Option 2: Tkinter Desktop App

```bash
# 1. Clone and install (same as above)
git clone https://github.com/KirkaSSS/Phosphor-GUI.git
cd Phosphor-GUI
pip install -r requirements.txt

# 2. Run desktop app
python tkinter_app.py
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install -r requirements.txt
```

**Core packages:**
- `streamlit` - Web interface framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `torch` - Neural network backend
- `catboost` - Gradient boosting models
- `scikit-learn` - ML utilities
- `openpyxl` - Excel file handling

---

## 📖 Usage

### Streamlit Interface

1. **Launch app**: `streamlit run streamlit_app.py`
2. **Upload files** in sidebar:
   - Training dataset (.xlsx) with known Dq/B values
   - Prediction dataset (.xlsx) with new candidates
3. **Configure settings** (optional):
   - Target Dq/B range (default: 2.8-3.8 for red phosphors)
   - Target emission range (default: 650-720 nm)
   - Random state for reproducibility
4. **Run pipeline**: Click "▶️ Run Complete Pipeline"
5. **Review results**:
   - Top 10 recommendations with scores
   - Statistical analysis and visualizations
   - Download full results (Excel/CSV)

### Tkinter Interface

1. **Launch app**: `python tkinter_app.py`
2. **Select files**:
   - Browse training dataset
   - Browse prediction dataset
   - Choose output folder
3. **Adjust settings** in left panel
4. **Run**: Click "▶ Run Complete Pipeline"
5. **View results** in tabs:
   - Console output (live logging)
   - Top recommendations table
   - Statistics summary

---

## 📊 Input Data Format

### Training Dataset (`training_data.xlsx`)

| Formula | Feature_1 | Feature_2 | ... | Feature_N | Dq/B |
|---------|-----------|-----------|-----|-----------|------|
| LaAlO₃  | 2.015     | 89.5      | ... | 0.32      | 3.15 |
| CaTiO₃  | 1.968     | 90.0      | ... | 0.28      | 2.85 |

**Requirements:**
- Column 1: Chemical formula (string)
- Columns 2 to N-1: Structural/optical descriptors (numeric)
- Last column: Experimental Dq/B value (float)

### Prediction Dataset (`prediction_data.xlsx`)

| Formula   | Feature_1 | Feature_2 | ... | Feature_N |
|-----------|-----------|-----------|-----|-----------|
| Ca₂MgWO₆  | 2.105     | 88.2      | ... | 0.35      |
| Sr₂ScNbO₆ | 2.142     | 87.5      | ... | 0.38      |

**Requirements:**
- Column 1: Chemical formula (string)
- Columns 2 to N: Same descriptors as training set (same order!)
- No Dq/B column (will be predicted)

---

## 📈 Output Files

After running the pipeline, you'll get:

### Excel Report (`expert_system_recommendations.xlsx`)
Comprehensive evaluation with columns:
- Formula
- Predicted Dq/B (from both CatBoost and Neural Network)
- Uncertainty estimate
- Predicted emission wavelength (nm)
- Performance, Confidence, Feasibility, Novelty scores
- Total composite score
- Tier classification (1-4)
- Recommendation and rationale

### Text Report (`expert_system_report.txt`)
Summary with:
- Tier distribution statistics
- Top 10 recommendations
- Rationale for each candidate

### Visualizations (Streamlit only)
- Score distribution histogram
- Tier pie chart
- Interactive emission vs Dq/B scatter plot

---

## 🎨 Screenshots

### Streamlit Interface

**Overview Tab:**
![Overview](assets/screenshots/streamlit_overview.png)

**Top Recommendations:**
![Recommendations](assets/screenshots/streamlit_recommendations.png)

**Statistics Dashboard:**
![Statistics](assets/screenshots/streamlit_statistics.png)

### Tkinter Interface

**Main Window:**
![Tkinter Main](assets/screenshots/tkinter_main.png)

---

## 🏆 Tier Classification System

| Tier | Score Range | Meaning | Action |
|------|-------------|---------|--------|
| **Tier 1** | 75-100 | High confidence, excellent properties, feasible synthesis | **SYNTHESIZE** priority candidates |
| **Tier 2** | 65-74 | Good properties, higher uncertainty or complex synthesis | **CONSIDER** as secondary targets |
| **Tier 3** | 55-64 | Edge cases, useful for model validation | Synthesize **1-2 for testing** model limits |
| **Tier 4** | <55 | Poor properties, low confidence, or infeasible | **DO NOT SYNTHESIZE** |

---

## ⚙️ Configuration

### Target Ranges

Customize for different phosphor types:

**Red phosphors (default):**
```python
Dq/B: 2.8 - 3.8
Emission: 650 - 720 nm
```

**NIR phosphors:**
```python
Dq/B: 2.0 - 2.6
Emission: 700 - 850 nm
```

**Deep red phosphors:**
```python
Dq/B: 3.8 - 4.5
Emission: 620 - 650 nm
```

### Scoring Weights

In `expert_system_scoring.py`, adjust composite score calculation:

```python
total_score = (
    0.40 * performance_score +    # Optical properties
    0.30 * confidence_score +     # ML prediction reliability
    0.20 * feasibility_score +    # Synthesis difficulty
    0.10 * novelty_score          # Literature coverage
)
```

---

## 🌐 Cloud Deployment (Streamlit)

Deploy your GUI to the cloud for free:

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `KirkaSSS/Phosphor-GUI`
6. Set main file: `streamlit_app.py`
7. Click "Deploy"

Your app will be live at: `https://phosphor-gui.streamlit.app`

Share the URL with collaborators - no installation needed! 🎉

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Streamlit won't open in browser
**Solution:**
```bash
# Manually open: http://localhost:8501
# Or specify browser:
streamlit run streamlit_app.py --server.headless true
```

### Problem: "Address already in use"
**Solution:**
```bash
# Use different port:
streamlit run streamlit_app.py --server.port 8502
```

### Problem: Tkinter not working on macOS
**Solution:**
Use Streamlit instead - Tkinter has known issues on some macOS versions.

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[User Guide](docs/USER_GUIDE.md)** - Step-by-step usage tutorial
- **[API Reference](docs/API.md)** - Function documentation
- **[Example Workflows](docs/EXAMPLES.md)** - Common use cases

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- 🎨 UI/UX improvements
- 📊 Additional visualizations
- 🔧 New features (batch processing, API endpoints)
- 🐛 Bug fixes
- 📖 Documentation improvements

Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Author

**Snežana (Miladinović, Dragan) Đurković**  
*PhD Candidate*

**Affiliation:**  
Institute for Nuclear Sciences "Vinča"  
University of Belgrade, Serbia

**Research Group:**  
OMAS (Optical Materials and Spectroscopy Group)  
Supervisor: Prof. Dr. Miroslav D. Dramićanin

**Contact:**  
📧 snezana.djurkovic@vin.bg.ac.rs  
🔗 [GitHub](https://github.com/KirkaSSS)  
🔗 [Main Research Repo](https://github.com/KirkaSSS/phD-AI)

---

## 🙏 Acknowledgments

- **OMAS Group** for research support
- **Materials Project** and **Crystallography Open Database** for structural data
- **Streamlit** team for excellent framework
- **PyTorch** and **CatBoost** communities

---

## 📊 Related Projects

- 🔬 [phD-AI](https://github.com/KirkaSSS/phD-AI) - Core ML models and expert system
- 📚 [Research Publications](#) - Coming soon

---

## ⭐ Star History

If you find this project useful, please consider starring it! ⭐

---

<div align="center">

**Built with 🧪 for accelerated materials discovery**

[Report Bug](https://github.com/KirkaSSS/Phosphor-GUI/issues) · [Request Feature](https://github.com/KirkaSSS/Phosphor-GUI/issues) · [Documentation](docs/)

</div>
