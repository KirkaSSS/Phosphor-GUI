# GUI Installation & Usage Guide

## 🎨 Available GUI Options

Tvoj ekspertski sistem sada ima **DVA GUI-ja**:

1. **Tkinter GUI** (`phosphor_expert_system_gui.py`) - Desktop aplikacija
2. **Streamlit GUI** (`phosphor_streamlit_app.py`) - Web-based interfejs ⭐ **PREPORUČENO**

---

## ⭐ **PREPORUKA: Streamlit GUI**

**Zašto?**
- ✅ Najlakša instalacija
- ✅ Moderan izgled
- ✅ Radi na svim operativnim sistemima
- ✅ Interaktivne visualizacije (Plotly)
- ✅ Automatski download rezultata
- ✅ Live preview u browser-u

---

## 📦 **Instalacija**

### **Za Streamlit GUI** (Preporučeno)

```bash
# Instalacija samo jednom
pip install streamlit plotly openpyxl
```

**To je sve!** 🎉

### **Za Tkinter GUI** (Desktop)

```bash
# Tkinter dolazi sa Python-om, samo dodaj:
pip install matplotlib
```

---

## 🚀 **Pokretanje**

### **Streamlit GUI** (Web Interface)

#### **Korak 1: Pozicioniraj se u folder**
```bash
cd C:\Users\YourUsername\Desktop\phD-AI
```

#### **Korak 2: Pokreni aplikaciju**
```bash
streamlit run phosphor_streamlit_app.py
```

#### **Korak 3: Otvori u browser-u**
- Automatski će se otvoriti `http://localhost:8501`
- Ako se ne otvori, kopiraj link iz terminala

#### **Korak 4: Koristi interfejs**
1. Upload training dataset
2. Upload prediction dataset
3. Podesi parametre (opciono)
4. Klikni **"Run Complete Pipeline"**
5. Download rezultate

---

### **Tkinter GUI** (Desktop App)

```bash
cd C:\Users\YourUsername\Desktop\phD-AI
python phosphor_expert_system_gui.py
```

Desktop window će se otvoriti direktno.

---

## 📂 **Struktura Fajlova za GUI**

```
phD-AI/
├── phosphor_expert_system_gui.py      ← Desktop GUI
├── phosphor_streamlit_app.py          ← Web GUI ⭐
├── expert_system_scoring.py           ← Potreban!
├── integrated_prediction_pipeline.py  ← Potreban!
│
├── Cr3_dqb_training_set.xlsx         ← Training data
├── To_predict.xlsx                    ← Prediction data
└── (ostali Python fajlovi...)
```

**VAŽNO:** GUI fajlovi **MORAJU** biti u istom folderu sa `expert_system_scoring.py` i `integrated_prediction_pipeline.py`!

---

## 🖥️ **Streamlit GUI - Features**

### **Tab 1: Overview**
- Status dashboard
- Quick metrics (candidates, Tier 1 count)
- Usage instructions
- Workflow diagram

### **Tab 2: Top Recommendations**
- Top 10 candidates
- Color-coded by tier (green = Tier 1, yellow = Tier 2)
- Score, Dq/B, emission wavelength
- Rationale for each recommendation

### **Tab 3: Statistics**
- Tier distribution charts
- Score histograms
- Interactive scatter plots (Emission vs Dq/B)
- Summary metrics

### **Tab 4: Full Results**
- Filterable table (by Tier, score)
- Download buttons (Excel, CSV)
- Show top N rows option

---

## 🖼️ **Tkinter GUI - Features**

### **Left Panel:**
- File selection (training, prediction, output folder)
- Expert system settings (Dq/B range, emission range)
- Random state configuration
- Run button with progress bar

### **Right Panel - Tabs:**
1. **Console Output** - Live log messages
2. **Top Recommendations** - Table with top 20
3. **Statistics** - Summary text report
4. **Visualization** - (placeholder for future plots)

### **Quick Actions:**
- View Results (open Excel)
- Open Report (open text file)
- Open Output Folder

---

## 📊 **Comparison: Streamlit vs Tkinter**

| Feature | Streamlit | Tkinter |
|---------|-----------|---------|
| **Installation** | `pip install streamlit plotly` | Built-in with Python |
| **Interface** | Modern web UI | Classic desktop |
| **Visualizations** | Interactive Plotly charts | Basic matplotlib |
| **File handling** | Upload in browser | Browse from filesystem |
| **Download** | In-browser download | Files saved to disk |
| **Cross-platform** | ✅ Windows, Mac, Linux | ✅ All platforms |
| **Multi-user** | ✅ Can share URL | ❌ Single user |
| **Easy sharing** | ✅ Deploy to cloud | ❌ Need installer |

---

## 🎯 **Quick Start - Streamlit (5 minuta)**

```bash
# 1. Instalacija (jednom)
pip install streamlit plotly openpyxl

# 2. Navigacija
cd your_project_folder

# 3. Pokretanje
streamlit run phosphor_streamlit_app.py

# 4. Otvori browser na http://localhost:8501

# 5. Upload files i klikni Run!
```

---

## ⚙️ **Podešavanja - Streamlit**

### **Sidebar (leva strana):**

#### **Input Files:**
- Klikni "Browse files" za training dataset
- Klikni "Browse files" za prediction dataset

#### **Target Ranges:**
- **Dq/B Range:** 2.8 - 3.8 (default za crvene fosfore)
- **Emission Range:** 650 - 720 nm (default)

**Za NIR fosfore:**
```
Dq/B: 2.0 - 2.6
Emission: 700 - 850 nm
```

#### **ML Settings:**
- **Random State:** 42 (default)
- **Optimize:** Ne (osim ako imaš 30+ minuta)

---

## 🐛 **Troubleshooting**

### **Problem: "ModuleNotFoundError: No module named 'streamlit'"**

**Rešenje:**
```bash
pip install streamlit plotly openpyxl
```

### **Problem: "Cannot find expert_system_scoring.py"**

**Rešenje:**
- Proveri da li su svi fajlovi u istom folderu
- Koristi `ls` (Mac/Linux) ili `dir` (Windows) da proveriš:

```bash
# Trebalo bi da vidiš:
expert_system_scoring.py
integrated_prediction_pipeline.py
phosphor_streamlit_app.py
```

### **Problem: Streamlit se ne otvara u browser-u**

**Rešenje:**
1. Kopiraj URL iz terminala (npr. `http://localhost:8501`)
2. Otvori ga ručno u Chrome/Firefox

### **Problem: "Address already in use" (Streamlit)**

**Rešenje:**
```bash
# Proveri koji proces koristi port 8501
# Windows:
netstat -ano | findstr :8501

# Zatvori prethodni Streamlit ili koristi drugi port:
streamlit run phosphor_streamlit_app.py --server.port 8502
```

### **Problem: Tkinter ne radi na Mac-u**

**Rešenje:**
Tkinter ponekad ima problema na Mac-u. Koristi **Streamlit** umesto toga! 😊

---

## 📱 **Deljenje Streamlit Aplikacije**

Ako želiš da podeliš GUI sa kolegama:

### **Opcija 1: Lokalna mreža**
```bash
# Pokreni sa:
streamlit run phosphor_streamlit_app.py --server.address 0.0.0.0

# Kolege mogu pristupiti sa:
http://YOUR_IP_ADDRESS:8501
# (tvoj IP možeš naći sa: ipconfig na Windows ili ifconfig na Mac/Linux)
```

### **Opcija 2: Cloud Deployment (Streamlit Cloud)**

1. Push kod na GitHub
2. Idi na https://streamlit.io/cloud
3. Connect GitHub repo
4. Deploy!
5. Dobićeš javni URL: `https://yourapp.streamlit.app`

**Besplatno** za public repos! 🎉

---

## 🎨 **Customization**

### **Promeni boje (Streamlit):**

U `phosphor_streamlit_app.py`, nađi CSS sekciju i promeni:

```python
# Original:
color: #2196F3;  # Plava

# Promeni na:
color: #9C27B0;  # Ljubičasta
# ili:
color: #FF5722;  # Narandžasta
```

### **Dodaj logo:**

```python
# Na vrhu Streamlit app-a:
st.image("logo.png", width=200)
```

---

## 📖 **Best Practices**

### **Za Streamlit:**
1. ✅ Uvek testuj sa malim dataset-om prvo
2. ✅ Koristi progress bar za duge operacije
3. ✅ Dodaj `st.cache_data` za brže učitavanje
4. ✅ Omogući download rezultata

### **Za Tkinter:**
1. ✅ Koristi threading da ne freezuje UI
2. ✅ Dodaj progress bar
3. ✅ Log poruke u console widget
4. ✅ Validacija input-a pre pokretanja

---

## 🚀 **Next Steps**

Nakon što pokreneš GUI:

1. **Test run:** Probaj sa tvojim dataset-ima
2. **Customize:** Prilagodi boje, range-ove, itd.
3. **Share:** Podeli sa kolegama (Streamlit Cloud)
4. **Improve:** Dodaj nove feature-e po potrebi

---

## 💡 **Tips & Tricks**

### **Streamlit:**
- `Ctrl+C` u terminalu da zaustaviš aplikaciju
- App se automatski restartuje kad sačuvaš kod
- Koristi `st.sidebar` za organizaciju kontrola
- `st.cache_data` za cache-ovanje sporih operacija

### **Tkinter:**
- ESC za zatvaranje window-a
- Threading za duge operacije
- messagebox za error handling
- filedialog za file selection

---

## 📚 **Resources**

### **Streamlit:**
- Docs: https://docs.streamlit.io
- Gallery: https://streamlit.io/gallery
- Forum: https://discuss.streamlit.io

### **Tkinter:**
- Docs: https://docs.python.org/3/library/tkinter.html
- Tutorial: https://realpython.com/python-gui-tkinter/

---

## ✅ **Checklist Pre Pokretanja**

```
□ Instaliran Python 3.8+
□ Instaliran streamlit (ili matplotlib za Tkinter)
□ expert_system_scoring.py postoji
□ integrated_prediction_pipeline.py postoji
□ GUI fajl preuzet i sačuvan
□ Svi fajlovi u istom folderu
□ Training dataset spreman (.xlsx)
□ Prediction dataset spreman (.xlsx)
```

---

**Srećno sa GUI-jem! 🎉**

Ako nešto ne radi, javi mi i pomožću ti da rešimo! 🚀
