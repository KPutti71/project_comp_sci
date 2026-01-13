# PyLBM Fluid Simulation Project

This project uses **PyLBM** (Lattice Boltzmann Method) for fluid simulations.
A **Python virtual environment** is used to ensure reproducibility and avoid
dependency conflicts.

The instructions below assume **Ubuntu/Debian** and **Python 3.10**.

---

## 1. Install system prerequisites

### Python virtual environment support
```bash
sudo apt update
sudo apt install -y python3.10-venv
```
---

## 3. Create and activate the virtual environment

From the **project root** (`project_comp_sci`):

```bash
python3 -m venv pylbm-env
source pylbm-env/bin/activate
```

Upgrade packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

---

## 4. Install Python dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import pylbm, numpy; print('pylbm ok:', pylbm.__version__, 'numpy:', numpy.__version__)"
```

---

## Reproducibility

To recreate the environment from scratch on another machine:

```bash
python3 -m venv pylbm-env
source pylbm-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---
