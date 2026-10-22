# Network Intrusion Detection - Web Application

Flask backend + HTML/CSS/JavaScript frontend for the project
**"Network Intrusion Detection Using Random Forest and Deep Neural Networks"** (NSL-KDD dataset).

## What the app does
| Tab | Feature |
|---|---|
| Overview | Dataset stats, Random Forest vs DNN metrics, confusion matrices, ROC curves, top-15 feature importances |
| Live Detection | Load a real normal/attack connection (or type your own values) and see both models' intrusion probability |
| Batch Analysis | Upload a CSV of connections, get predictions for every row, download the results as CSV |
| About | How the pipeline works + API endpoints |

## Run it in VS Code (Windows / macOS / Linux)
You need **Python 3.9 - 3.12** (TensorFlow does not support newer versions yet - see "No TensorFlow?" below).

1. Unzip the folder and open it in VS Code (**File > Open Folder** > `nids-webapp`).
2. Open the terminal in VS Code (**Terminal > New Terminal**) and create a virtual environment:
   ```
   python -m venv .venv
   ```
   Activate it:
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`  (if blocked: `Set-ExecutionPolicy -Scope Process Bypass`)
   - Windows (cmd): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   python app.py
   ```
   The **first run trains both models automatically (about 1-3 minutes)** and saves them in `model_artifacts/`.
   Later runs start in seconds.
5. Open **http://127.0.0.1:5000** in your browser. Press `Ctrl+C` in the terminal to stop.

### No TensorFlow? (install fails / Python too new)
Use the lightweight requirements instead:
```
pip install -r requirements-lite.txt
python app.py
```
The app then builds the neural network with scikit-learn's `MLPClassifier` using the same
128-64-32 architecture. The About tab tells you which one was used. To force this mode even when
TensorFlow is installed, set the environment variable `NIDS_FORCE_MLP=1` and delete `model_artifacts/`.

### Retrain
Delete the `model_artifacts/` folder (or run `python train.py`) to retrain from `data/Train_data.csv`.

## Project structure
```
nids-webapp/
  app.py                 Flask server + REST API
  train.py               Preprocessing, training and evaluation (RF + DNN)
  data/Train_data.csv    NSL-KDD training data
  templates/index.html   Page layout
  static/style.css       Styling
  static/app.js          Frontend logic (no external libraries, works offline)
  model_artifacts/       Created on first run (models, scaler, metrics)
  requirements.txt       Packages (with TensorFlow)
  requirements-lite.txt  Packages (without TensorFlow)
```

## API
| Endpoint | Description |
|---|---|
| `GET /api/summary` | Dataset info, metrics, confusion matrices, ROC points, feature importances |
| `GET /api/sample?label=normal\|anomaly\|any` | Random real connection from the unseen test split |
| `GET /api/sample_csv` | Downloadable 100-row CSV to try the batch tab |
| `POST /api/predict` | Body `{"features": {...41 values...}}` returns RF and DNN probability + label |
| `POST /api/predict_csv` | Multipart upload field `file` (CSV). Add `?download=1` to get a CSV back |

## Method summary
- Categorical features (`protocol_type`, `service`, `flag`) are label-encoded; target `normal = 0`, `anomaly = 1`
  (anomaly is the positive class, so precision/recall describe how well attacks are caught).
- `StandardScaler` is fit on the training split only; stratified 80/20 split, `random_state = 42`.
- Random Forest: 200 trees. DNN: 128 - 64 - 32 - 1 (sigmoid), dropout 0.3, Adam, early stopping.
- A connection is flagged as an anomaly when the predicted probability is >= 0.5.

## Note on results
The Random Forest result is the same on every run. Neural-network results can differ slightly between
runs/machines (different TensorFlow versions and hardware), so the DNN numbers shown in the app may
differ by a fraction of a percent from the ones in your report.
