# Setup

## Requirements
- Python version 3.9 or higher
- Git
The following will be created or downloaded:
- `encoders.pkl`

- `customer_churn_model.pkl`
- `model_comparison.csv`
- `feature_importance.png`
- `correlation_heatmap.png`

## 1. Clone the repo

```bash
git clone https://github.com/Tech-bro-here/POKAK_TASKS.git
cd POKAK_TASKS
```
## 2. Set up a virtual environment (optional)
```bash
python -m venv venv
```
activate the env:
- windows: `venv\Scripts\activate`
- mac/linux: `source venv/bin/activate`

## 3. Install the requirements
```bash
pip install -r requirements.txt
```
## 4. Training the model (tasks 1-3)
```bash
python costumer_churning.py
```
This will clean and prepare the data, train the three models, print the evaluation table and save the following files: `encoders.pkl`, `customer_churn_model.pkl`, `model_comparison.csv`, `feature_importance.png`, `correlation_heatmap.png`
The pre-trained `.pkl` files may be found in the repo, however, by retraining the model you will overwrite them with your versions, which is useful to avoid pickle version errors.
## 5. Launching the API (task 4)
```bash
uvicorn app:app --reload
```
The API will be available at: `http://127.0.0.1:8000`
## 6. Testing the API
### Browser
Go to `http://127.0.0.1:8000/docs` and click on `POST /predict` and then on `Try it out` and `Execute`. An example request will be prefilled.
### Terminal (curl)
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```
On Windows it may be better to use the browser option or escape the quotes.
You should get a response similar to this:
```json
{"prediction": "No Churn", "churn_probability": 0.17}
```
The probability may vary between runs.
## Common issues
| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure that you've activated the virtual environment and have installed the requirements with `pip install -r requirements.txt` |
| `FileNotFoundError` for the `.csv` | Make sure that you are in the project's directory when issuing the command |
| `FileNotFoundError` for a `.pkl` when starting the API | Make sure to train the model first by running `python costumer_churning.py` |
| Pickle or version errors | Run `python costumer_churning.py` again to retrain the model |
| Port already occupied | Run `uvicorn app:app --port 8001` instead |