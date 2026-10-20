# Personal Finance Analytics

A Flask and PostgreSQL MVP for private transaction tracking, monthly/category budgets, saved LSTM forecasts, and budget alerts. The HTML/CSS/JavaScript frontend is served by Flask. The application loads separate monthly LSTM artifacts; it does not train models at startup.

## Run locally with Docker Compose

From this directory:

1. Create a local environment file and set a non-default session secret:

   ```powershell
   Copy-Item .env.example .env
   ```

   Edit `.env` and replace `DB_PASSWORD` and `SECRET_KEY`. The Compose defaults are only for local demonstrations.

2. Build and start the app and PostgreSQL:

   ```powershell
   docker compose up --build
   ```

   The app image installs PyTorch 2.5.1 from the official CPU wheel index; inference explicitly maps the saved LSTM weights to CPU. Docker excludes generated datasets and preprocessing outputs from the build context while retaining the trained model weights, scalers, application files, and migration files.

   On a new database volume, PostgreSQL applies all files in `dbms/migrations/` in filename order, including `006_persistent_financial_events.sql`. The database data is persisted in the `postgres_data` volume.

3. Open <http://localhost:8000>. Register, set the monthly overall/category budgets, optionally import a CSV, then use the dashboard. PostgreSQL health is available at <http://localhost:8000/api/health>. Both host-published container ports bind to `127.0.0.1`; they are not directly reachable from other machines.

For an existing PostgreSQL database that already has migrations 001–005, apply migration 006 once (this does not reset or replace existing data):

```powershell
docker compose exec -T db psql -U finance_app -d personal_finance -f /docker-entrypoint-initdb.d/006_persistent_financial_events.sql
```

If `.env` overrides the database name or user, use those values in the command. This forward migration adds recurring-payment types, persistent alert/anomaly fields, and the per-user inference cache.

## Run Flask from the `finance` Conda environment

The `finance` environment must contain packages in `backend/requirements.txt`. Install them if needed:

```powershell
conda activate finance
python -m pip install -r backend/requirements.txt
```

Start PostgreSQL with Compose, then open another PowerShell window from `Personal_Finance`:

```powershell
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_NAME = "personal_finance"
$env:DB_USER = "finance_app"
$env:DB_PASSWORD = "<the DB_PASSWORD from .env>"
$env:SECRET_KEY = "<the SECRET_KEY from .env>"
python -m backend.app
```

Visit <http://127.0.0.1:8000>. Do not use the former FastAPI entry point or run the ML training scripts for normal application startup.

## CSV import

The onboarding CSV must contain `date` (also accepts `txn_date` or `transaction_date`), `category`, and numeric positive `amount`. It accepts the existing standardized dataset format (`transaction_id,user_id,date,transaction_type,category,amount,payment_mode,location,notes`). Optional columns include `transaction_type`, `description`/`notes`, `transaction_id`, `payment_mode`, and `location`. Imported `user_id` is ignored: every imported row is assigned to the authenticated account. Categories must match the PostgreSQL category lookup, dates cannot be in the future, and malformed files are rejected before any import rows are inserted.

## Monthly LSTM preparation and training

The app's monthly forecasts require monthly artifacts trained for the monthly target. Existing transaction-step and daily artifacts are not used or overwritten. Run these commands from `Personal_Finance` in the `finance` Conda environment when preparing/retraining:

```powershell
conda activate finance
python preprocessing\preprocessing.py
python ml\lstm\train_lstm.py
python ml\lstm_categories\preprocessing\feature_extraction.py --monthly-only
python ml\lstm_categories\preprocessing\sequence_preparation.py
$env:CUDA_VISIBLE_DEVICES = "-1"
python ml\lstm_categories\train_lstm.py
```

The overall monthly weights, scaler, metrics, predictions, and report are written under `ml/lstm/monthly/`. Category monthly weights and per-category scaler/metrics/predictions are written under `ml/lstm_categories/models/monthly/<category>/`. The category preparation artifacts are written under `ml/lstm_categories/preprocessing/processed/monthly/`. The overall pipeline aggregates positive Expense amounts per user and fully covered calendar month, fills internal missing months with zero, and uses the existing 10-month input, seven-month output, and 80/20 chronological sequence split. The category pipeline aggregates monthly spending per user/category, fills absent category-months within fully covered months of that user's range with zero, and uses 12 monthly input rows with seven monthly targets and its existing per-user/category split.

The category split creates sequences only for user/category pairs with enough covered monthly history. Categories without eligible training and held-out windows are skipped; inference reports them as unavailable rather than changing the split or using a daily artifact. Both monthly model pipelines must be run before those model predictions are available.

## Model behavior and dashboard interpretation

- The overall model under `ml/lstm/monthly/` consumes the latest 10 complete monthly Expense totals from one contiguous segment of the authenticated user's data and predicts seven monthly totals. Its saved scaler is loaded with the model. The dashboard labels each output with its target calendar month and compares a forecast against a monthly budget only when the target month is the current month.
- A category model under `ml/lstm_categories/models/monthly/<category>/` consumes 12 monthly category feature rows and predicts seven monthly category totals. It is only loaded when its monthly weights and category scaler exist. Legacy transaction-step and daily model files are preserved but are not treated as monthly forecasts.
- Monthly inference excludes incomplete first/last calendar months and never bridges the unknown interval between imported history and later dashboard-entered records. Missing months within a contiguous covered segment are represented as zero spending. Insufficient history or unavailable monthly artifacts are shown as unavailable; the app does not substitute fabricated predictions.
- Dashboard refresh runs inference and updates the current user's saved predictions. Insufficient history or unavailable model artifacts are shown as unavailable; the app does not substitute fabricated predictions.
- Pie chart shares use expense transactions from the current and previous five calendar months. Actual overspending alerts are red and take precedence over yellow projected-overspend alerts.
- Monthly cash-flow totals show the base budget, typed recurring income and expenditure, total spendable amount, and net amount after recurring obligations. Legacy recurring rows with no known type are preserved but excluded from the cash-flow calculation.
- The account's overall alert/projection threshold is net spendable amount after recurring obligations; the saved base budget remains unchanged and is displayed separately. Category alerts use the corresponding category budget.
- Actual expense anomalies use daily totals. Predicted category anomalies compare monthly forecasts with monthly history, using a six-month lookback and Tukey upper fences, with scaled-MAD and twice-median fallbacks when the history has zero IQR. Active events are persisted and deduplicated per user/event.
- When CSV history ends before the first dashboard-entered transaction date (or account creation date when no manual transaction exists), the dates strictly between those points are unknown, not zero-spend. Inference and anomaly detection use a single contiguous segment and never fill or bridge that interval. The history date entered in the dashboard is the boundary; database insertion time is not used.
- Users can enter both expenses and income on the dashboard. Only expenses contribute to spending totals, category shares, and expenditure anomaly detection.

## Database and session configuration

The existing PostgreSQL migrations remain the schema source: `users`, `user_preferences`, `transactions`, `categories`, `budgets`, `model_registry`, `predictions`, `recurring_payments`, `alerts`, and `anomalies` are reused. Migration 006 extends recurring payments, alerts, and anomalies and adds `inference_runs` for caching each user's latest inference results. Credentials are stored as Werkzeug password hashes. Authentication uses Flask's signed, HTTP-only, SameSite=Lax session cookie and CSRF tokens on POST requests. User IDs for data access come only from that session.

Configure `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, and `SECRET_KEY`. The local `.env.example` uses `SESSION_COOKIE_SECURE=false` for plain-HTTP development. Set `SESSION_COOKIE_SECURE=true` behind HTTPS in a deployed environment.

### Azure VM deployment with Docker Compose

For an Azure VM deployment using this Compose file, install and configure an HTTPS reverse proxy on the VM (for example, Nginx with a valid TLS certificate) and proxy requests to `http://127.0.0.1:8000`. Set `SESSION_COOKIE_SECURE=true` in the VM's `.env` before starting the app. Keep PostgreSQL's port closed in the Azure Network Security Group; its Compose port is loopback-bound for VM-local administration only. Restrict inbound VM rules to SSH administration and HTTPS (plus HTTP only if needed for certificate issuance/redirects). Use strong, unique database and Flask secret values, and configure encrypted database backups. Do not expose port 8000 directly to the internet or treat the Compose PostgreSQL volume as a managed/high-availability database.

The dashboard's **Delete account** action requires confirmation and removes the signed-in user's account and associated transactions/import records, budgets, and analytics from PostgreSQL. It does not delete shared lookup data or other users' records.

## Tests and deployment

From `Personal_Finance` with the `finance` environment active:

```powershell
python -m unittest discover -s backend -p "test_*.py" -v
docker compose config
docker compose build app
docker compose up -d
Invoke-RestMethod http://localhost:8000/api/health
```

For an existing database, these commands preserve the existing PostgreSQL volume and data; do not use `docker compose down -v`. The Docker image copies the `ml/lstm` and `ml/lstm_categories/models` directories, including monthly artifacts generated before the image build. If the monthly model files have not been generated, corresponding predictions remain unavailable. No Azure deployment is performed or implied by this repository configuration.
