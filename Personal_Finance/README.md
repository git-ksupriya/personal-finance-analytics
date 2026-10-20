# Personal Finance Analytics

A Flask and PostgreSQL MVP for private transaction tracking, monthly/category budgets, saved LSTM forecasts, and budget alerts. The HTML/CSS/JavaScript frontend is served by Flask. The application uses the existing overall and category-wise LSTM artifacts; it does not train models at startup.

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

   On a new database volume, PostgreSQL applies `dbms/migrations/001_users.sql` through `005_app_auth_and_predictions.sql` in filename order. The database data is persisted in the `postgres_data` volume.

3. Open <http://localhost:8000>. Register, set the monthly overall/category budgets, optionally import a CSV, then use the dashboard. PostgreSQL health is available at <http://localhost:8000/api/health>.

For an existing PostgreSQL database, apply the new migration after the earlier schema is already installed:

```powershell
docker compose exec -T db psql -U finance_app -d personal_finance -f /docker-entrypoint-initdb.d/005_app_auth_and_predictions.sql
```

If `.env` overrides the database name or user, use those values in the command. The migration adds a unique case-insensitive `username` to existing users (initially derived from their user ID) and a prediction generation timestamp.

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

## Model behavior and dashboard interpretation

- The overall `ml/lstm/lstm_model.pt` model expects the last 10 positive transaction amounts, scaled with `ml/utils/scaler.pkl`; it outputs seven **transaction-step amounts**, not calendar-day totals. The dashboard labels these steps accurately and does not use them for monthly budget alerts.
- Each category LSTM expects 90 daily six-feature rows in the category-specific feature order and uses its matching scaler and saved weights from `ml/lstm_categories/models/<category>/`. Inference uses expense-only daily totals and zero-filled calendar days. Category forecasts require a recent expense and at least 90 calendar days of that category’s history.
- For a comparable budget-period estimate, the dashboard adds the current month's actual category spending to the mean predicted daily amount over the remaining days of the month. This is an explicitly labelled extrapolation from the next-seven-day forecast, not a native month-long LSTM output. It is unavailable when applicable category forecasts are missing.
- Dashboard refresh runs inference and updates the current user's saved predictions. Insufficient history or unavailable model artifacts are shown as unavailable; the app does not substitute fabricated predictions.
- Pie chart shares use expense transactions from the current and previous five calendar months. Actual overspending alerts are red and take precedence over yellow projected-overspend alerts.

## Database and session configuration

The existing PostgreSQL migrations remain the schema source: `users`, `user_preferences`, `transactions`, `categories`, `budgets`, `model_registry`, and `predictions` are reused. The new migration adds only the login username and prediction generation timestamp. Credentials are stored as Werkzeug password hashes. Authentication uses Flask's signed, HTTP-only, SameSite=Lax session cookie and CSRF tokens on POST requests. User IDs for data access come only from that session.

Configure `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, and `SECRET_KEY`. Set `SESSION_COOKIE_SECURE=true` behind HTTPS in a deployed environment. Use a managed persistent PostgreSQL instance and strong secret values for Azure; the Compose database volume is for local development.

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

For an existing database, these commands preserve the existing PostgreSQL volume and data; do not use `docker compose down -v`. The Docker image includes the model weights, category scalers, and model code required for inference. No Azure deployment is performed or implied by this repository configuration.
