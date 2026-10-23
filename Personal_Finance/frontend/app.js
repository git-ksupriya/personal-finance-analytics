const csrfToken = () => document.querySelector('meta[name="csrf-token"]')?.content || "";

async function requestJson(url, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("X-CSRF-Token", csrfToken());
    if (!(options.body instanceof FormData) && options.body) {
        headers.set("Content-Type", "application/json");
    }
    const response = await fetch(url, { ...options, headers, credentials: "same-origin" });
    const data = await response.json().catch(() => ({}));
    if (response.status === 401) {
        window.location.assign("/login");
        throw new Error("Please sign in again.");
    }
    if (!response.ok) {
        throw new Error(data.error || "The request could not be completed.");
    }
    return data;
}

function showMessage(element, text, isError = false) {
    if (!element) return;
    element.textContent = text;
    element.classList.toggle("error", isError);
}

function bindAuthenticationForm(formId, endpoint) {
    const form = document.getElementById(formId);
    if (!form) return;
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = form.querySelector('button[type="submit"]');
        const message = document.getElementById("form-message");
        button.disabled = true;
        showMessage(message, "Please wait…");
        try {
            const body = Object.fromEntries(new FormData(form).entries());
            const result = await requestJson(endpoint, {
                method: "POST",
                body: JSON.stringify(body),
            });
            window.location.assign(result.redirect);
        } catch (error) {
            showMessage(message, error.message, true);
        } finally {
            button.disabled = false;
        }
    });
}

async function loadCategories() {
    return requestJson("/api/categories");
}

async function setupOnboarding() {
    const container = document.getElementById("category-budgets");
    if (!container) return;
    try {
        const categories = await loadCategories();
        const recurringContainer = document.getElementById("recurring-payments");
        const addRecurringRow = () => {
            const row = document.createElement("div");
            row.className = "recurring-row";
            const fields = [
                ["Payment type", "select", "recurring_payment_type[]", ["income", "expenditure"]],
                ["Category", "select", "recurring_category[]", categories],
                ["Description", "text", "recurring_description[]"],
                ["Amount (INR)", "number", "recurring_amount[]"],
                ["Next due date", "date", "recurring_next_due_date[]"],
            ];
            for (const [labelText, type, name, options] of fields) {
                const label = document.createElement("label");
                label.textContent = labelText;
                const input = type === "select"
                    ? document.createElement("select")
                    : document.createElement("input");
                input.name = name;
                if (type === "select") {
                    for (const value of options) {
                        const option = document.createElement("option");
                        option.value = value;
                        option.textContent = value;
                        input.append(option);
                    }
                } else {
                    input.type = type;
                    input.required = true;
                    if (type === "number") {
                        input.min = "0.01";
                        input.step = "0.01";
                    } else if (name === "recurring_description[]") {
                        input.maxLength = 500;
                    } else {
                        input.value = new Date().toISOString().slice(0, 10);
                    }
                }
                label.append(input);
                row.append(label);
            }
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "secondary";
            remove.textContent = "Remove";
            remove.addEventListener("click", () => row.remove());
            row.append(remove);
            recurringContainer.append(row);
        };
        document.getElementById("add-recurring-payment")
            ?.addEventListener("click", addRecurringRow);
        container.replaceChildren();
        for (const category of categories) {
            const label = document.createElement("label");
            label.textContent = `${category} (INR)`;
            const input = document.createElement("input");
            input.type = "number";
            input.min = "0";
            input.step = "0.01";
            input.name = `category_budget[${category}]`;
            input.placeholder = "Optional";
            label.append(input);
            container.append(label);
        }
    } catch (error) {
        showMessage(document.getElementById("form-message"), error.message, true);
    }

    const form = document.getElementById("onboarding-form");
    form?.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = form.querySelector('button[type="submit"]');
        const message = document.getElementById("form-message");
        button.disabled = true;
        showMessage(message, "Saving budgets and importing transactions…");
        try {
            const result = await requestJson("/api/onboarding", {
                method: "POST",
                body: new FormData(form),
            });
            showMessage(message, `Saved. Imported ${result.imported_transactions} transactions.`);
            window.location.assign(result.redirect);
        } catch (error) {
            showMessage(message, error.message, true);
        } finally {
            button.disabled = false;
        }
    });
}

function money(value) {
    return value == null
        ? "Not available"
        : new Intl.NumberFormat("en-IN", {
              style: "currency",
              currency: "INR",
              maximumFractionDigits: 2,
          }).format(value);
}

const chartColors = ["#2878b5", "#e49a24", "#339966", "#aa62a5", "#e15b64", "#64748b", "#14a6a6", "#8157bd", "#a47844", "#40724a"];

function renderPieChart(rows) {
    const chart = document.getElementById("category-chart");
    const legend = document.getElementById("category-legend");
    if (!chart || !legend) return;
    legend.replaceChildren();
    if (!rows.length) {
        chart.style.background = "#e5e7eb";
        legend.textContent = "No expense data in this period.";
        return;
    }
    let cursor = 0;
    const stops = [];
    rows.forEach((row, index) => {
        const color = chartColors[index % chartColors.length];
        const next = cursor + row.percentage;
        stops.push(`${color} ${cursor}% ${next}%`);
        cursor = next;
        const item = document.createElement("li");
        const swatch = document.createElement("span");
        swatch.className = "legend-swatch";
        swatch.style.backgroundColor = color;
        item.append(swatch, document.createTextNode(`${row.category}: ${row.percentage}% (${money(row.amount)})`));
        legend.append(item);
    });
    chart.style.background = `conic-gradient(${stops.join(", ")})`;
}

function renderTableRows(body, rows, emptyText, renderRow) {
    body.replaceChildren();
    if (!rows.length) {
        const row = body.insertRow();
        const cell = row.insertCell();
        cell.colSpan = body.closest("table").querySelectorAll("thead th").length;
        cell.textContent = emptyText;
        return;
    }
    for (const item of rows) {
        const row = body.insertRow();
        renderRow(row, item);
    }
}

function renderDashboard(data) {
    document.getElementById("actual-spending").textContent = money(data.actual_monthly_spending);
    document.getElementById("overall-budget").textContent = money(data.monthly_budget);
    document.getElementById("recurring-income").textContent = money(data.recurring_monthly_income);
    document.getElementById("recurring-expenditure").textContent = money(data.recurring_monthly_expenditure);
    document.getElementById("total-spendable").textContent = money(data.total_spendable_amount);
    document.getElementById("net-spendable").textContent = money(data.net_after_recurring_obligations);
    document.getElementById("projected-spending").textContent = money(data.projected_monthly_spending);
    document.getElementById("projected-budget").textContent = money(data.monthly_budget);
    document.getElementById("projection-note").textContent =
        data.projection_method || data.category_inference.reason || "Not enough applicable model history.";

    const overall = document.getElementById("overall-forecast");
    if (data.overall_lstm.available) {
        overall.replaceChildren();
        data.overall_lstm.values.forEach((value, index) => {
            const item = document.createElement("span");
            const month = data.overall_lstm.target_periods?.[index] || "Unknown month";
            item.textContent = `${month.slice(0, 7)}: ${money(value)}`;
            overall.append(item);
        });
    } else {
        overall.textContent = data.overall_lstm.reason;
    }

    renderPieChart(data.category_spending_six_months);

    const categories = Object.entries(data.category_predictions).map(([name, item]) => ({
        name, ...item,
    }));
    renderTableRows(
        document.getElementById("category-projections"),
        categories,
        data.category_inference.reason || "No monthly category forecasts available.",
        (row, item) => {
            [item.name, money(item.actual_month), money(item.projected_month), money(item.budget)]
                .forEach((value) => (row.insertCell().textContent = value));
        },
    );

    const alertList = document.getElementById("alerts-list");
    alertList.replaceChildren();
    if (!data.alerts.length) {
        const message = document.createElement("p");
        message.textContent = "No budget alerts.";
        alertList.append(message);
    } else {
        for (const alert of data.alerts) {
            const item = document.createElement("article");
            item.className = `alert alert-${alert.severity}`;
            const title = document.createElement("strong");
            title.textContent = `${alert.severity === "red" ? "Actual overspend" : "Predicted overspend"} · ${alert.scope}`;
            const text = document.createElement("p");
            text.textContent = alert.message;
            item.append(title, text);
            alertList.append(item);
        }
    }
    const anomalyList = document.getElementById("anomalies-list");
    anomalyList.replaceChildren();
    if (!data.anomalies.length) {
        const message = document.createElement("p");
        message.textContent = "No unusual spending detected.";
        anomalyList.append(message);
    } else {
        for (const anomaly of data.anomalies) {
            const item = document.createElement("article");
            item.className = "alert alert-yellow";
            const title = document.createElement("strong");
            title.textContent = `${anomaly.type === "predicted" ? "Predicted" : "Actual"} unusual spending · ${anomaly.scope}`;
            const text = document.createElement("p");
            text.textContent = `${anomaly.target_period || "Forecast"}: observed ${money(anomaly.observed_amount)}, threshold ${money(anomaly.threshold_amount)}. ${anomaly.reason || ""}`;
            item.append(title, text);
            anomalyList.append(item);
        }
    }

    const recurringList = document.getElementById("recurring-list");
    recurringList.replaceChildren();
    if (!data.recurring_payments.length) {
        recurringList.textContent = "No recurring items configured.";
    } else {
        for (const payment of data.recurring_payments) {
            const item = document.createElement("p");
            item.textContent = `${payment.payment_type || "Unclassified"} · ${payment.category} · ${payment.description} · ${money(payment.amount)} monthly`;
            recurringList.append(item);
        }
    }

    renderTableRows(
        document.getElementById("transactions-list"),
        data.transactions,
        "No transactions yet.",
        (row, item) => {
            [item.date, item.type, item.description, item.category, money(item.amount)]
                .forEach((value) => (row.insertCell().textContent = value));
        },
    );
    showMessage(
        document.getElementById("dashboard-message"),
        data.inference_errors.length
            ? "Analytics loaded; one or more model forecasts were unavailable."
            : "Analytics updated.",
        data.inference_errors.length > 0,
    );
}

async function refreshDashboard() {
    const message = document.getElementById("dashboard-message");
    showMessage(message, "Refreshing your analytics and model forecasts…");
    try {
        const data = await requestJson("/api/dashboard");
        renderDashboard(data);
    } catch (error) {
        showMessage(message, error.message, true);
    }
}

async function setupDashboard() {
    if (!document.getElementById("transaction-form")) return;
    try {
        const categories = await loadCategories();
        const select = document.getElementById("transaction-category");
        select.replaceChildren();
        for (const name of categories) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            select.append(option);
        }
    } catch (error) {
        showMessage(document.getElementById("dashboard-message"), error.message, true);
    }

    const dateInput = document.querySelector('#transaction-form input[name="date"]');
    const transactionType = document.getElementById("transaction-type");
    const transactionSubmit = document.getElementById("transaction-submit");
    if (dateInput) dateInput.value = new Date().toISOString().slice(0, 10);
    transactionType?.addEventListener("change", () => {
        transactionSubmit.textContent = `Add ${transactionType.value.toLowerCase()}`;
    });
    document.getElementById("refresh-button")?.addEventListener("click", refreshDashboard);
    document.getElementById("transaction-form")?.addEventListener("submit", async (event) => {
        event.preventDefault();
        const form = event.currentTarget;
        const button = form.querySelector('button[type="submit"]');
        const message = document.getElementById("transaction-message");
        button.disabled = true;
        showMessage(message, "Saving transaction…");
        try {
            await requestJson("/api/transactions", {
                method: "POST",
                body: JSON.stringify(Object.fromEntries(new FormData(form).entries())),
            });
            form.reset();
            if (dateInput) dateInput.value = new Date().toISOString().slice(0, 10);
            showMessage(message, "Transaction saved.");
            await refreshDashboard();
        } catch (error) {
            showMessage(message, error.message, true);
        } finally {
            button.disabled = false;
        }
    });
    await refreshDashboard();
}

document.addEventListener("DOMContentLoaded", () => {
    bindAuthenticationForm("login-form", "/api/login");
    bindAuthenticationForm("register-form", "/api/register");
    setupOnboarding();
    setupDashboard();

    document.getElementById("logout-button")?.addEventListener("click", async () => {
        try {
            const result = await requestJson("/api/logout", { method: "POST" });
            window.location.assign(result.redirect);
        } catch (error) {
            showMessage(document.getElementById("form-message"), error.message, true);
        }
    });

    document.getElementById("delete-account-button")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        if (!window.confirm("Delete your account and permanently remove your transactions, uploaded transaction records, budgets, and analytics? This cannot be undone.")) {
            return;
        }
        button.disabled = true;
        try {
            const result = await requestJson("/api/account/delete", { method: "POST" });
            window.location.assign(result.redirect);
        } catch (error) {
            button.disabled = false;
            showMessage(document.getElementById("dashboard-message"), error.message, true);
        }
    });
});
