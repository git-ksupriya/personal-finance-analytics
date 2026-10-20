const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL ||
    "http://127.0.0.1:8000";

const TOKEN_KEY = "finance_access_token";
const USER_KEY = "finance_user";

// ==================================================
// SESSION MANAGEMENT
// ==================================================

export function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

export function saveToken(token) {
    if (!token) {
        throw new Error("Access token is missing.");
    }

    localStorage.setItem(TOKEN_KEY, token);
}

export function getSavedUser() {
    const user = localStorage.getItem(USER_KEY);

    if (!user) {
        return null;
    }

    try {
        return JSON.parse(user);
    } catch {
        localStorage.removeItem(USER_KEY);
        return null;
    }
}

export function saveUser(user) {
    localStorage.setItem(
        USER_KEY,
        JSON.stringify(user)
    );
}

export function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

export function isLoggedIn() {
    return Boolean(getToken());
}


// ==================================================
// CENTRAL API REQUEST HANDLER
// ==================================================

export async function apiRequest(
    endpoint,
    options = {}
) {
    const headers = new Headers(
        options.headers || {}
    );

    const token = getToken();

    if (token) {
        headers.set(
            "Authorization",
            `Bearer ${token}`
        );
    }

    const isFormData =
        options.body instanceof FormData;

    if (
        options.body !== undefined &&
        !isFormData &&
        !headers.has("Content-Type")
    ) {
        headers.set(
            "Content-Type",
            "application/json"
        );
    }

    let response;

    try {
        response = await fetch(
            `${API_BASE_URL}${endpoint}`,
            {
                ...options,
                headers,
            }
        );
    } catch {
        throw new Error(
            "Unable to connect to the backend. " +
            "Check the backend address and availability."
        );
    }

    const responseText = await response.text();

    let data = null;

    if (responseText) {
        try {
            data = JSON.parse(responseText);
        } catch {
            data = responseText;
        }
    }

    if (!response.ok) {
        const message =
            data && typeof data === "object"
                ? data.detail || data.message
                : null;

        const error = new Error(
            message ||
            `Request failed with status ${response.status}.`
        );

        error.status = response.status;

        if (response.status === 401 && token) {
            clearSession();
        }

        throw error;
    }

    return data;
}


// ==================================================
// AUTHENTICATION
// ==================================================

export async function registerUser(userData) {
    return apiRequest(
        "/auth/register",
        {
            method: "POST",
            body: JSON.stringify(userData),
        }
    );
}

export async function loginUser(email, password) {
    const data = await apiRequest(
        "/auth/login",
        {
            method: "POST",
            body: JSON.stringify({
                email,
                password,
            }),
        }
    );

    if (!data?.access_token) {
        throw new Error(
            "The login response did not contain an access token."
        );
    }

    saveToken(data.access_token);

    if (data.user) {
        saveUser(data.user);
    }

    return data;
}

export async function getCurrentUser() {
    const data = await apiRequest("/auth/me");

    saveUser(data);

    return data;
}

export function logoutUser() {
    clearSession();
}


// ==================================================
// TRANSACTIONS
// ==================================================

export async function getTransactions(params = {}) {
    const query = new URLSearchParams(params);
    const suffix = query.toString()
        ? `?${query.toString()}`
        : "";

    return apiRequest(
        `/transactions/${suffix}`
    );
}

export async function getTransaction(transactionId) {
    return apiRequest(
        `/transactions/${encodeURIComponent(transactionId)}`
    );
}

export async function addTransaction(transaction) {
    return apiRequest(
        "/transactions/",
        {
            method: "POST",
            body: JSON.stringify(transaction),
        }
    );
}

export async function importTransactionsCSV(file) {
    if (!(file instanceof File)) {
        throw new Error("Please select a CSV file.");
    }

    const formData = new FormData();

    formData.append("file", file);

    return apiRequest(
        "/transactions/import-csv",
        {
            method: "POST",
            body: formData,
        }
    );
}

export async function getCategories() {
    return apiRequest("/transactions/categories");
}

export async function getPaymentModes() {
    return apiRequest("/transactions/payment-modes");
}

export async function getLocations() {
    return apiRequest("/transactions/locations");
}


// ==================================================
// ANALYTICS
// ==================================================

export async function getAnalyticsSummary(days = 30) {
    return apiRequest(
        `/analytics/summary?days=${encodeURIComponent(days)}`
    );
}

export async function getDailySpending(days = 30) {
    return apiRequest(
        `/analytics/daily-spending?days=${encodeURIComponent(days)}`
    );
}

export async function getCategoryBreakdown(days = 30) {
    return apiRequest(
        `/analytics/category-breakdown?days=${encodeURIComponent(days)}`
    );
}

export async function getBudgetComparison() {
    return apiRequest(
        "/analytics/budget-comparison"
    );
}

export async function getRecentTransactions(limit = 10) {
    return apiRequest(
        `/analytics/recent-transactions?limit=${encodeURIComponent(limit)}`
    );
}

export async function getWeeklySummary() {
    return apiRequest(
        "/analytics/weekly-summary"
    );
}


// ==================================================
// WEEKLY BUDGETS
// ==================================================

export async function getWeeklyBudgets(includePast = false) {
    return apiRequest(
        `/budgets/?include_past=${includePast}`
    );
}

export async function getWeeklyBudget(budgetId) {
    return apiRequest(
        `/budgets/${encodeURIComponent(budgetId)}`
    );
}

export async function saveWeeklyBudget(budget) {
    return apiRequest(
        "/budgets/",
        {
            method: "POST",
            body: JSON.stringify(budget),
        }
    );
}

export async function deleteWeeklyBudget(budgetId) {
    return apiRequest(
        `/budgets/${encodeURIComponent(budgetId)}`,
        {
            method: "DELETE",
        }
    );
}


// ==================================================
// AI PREDICTIONS
// ==================================================

export async function getPredictions(
    days = 7,
    predictionType = "overall"
) {
    const query = new URLSearchParams({
        days: String(days),
        prediction_type: predictionType,
    });

    return apiRequest(
        `/ai/predictions?${query.toString()}`
    );
}

export async function generatePredictions() {
    return apiRequest(
        "/ai/predict",
        {
            method: "POST",
        }
    );
}

export async function getBudgetRisk() {
    return apiRequest("/ai/budget-risk");
}


// ==================================================
// ALERTS
// ==================================================

export async function getAlerts(unreadOnly = false) {
    return apiRequest(
        `/ai/alerts?unread_only=${unreadOnly}`
    );
}

export async function markAlertAsRead(alertId) {
    return apiRequest(
        `/ai/alerts/${encodeURIComponent(alertId)}/read`,
        {
            method: "PATCH",
        }
    );
}


// ==================================================
// CONFIGURATION
// ==================================================

export { API_BASE_URL };