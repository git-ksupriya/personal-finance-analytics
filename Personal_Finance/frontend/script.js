async function loadData(url, tableId, columns) {

    const tableBody = document.getElementById(tableId);

    try {

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error("API request failed");
        }

        const data = await response.json();

        if (data.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="${columns.length}">
                        No data available
                    </td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = data.map(row => {

            return `
                <tr>
                    ${columns.map(column => `
                        <td>${row[column] ?? ""}</td>
                    `).join("")}
                </tr>
            `;

        }).join("");

    } catch (error) {

        console.error(error);

        tableBody.innerHTML = `
            <tr>
                <td colspan="${columns.length}">
                    Unable to load data from database.
                </td>
            </tr>
        `;
    }
}


// ========================================================
// CATEGORIES
// ========================================================

async function loadCategories() {

    const response = await fetch("/api/categories");

    const data = await response.json();

    const table = document.getElementById("categoriesTable");

    table.innerHTML = data.map(row => `
        <tr>
            <td>${row.category}</td>
        </tr>
    `).join("");
}


// ========================================================
// TOP TRANSACTIONS
// ========================================================

async function loadTopTransactions() {

    await loadData(
        "/api/top-transactions",
        "topTransactionsTable",
        [
            "transaction_id",
            "user_id",
            "category",
            "amount",
            "transaction_type",
            "date",
            "description"
        ]
    );
}


// ========================================================
// CATEGORY SUMMARY
// ========================================================

async function loadCategorySummary() {

    await loadData(
        "/api/category-summary",
        "categorySummaryTable",
        [
            "category",
            "transaction_count",
            "total_amount"
        ]
    );
}


// ========================================================
// TRANSACTION DETAILS
// ========================================================

async function loadTransactionDetails() {

    await loadData(
        "/api/transaction-details",
        "transactionDetailsTable",
        [
            "transaction_id",
            "user_id",
            "amount",
            "transaction_type",
            "date",
            "description",
            "category"
        ]
    );
}


// ========================================================
// USER SPENDING
// ========================================================

async function loadUserSpending() {

    await loadData(
        "/api/user-spending",
        "userSpendingTable",
        [
            "user_id",
            "date",
            "amount",
            "transaction_type"
        ]
    );
}


// ========================================================
// MAX CATEGORY FOR USER
// ========================================================

async function findMaxCategory() {

    const userId = document.getElementById("userId").value.trim();

    if (!userId) {
        alert("Please enter a User ID, for example U018");
        return;
    }

    try {

        const response = await fetch(
            `/api/max-category/${encodeURIComponent(userId)}`
        );

        const data = await response.json();

        const result = document.getElementById("maxCategoryResult");

        if (!response.ok) {
            result.innerHTML = `
                <p>${data.detail}</p>
            `;
            return;
        }

        result.innerHTML = `
            <div class="result-card">
                <h3>Highest Spending Category</h3>
                <p><strong>Category:</strong> ${data.category}</p>
                <p><strong>Amount:</strong> ₹${data.amount}</p>
            </div>
        `;

    } catch (error) {

        console.error(error);

        document.getElementById("maxCategoryResult").innerHTML =
            "<p>Unable to connect to backend.</p>";
    }
}