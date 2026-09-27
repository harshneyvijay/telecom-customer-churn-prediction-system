const API = "";

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
};

async function loadMetrics() {
  const res = await fetch(`${API}/api/metrics`);
  const data = await res.json();

  const headline = data[data.headline_model] || data.logistic_regression;
  const grid = document.getElementById("metric-grid");
  grid.innerHTML = `
    ${metricCard("ROC-AUC", headline.roc_auc.toFixed(3))}
    ${metricCard("F1 Score", headline.f1.toFixed(3))}
    ${metricCard("Recall", (headline.recall * 100).toFixed(1) + "%")}
    ${metricCard("Models Evaluated", "3")}
  `;

  const tbody = document.querySelector("#model-table tbody");
  tbody.innerHTML = "";
  ["logistic_regression", "random_forest", "xgboost"].forEach((key) => {
    const m = data[key];
    if (!m) return;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${m.display_name}</td>
      <td>${m.roc_auc.toFixed(3)}</td>
      <td>${m.f1.toFixed(3)}</td>
      <td>${(m.recall * 100).toFixed(1)}%</td>
      <td>${(m.precision * 100).toFixed(1)}%</td>
      <td>${(m.accuracy * 100).toFixed(1)}%</td>
      <td>${m.threshold.toFixed(2)}</td>
    `;
    tbody.appendChild(row);
  });

  renderRocChart(data);
}

function metricCard(label, value) {
  return `<div class="metric-card">
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
  </div>`;
}

function renderRocChart(data) {
  const ctx = document.getElementById("roc-chart");
  const labels = ["logistic_regression", "random_forest", "xgboost"]
    .filter((k) => data[k]).map((k) => data[k].display_name);
  const values = ["logistic_regression", "random_forest", "xgboost"]
    .filter((k) => data[k]).map((k) => data[k].roc_auc);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: "#2F5D4C", borderRadius: 4 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { min: 0.7, max: 0.9 } },
      responsive: true,
      maintainAspectRatio: false,
    },
  });
}

async function loadInsights() {
  const res = await fetch(`${API}/api/insights`);
  const data = await res.json();

  // Class distribution
  const classCtx = document.getElementById("class-chart");
  new Chart(classCtx, {
    type: "doughnut",
    data: {
      labels: ["Not Churned", "Churned"],
      datasets: [{
        data: [data.class_distribution.not_churned, data.class_distribution.churned],
        backgroundColor: ["#2F5D4C", "#B34535"],
      }],
    },
    options: { responsive: true, maintainAspectRatio: false },
  });

  // Feature importance (global)
  renderImportanceChart(data.feature_importance);

  // Probability distribution
  if (data.probability_distribution) {
    renderProbChart(data.probability_distribution);
    document.getElementById("prob-note").textContent =
      "Distribution computed on the held-out test set.";
  } else {
    document.getElementById("prob-note").textContent =
      "Run model_training/train_model.py to populate this chart with real test-set probabilities.";
  }
}

function renderImportanceChart(importance) {
  const ctx = document.getElementById("importance-chart");
  if (!importance || importance.length === 0) {
    ctx.parentElement.innerHTML =
      '<p class="placeholder-text">Run model_training/train_model.py to compute feature importance.</p>';
    return;
  }
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: importance.map((f) => f.feature),
      datasets: [{ data: importance.map((f) => f.importance), backgroundColor: "#2F5D4C" }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      responsive: true,
      maintainAspectRatio: false,
    },
  });
}

function renderProbChart(dist) {
  const ctx = document.getElementById("prob-chart");
  const bin = (arr) => {
    const bins = new Array(10).fill(0);
    arr.forEach((p) => {
      const idx = Math.min(9, Math.floor(p * 10));
      bins[idx]++;
    });
    return bins;
  };
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.9", ".9-1"],
      datasets: [
        { label: "Not Churned", data: bin(dist.not_churned), backgroundColor: "#2F5D4C99" },
        { label: "Churned", data: bin(dist.churned), backgroundColor: "#B3453599" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 10 } } } },
    },
  });
}

// ---- Single prediction ----
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("predict-error");
  errorEl.textContent = "";

  const formData = new FormData(e.target);
  const payload = {};
  formData.forEach((value, key) => {
    if (["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"].includes(key)) {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  });

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errorEl.textContent = data.error || "Prediction failed.";
      return;
    }

    renderResult(data);
    renderLocalExplanation(data.explanation);
  } catch (err) {
    errorEl.textContent = "Could not reach the prediction API.";
  }
});

function renderResult(data) {
  const card = document.getElementById("result-card");
  const pct = (data.churn_probability * 100).toFixed(0);
  card.innerHTML = `
    <div class="result-label">Churn Probability</div>
    <div class="result-probability">${pct}%</div>
    <span class="risk-badge risk-${data.risk_level}">${data.risk_level.toUpperCase()} RISK</span>
    <div class="result-outcome">Predicted outcome: <strong>${data.predicted_outcome}</strong><br>
    Model: ${MODEL_LABELS[data.model_used] || data.model_used} &middot; Threshold: ${data.threshold_used}</div>
  `;
}

function renderLocalExplanation(explanation) {
  const list = document.getElementById("local-explanation");
  if (!explanation || explanation.length === 0) {
    list.innerHTML = '<li class="placeholder-text">No explanation available.</li>';
    return;
  }
  list.innerHTML = explanation
    .map(
      (item) => `<li>
        <span>${item.feature}</span>
        <span class="${item.contribution > 0 ? "contrib-pos" : "contrib-neg"}">
          ${item.contribution > 0 ? "+" : ""}${item.contribution.toFixed(3)} (${item.direction})
        </span>
      </li>`
    )
    .join("");
}

// ---- Batch prediction ----
document.getElementById("batch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const statusEl = document.getElementById("batch-status");
  const fileInput = document.getElementById("batch-file");

  if (!fileInput.files.length) {
    statusEl.textContent = "Please choose a CSV file.";
    return;
  }

  statusEl.textContent = "Uploading and scoring...";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch(`${API}/api/predict/batch`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      statusEl.textContent = err.error || "Batch prediction failed.";
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "churn_predictions.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    statusEl.textContent = "Done — predictions downloaded as churn_predictions.csv";
  } catch (err) {
    statusEl.textContent = "Could not reach the batch prediction API.";
  }
});

loadMetrics();
loadInsights();
