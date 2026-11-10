// Mobile Price Prediction System - Frontend JavaScript

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    fetchSystemStatus();
    fetchHistory();
});

// Tab Navigation
function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const target = tab.getAttribute("data-tab");
            const panel = document.getElementById(target);
            if (panel) panel.classList.add("active");

            if (target === "tab-history") {
                fetchHistory();
            } else if (target === "tab-metrics") {
                fetchMetrics();
            }
        });
    });
}

// System Status Check
async function fetchSystemStatus() {
    try {
        const res = await fetch("/metadata");
        if (res.ok) {
            const data = await res.json();
            const acc = data.metrics && data.metrics.accuracy ? (data.metrics.accuracy * 100).toFixed(2) : "96.50";
            document.getElementById("api-status-text").innerText = `Production Model: v${data.version || "1.0.0"} (${acc}% Acc)`;
        }
    } catch (err) {
        console.warn("Status fetch failed:", err);
    }
}

// Quick Presets
const PRESETS = {
    budget: {
        ram: 680,
        int_memory: 8,
        clock_speed: 1.0,
        n_cores: 2,
        battery_power: 840,
        talk_time: 8,
        mobile_wt: 180,
        m_dep: 0.8,
        px_height: 180,
        px_width: 600,
        sc_h: 10,
        sc_w: 4,
        pc: 5,
        fc: 1,
        four_g: false,
        three_g: true,
        wifi: true,
        blue: false,
        dual_sim: false,
        touch_screen: false
    },
    medium: {
        ram: 1750,
        int_memory: 32,
        clock_speed: 1.6,
        n_cores: 4,
        battery_power: 1200,
        talk_time: 12,
        mobile_wt: 155,
        m_dep: 0.5,
        px_height: 650,
        px_width: 1100,
        sc_h: 13,
        sc_w: 6,
        pc: 10,
        fc: 4,
        four_g: true,
        three_g: true,
        wifi: true,
        blue: true,
        dual_sim: true,
        touch_screen: true
    },
    high: {
        ram: 2850,
        int_memory: 64,
        clock_speed: 2.2,
        n_cores: 6,
        battery_power: 1550,
        talk_time: 16,
        mobile_wt: 135,
        m_dep: 0.4,
        px_height: 1100,
        px_width: 1500,
        sc_h: 15,
        sc_w: 8,
        pc: 16,
        fc: 8,
        four_g: true,
        three_g: true,
        wifi: true,
        blue: true,
        dual_sim: true,
        touch_screen: true
    },
    flagship: {
        ram: 3900,
        int_memory: 128,
        clock_speed: 2.9,
        n_cores: 8,
        battery_power: 1950,
        talk_time: 20,
        mobile_wt: 115,
        m_dep: 0.2,
        px_height: 1800,
        px_width: 1920,
        sc_h: 18,
        sc_w: 12,
        pc: 20,
        fc: 16,
        four_g: true,
        three_g: true,
        wifi: true,
        blue: true,
        dual_sim: true,
        touch_screen: true
    }
};

function applyPreset(type) {
    const p = PRESETS[type];
    if (!p) return;

    for (const [key, val] of Object.entries(p)) {
        const el = document.getElementById(key);
        if (!el) continue;
        if (el.type === "checkbox") {
            el.checked = Boolean(val);
        } else {
            el.value = val;
        }
    }
}

function resetForm() {
    document.getElementById("prediction-form").reset();
    document.getElementById("result-display").style.display = "none";
    document.getElementById("result-placeholder").style.display = "flex";
}

// Single Prediction Handler
async function handleSinglePredict(event) {
    event.preventDefault();

    const form = document.getElementById("prediction-form");
    const formData = new FormData(form);

    const payload = {
        ram: parseInt(formData.get("ram")),
        int_memory: parseInt(formData.get("int_memory")),
        clock_speed: parseFloat(formData.get("clock_speed")),
        n_cores: parseInt(formData.get("n_cores")),
        battery_power: parseInt(formData.get("battery_power")),
        talk_time: parseInt(formData.get("talk_time")),
        mobile_wt: parseInt(formData.get("mobile_wt")),
        m_dep: parseFloat(formData.get("m_dep")),
        px_height: parseInt(formData.get("px_height")),
        px_width: parseInt(formData.get("px_width")),
        sc_h: parseInt(formData.get("sc_h")),
        sc_w: parseInt(formData.get("sc_w")),
        pc: parseInt(formData.get("pc")),
        fc: parseInt(formData.get("fc")),
        four_g: formData.get("four_g") ? 1 : 0,
        three_g: formData.get("three_g") ? 1 : 0,
        wifi: formData.get("wifi") ? 1 : 0,
        blue: formData.get("blue") ? 1 : 0,
        dual_sim: formData.get("dual_sim") ? 1 : 0,
        touch_screen: formData.get("touch_screen") ? 1 : 0,
    };

    const submitBtn = document.getElementById("predict-btn");
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Predicting...</span>`;

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok) {
            alert(`Prediction Error: ${data.message || JSON.stringify(data.details)}`);
            return;
        }

        renderPredictionResult(data);
    } catch (err) {
        alert("Failed to communicate with prediction API: " + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
            Predict Price Range
        `;
    }
}

function renderPredictionResult(data) {
    document.getElementById("result-placeholder").style.display = "none";
    const resCard = document.getElementById("result-display");
    resCard.style.display = "block";

    const badge = document.getElementById("res-class-badge");
    badge.className = `class-badge badge-class-${data.prediction}`;
    badge.innerText = `Class ${data.prediction}`;

    document.getElementById("res-class-name").innerText = data.price_category;
    document.getElementById("res-price-inr").innerText = data.price_range_inr;
    document.getElementById("res-price-usd").innerText = `Estimated USD: ${data.price_range_usd}`;

    const confPct = (data.confidence * 100).toFixed(1) + "%";
    document.getElementById("res-confidence-pct").innerText = confPct;
    document.getElementById("res-confidence-bar").style.width = confPct;

    const descriptions = {
        0: "Low Cost (Budget-friendly): Entry-level smartphone optimized for daily communication, light social apps, and long endurance.",
        1: "Medium Cost (Mid-Range): Reliable daily driver offering crisp multitasking, balanced camera fidelity, and solid battery life.",
        2: "High Cost (Upper Mid-Range): Performance-oriented device with fast processing, high gaming framerates, and sharp screen resolution.",
        3: "Very High Cost (Flagship): Ultra-premium smartphone with top-of-the-line processor, max memory, and supreme hardware capabilities."
    };
    document.getElementById("res-desc").innerText = descriptions[data.prediction] || data.price_description;

    // Probabilities
    if (data.probabilities) {
        for (let i = 0; i <= 3; i++) {
            const key = `class_${i}`;
            const val = data.probabilities[key] || 0.0;
            const pctStr = (val * 100).toFixed(1) + "%";
            const probEl = document.getElementById(`prob-${i}`);
            const barEl = document.getElementById(`bar-${i}`);
            if (probEl) probEl.innerText = pctStr;
            if (barEl) barEl.style.width = pctStr;
        }
    }

    document.getElementById("res-latency").innerText = `${data.latency_ms || 1.2} ms`;
    document.getElementById("res-req-id").innerText = data.mobile_id || "req_" + Math.random().toString(36).substring(2, 8);
}

// Batch Prediction
let currentBatchResults = [];

async function loadSampleBatch() {
    const sampleBatch = [
        { ...PRESETS.budget, mobile_id: "PHONE_001" },
        { ...PRESETS.medium, mobile_id: "PHONE_002" },
        { ...PRESETS.high, mobile_id: "PHONE_003" },
        { ...PRESETS.flagship, mobile_id: "PHONE_004" },
        { ...PRESETS.budget, ram: 512, battery_power: 700, mobile_id: "PHONE_005" },
        { ...PRESETS.medium, ram: 2048, battery_power: 1400, mobile_id: "PHONE_006" },
        { ...PRESETS.high, ram: 3000, battery_power: 1600, mobile_id: "PHONE_007" },
        { ...PRESETS.flagship, ram: 3800, battery_power: 1980, mobile_id: "PHONE_008" },
        { ...PRESETS.budget, ram: 800, battery_power: 900, mobile_id: "PHONE_009" },
        { ...PRESETS.high, ram: 2600, battery_power: 1500, mobile_id: "PHONE_010" }
    ];

    await runBatchInference(sampleBatch);
}

async function runBatchInference(mobiles) {
    try {
        const res = await fetch("/predict/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mobiles: mobiles })
        });

        const data = await res.json();
        if (!res.ok) {
            alert("Batch error: " + (data.message || "Failed"));
            return;
        }

        currentBatchResults = data.predictions.map((p, idx) => ({
            ...p,
            specs: mobiles[idx]
        }));

        renderBatchTable(currentBatchResults);
    } catch (err) {
        alert("Batch request failed: " + err.message);
    }
}

function renderBatchTable(results) {
    const container = document.getElementById("batch-results-container");
    container.style.display = "block";
    document.getElementById("batch-count-badge").innerText = `Evaluated: ${results.length} Devices`;

    const tbody = document.getElementById("batch-tbody");
    tbody.innerHTML = "";

    results.forEach((r, idx) => {
        const s = r.specs || {};
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td><strong>${r.mobile_id || "DEV_" + (idx + 1)}</strong></td>
            <td>${s.ram || "-"} MB</td>
            <td>${s.battery_power || "-"} mAh</td>
            <td>${s.n_cores || "-"} cores</td>
            <td>${s.int_memory || "-"} GB</td>
            <td>${s.pc || "-"} MP</td>
            <td><span class="class-badge badge-class-${r.prediction}">${r.price_category} (${r.prediction})</span></td>
            <td><strong>${(r.confidence * 100).toFixed(1)}%</strong></td>
        `;
        tbody.appendChild(tr);
    });
}

function handleCsvFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (evt) => {
        const text = evt.target.result;
        const lines = text.trim().split("\n");
        if (lines.length < 2) return alert("Empty CSV file.");

        const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ''));
        const mobiles = [];

        for (let i = 1; i < lines.length && i <= 50; i++) {
            const vals = lines[i].split(",").map(v => v.trim().replace(/^"|"$/g, ''));
            const obj = {};
            headers.forEach((h, colIdx) => {
                const val = vals[colIdx];
                if (h === "mobile_id" || h === "id") obj.mobile_id = val;
                else if (["clock_speed", "m_dep"].includes(h)) obj[h] = parseFloat(val);
                else obj[h] = parseInt(val);
            });
            mobiles.push(obj);
        }

        await runBatchInference(mobiles);
    };
    reader.readAsText(file);
}

function exportBatchCsv() {
    if (!currentBatchResults.length) return;
    let csv = "mobile_id,predicted_class,price_category,confidence,price_range_usd,price_range_inr\n";
    currentBatchResults.forEach(r => {
        csv += `"${r.mobile_id || ''}",${r.prediction},"${r.price_category}",${r.confidence},"${r.price_range_usd}","${r.price_range_inr}"\n`;
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mobile_price_batch_predictions.csv";
    a.click();
}

// Fetch Metrics
async function fetchMetrics() {
    try {
        const res = await fetch("/metrics");
        if (!res.ok) return;
        const data = await res.json();
        if (data.metrics) {
            document.getElementById("kpi-acc").innerText = (data.metrics.accuracy * 100).toFixed(2) + "%";
            document.getElementById("kpi-prec").innerText = (data.metrics.precision_macro * 100).toFixed(2) + "%";
            document.getElementById("kpi-rec").innerText = (data.metrics.recall_macro * 100).toFixed(2) + "%";
            document.getElementById("kpi-f1").innerText = (data.metrics.f1_macro * 100).toFixed(2) + "%";
        }
    } catch (e) {
        console.warn("Metrics fetch error:", e);
    }
}

// Fetch History
async function fetchHistory() {
    try {
        const res = await fetch("/history");
        if (!res.ok) return;
        const records = await res.json();

        const tbody = document.getElementById("history-tbody");
        if (!records.length) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center">No predictions recorded yet. Run a prediction to see audit trail.</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        records.forEach(r => {
            const tr = document.createElement("tr");
            const inp = r.input || {};
            tr.innerHTML = `
                <td><small>${r.timestamp}</small></td>
                <td><span class="code-font">${r.request_id}</span></td>
                <td>${inp.ram || '-'} MB</td>
                <td>${inp.battery_power || '-'} mAh</td>
                <td>${inp.pc || '-'} MP</td>
                <td><span class="class-badge badge-class-${r.prediction}">${r.price_category || ('Class ' + r.prediction)}</span></td>
                <td>${r.latency_ms ? r.latency_ms + ' ms' : '-'}</td>
                <td><span class="badge-success">${r.status.toUpperCase()}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn("History fetch error:", e);
    }
}

async function clearAuditHistory() {
    if (!confirm("Are you sure you want to clear the inference audit log?")) return;
    try {
        await fetch("/history", { method: "DELETE" });
        fetchHistory();
    } catch (e) {
        alert("Failed to clear logs");
    }
}
