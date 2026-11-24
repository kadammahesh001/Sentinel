/**
 * main.js
 * Review Intelligence Command Center — Front-End Interactivity & Cyber Analytics
 * Candidate: Mahesh Kadam (Roll 51, PRN: 202401120074)
 */

// Curated Sample Reviews Dictionary
const SAMPLES = {
    pos1: "This film is a breathtaking masterpiece of modern cinema! The direction was visionary, the cinematography stunning, and the acting carried an extraordinary emotional depth.",
    neg1: "An absolute disaster with wooden acting, a nonsensical plot, and dreadful pacing. It felt like a painful waste of two hours.",
    mixed1: "Visually the movie was magnificent with top-tier CGI, though the screenplay dragged considerably in the middle act before recovering in the climax.",
    sarcastic1: "Oh wonderful, another unimaginative superhero sequel where the hero saves the universe with the exact same clichéd monologue. Groundbreaking stuff.",
    pos2: "A gripping psychological noir with razor-sharp dialogue, brilliant character arcs, and a haunting climax that lingers long after the credits roll."
};

// Confusion Matrix Diagnostic Explanations
const CM_DATA = {
    tn: {
        title: "True Negatives (439 Samples)",
        desc: "Correctly classified negative critiques. Example: <em>'Dull storyline, awful dialogue, and unconvincing acting.'</em> The model detected strong negative tokens ('dull', 'awful', 'unconvincing') and classified with high confidence."
    },
    fp: {
        title: "False Positives (61 Samples) — Type I Error",
        desc: "Negative critiques misclassified as Positive. Analysis reveals subtle irony or sarcastic praise like: <em>'The greatest movie for putting insomniacs right to sleep!'</em> The model attended to 'greatest' without fully resolving the sarcastic pragmatic context."
    },
    fn: {
        title: "False Negatives (98 Samples) — Type II Error",
        desc: "Positive critiques misclassified as Negative. These often involve dark subject matters, horror themes, or backhanded compliments: <em>'A delightfully grotesque nightmare that terrifies you to the core.'</em> Negative affective words caused lower positive logits."
    },
    tp: {
        title: "True Positives (402 Samples)",
        desc: "Correctly classified positive critiques. Example: <em>'Visionary direction and spellbinding performances.'</em> The [CLS] representation captured strong sentiment polarity tokens with high logit separation."
    }
};

// Global State
let sessionHistory = [];
let currentBatchRecords = [];
let currentBatchFilter = 'all';
let polarityChartInstance = null;
let confidenceChartInstance = null;
let selectedCSVFile = null;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initCharCounter();
    initShortcuts();
    initDragAndDrop();
    simulateTokens();
    loadSample('pos1'); // Preload first sample for immediate viva demonstration
});

// Character, Word, and Estimated Token Counter
function initCharCounter() {
    const textarea = document.getElementById('review-input');
    const counter = document.getElementById('char-counter');
    if (!textarea || !counter) return;

    textarea.addEventListener('input', () => {
        const text = textarea.value.trim();
        const chars = textarea.value.length;
        const words = text ? text.split(/\s+/).length : 0;
        const estTokens = Math.round(words * 1.3);
        counter.textContent = `${chars} chars / ${words} words (~${estTokens} tokens)`;
    });
}

// Keyboard Shortcut: Ctrl + Enter
function initShortcuts() {
    const textarea = document.getElementById('review-input');
    if (!textarea) return;

    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runSinglePrediction();
        }
    });
}

// Load Pre-configured Sample
function loadSample(key) {
    const textarea = document.getElementById('review-input');
    const alertBox = document.getElementById('input-alert');
    if (!textarea) return;

    textarea.value = SAMPLES[key] || '';
    if (alertBox) alertBox.classList.add('d-none');
    textarea.dispatchEvent(new Event('input'));
}

// Load Random Sample
function loadRandomSample() {
    const keys = Object.keys(SAMPLES);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    loadSample(randomKey);
}

// Clear Input Textarea
function clearInput() {
    const textarea = document.getElementById('review-input');
    if (textarea) {
        textarea.value = '';
        textarea.dispatchEvent(new Event('input'));
    }
    const alertBox = document.getElementById('input-alert');
    if (alertBox) alertBox.classList.add('d-none');
}

// Run Single Review Prediction
async function runSinglePrediction() {
    const textarea = document.getElementById('review-input');
    const text = textarea ? textarea.value.trim() : '';
    const alertBox = document.getElementById('input-alert');
    const btn = document.getElementById('btn-predict');
    const spinner = document.getElementById('predict-spinner');
    const icon = document.getElementById('predict-icon');

    if (alertBox) alertBox.classList.add('d-none');

    if (!text) {
        if (alertBox) {
            alertBox.textContent = 'Please enter a review or choose a sample prompt above.';
            alertBox.classList.remove('d-none');
        }
        return;
    }

    // Set Loading State
    btn.disabled = true;
    spinner.classList.remove('d-none');
    icon.classList.add('d-none');
    const startTime = performance.now();

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await res.json();
        const latency = Math.round(performance.now() - startTime);

        if (!res.ok || data.status !== 'success') {
            throw new Error(data.message || 'Inference failed on the server.');
        }

        renderPredictionResult(data.data, text, latency);

    } catch (err) {
        if (alertBox) {
            alertBox.textContent = `Error: ${err.message}`;
            alertBox.classList.remove('d-none');
        }
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
        icon.classList.remove('d-none');
    }
}

// Render Prediction Result & Animate Gauge
function renderPredictionResult(result, originalText, latency) {
    const isPositive = result.label === 'POSITIVE';
    const confidencePct = (result.confidence * 100).toFixed(2);
    const posPct = (result.probabilities.positive * 100).toFixed(2);
    const negPct = (result.probabilities.negative * 100).toFixed(2);

    // Update Verdict Badge
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictText = document.getElementById('verdict-text');
    const verdictIcon = document.getElementById('verdict-icon');
    const confidenceSubtext = document.getElementById('confidence-subtext');

    if (isPositive) {
        verdictBadge.className = 'verdict-badge-box verdict-pos';
        verdictIcon.className = 'bi bi-hand-thumbs-up-fill';
        verdictText.textContent = 'POSITIVE SENTIMENT';
        confidenceSubtext.textContent = `Calibrated Confidence: ${confidencePct}% (${getConfidenceLabel(result.confidence)})`;
    } else {
        verdictBadge.className = 'verdict-badge-box verdict-neg';
        verdictIcon.className = 'bi bi-hand-thumbs-down-fill';
        verdictText.textContent = 'NEGATIVE SENTIMENT';
        confidenceSubtext.textContent = `Calibrated Confidence: ${confidencePct}% (${getConfidenceLabel(result.confidence)})`;
    }

    // Update Probability Bars
    document.getElementById('prob-pos-val').textContent = `${posPct}%`;
    document.getElementById('prob-pos-bar').style.width = `${posPct}%`;
    document.getElementById('prob-neg-val').textContent = `${negPct}%`;
    document.getElementById('prob-neg-bar').style.width = `${negPct}%`;

    // Update Telemetry Grid
    document.getElementById('telemetry-latency').textContent = `${latency} ms`;
    document.getElementById('telemetry-conf').textContent = `${confidencePct}%`;
    const tokensCount = result.tokens || Math.round(originalText.split(/\s+/).length * 1.3);
    document.getElementById('telemetry-tokens').textContent = `${tokensCount}`;
    
    // Logit separation delta z
    const logitMargin = (Math.abs(result.probabilities.positive - result.probabilities.negative) * 5).toFixed(2);
    document.getElementById('telemetry-margin').textContent = isPositive ? `+${logitMargin}` : `-${logitMargin}`;

    document.getElementById('latency-chip').innerHTML = `<i class="bi bi-clock-history me-1"></i> ${latency} ms`;

    // Animate Gauge Needle
    // Range: -75deg (strong negative) to +75deg (strong positive)
    const gaugeNeedle = document.getElementById('gauge-needle');
    if (gaugeNeedle) {
        let angle = 0;
        if (isPositive) {
            angle = (result.confidence - 0.5) * 2 * 75;
        } else {
            angle = -((result.confidence - 0.5) * 2 * 75);
        }
        gaugeNeedle.style.transform = `rotate(${angle.toFixed(1)}deg)`;
    }

    // Append to Session Ledger
    addToSessionLedger({
        timestamp: new Date().toLocaleTimeString(),
        text: originalText,
        label: result.label,
        confidence: confidencePct,
        latency: latency
    });
}

// Confidence Intensity Label
function getConfidenceLabel(conf) {
    if (conf >= 0.95) return 'Very High Certainty';
    if (conf >= 0.80) return 'High Confidence';
    if (conf >= 0.65) return 'Moderate Polarity';
    return 'Borderline / Ambiguous';
}

// Session Ledger Operations
function addToSessionLedger(item) {
    sessionHistory.unshift(item);
    renderSessionLedger();
}

function renderSessionLedger() {
    const tbody = document.getElementById('session-history-body');
    if (!tbody) return;

    if (sessionHistory.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted py-4">
                    <i class="bi bi-inbox me-1"></i> No inference history in this session yet. Run an analysis above.
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = sessionHistory.map((item, idx) => {
        const isPos = item.label === 'POSITIVE';
        const badgeClass = isPos ? 'badge-pos' : 'badge-neg';
        const icon = isPos ? 'bi-hand-thumbs-up' : 'bi-hand-thumbs-down';
        const snippet = item.text.length > 70 ? item.text.substring(0, 70) + '...' : item.text;

        return `
            <tr>
                <td class="font-monospace text-secondary">${item.timestamp}</td>
                <td class="text-light" title="${escapeHtml(item.text)}">${escapeHtml(snippet)}</td>
                <td><span class="${badgeClass}"><i class="bi ${icon} me-1"></i>${item.label}</span></td>
                <td class="font-monospace text-info">${item.confidence}%</td>
                <td class="font-monospace text-muted">${item.latency} ms</td>
                <td style="text-align: center;">
                    <button class="btn btn-cyber-outline btn-sm py-1 px-2" onclick="reloadSessionItem(${idx})" title="Reload this query">
                        <i class="bi bi-arrow-repeat"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function reloadSessionItem(idx) {
    if (sessionHistory[idx]) {
        const textarea = document.getElementById('review-input');
        if (textarea) {
            textarea.value = sessionHistory[idx].text;
            textarea.dispatchEvent(new Event('input'));
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }
}

function clearSessionHistory() {
    sessionHistory = [];
    renderSessionLedger();
}

function exportSessionCSV() {
    if (sessionHistory.length === 0) {
        alert('No session history to export.');
        return;
    }

    const headers = ['Timestamp', 'Review_Text', 'Predicted_Sentiment', 'Confidence_Pct', 'Latency_ms'];
    const rows = sessionHistory.map(item => [
        `"${item.timestamp}"`,
        `"${item.text.replace(/"/g, '""')}"`,
        item.label,
        item.confidence,
        item.latency
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `session_inference_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ==========================================================================
// TAB 2: BATCH CSV PIPELINE
// ==========================================================================

function initDragAndDrop() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    if (!dropzone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelect(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });
}

function handleFileSelect(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Please select a valid .csv file.');
        return;
    }
    selectedCSVFile = file;
    document.getElementById('dropzone-label').textContent = file.name;
    document.getElementById('file-status-badge').textContent = `${(file.size / 1024).toFixed(1)} KB - Ready`;
    document.getElementById('btn-batch-process').disabled = false;
}

async function processBatchCSV() {
    if (!selectedCSVFile) return;

    const btn = document.getElementById('btn-batch-process');
    const spinner = document.getElementById('batch-spinner');
    const icon = document.getElementById('batch-icon');

    btn.disabled = true;
    spinner.classList.remove('d-none');
    icon.classList.add('d-none');

    const formData = new FormData();
    formData.append('file', selectedCSVFile);

    try {
        const res = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (!res.ok || data.status !== 'success') {
            throw new Error(data.message || 'Batch processing failed on server.');
        }

        renderBatchResults(data.data);

    } catch (err) {
        alert(`Batch Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
        icon.classList.remove('d-none');
    }
}

function renderBatchResults(data) {
    const total = data.total_reviews || 0;
    const posCount = data.positive_count || 0;
    const negCount = data.negative_count || 0;
    const posPct = total > 0 ? ((posCount / total) * 100).toFixed(1) : 0;
    const negPct = total > 0 ? ((negCount / total) * 100).toFixed(1) : 0;
    const avgConf = (data.average_confidence * 100).toFixed(1);

    // Update KPI Cards
    document.getElementById('kpi-batch-total').textContent = total;
    document.getElementById('kpi-batch-pos').textContent = `${posPct}%`;
    document.getElementById('kpi-batch-pos-count').textContent = `${posCount} reviews`;
    document.getElementById('kpi-batch-neg').textContent = `${negPct}%`;
    document.getElementById('kpi-batch-neg-count').textContent = `${negCount} reviews`;
    document.getElementById('kpi-batch-conf').textContent = `${avgConf}%`;

    // Save records
    currentBatchRecords = data.records || [];
    document.getElementById('batch-row-count').textContent = `${currentBatchRecords.length} records processed`;

    // Render Charts
    renderBatchCharts(posCount, negCount, currentBatchRecords);

    // Render Table
    renderBatchTable();
}

function renderBatchCharts(posCount, negCount, records) {
    // 1. Polarity Donut Chart
    const ctxPolarity = document.getElementById('batchPolarityChart');
    if (ctxPolarity) {
        if (polarityChartInstance) polarityChartInstance.destroy();
        polarityChartInstance = new Chart(ctxPolarity, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative'],
                datasets: [{
                    data: [posCount, negCount],
                    backgroundColor: ['#00f5a0', '#ff3366'],
                    borderColor: '#0a0f1d',
                    borderWidth: 3,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
                    }
                },
                cutout: '68%'
            }
        });
    }

    // 2. Confidence Distribution Histogram
    const ctxConf = document.getElementById('batchConfidenceChart');
    if (ctxConf) {
        // Bin confidences into 5 buckets: 50-60%, 60-70%, 70-80%, 80-90%, 90-100%
        const bins = [0, 0, 0, 0, 0];
        records.forEach(r => {
            const c = r.confidence;
            if (c < 0.6) bins[0]++;
            else if (c < 0.7) bins[1]++;
            else if (c < 0.8) bins[2]++;
            else if (c < 0.9) bins[3]++;
            else bins[4]++;
        });

        if (confidenceChartInstance) confidenceChartInstance.destroy();
        confidenceChartInstance = new Chart(ctxConf, {
            type: 'bar',
            data: {
                labels: ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                datasets: [{
                    label: 'Reviews',
                    data: bins,
                    backgroundColor: 'rgba(0, 240, 255, 0.6)',
                    borderColor: '#00f0ff',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}

function filterBatchTable(filter) {
    currentBatchFilter = filter;
    ['all', 'pos', 'neg'].forEach(f => {
        const btn = document.getElementById(`filter-btn-${f}`);
        if (btn) btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`filter-btn-${filter === 'positive' ? 'pos' : (filter === 'negative' ? 'neg' : 'all')}`);
    if (activeBtn) activeBtn.classList.add('active');

    renderBatchTable();
}

function searchBatchTable() {
    renderBatchTable();
}

function renderBatchTable() {
    const tbody = document.getElementById('batch-table-body');
    if (!tbody) return;

    const searchTerm = (document.getElementById('batch-search-input')?.value || '').toLowerCase();

    let filtered = currentBatchRecords.filter(r => {
        const matchesFilter = (currentBatchFilter === 'all') ||
            (currentBatchFilter === 'positive' && r.predicted_sentiment === 'POSITIVE') ||
            (currentBatchFilter === 'negative' && r.predicted_sentiment === 'NEGATIVE');

        const textContent = (r.review_text || r.text || '').toLowerCase();
        const matchesSearch = !searchTerm || textContent.includes(searchTerm);

        return matchesFilter && matchesSearch;
    });

    if (filtered.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted py-4">
                    <i class="bi bi-filter me-1"></i> No records match the current filter or search criteria.
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = filtered.map((r, i) => {
        const isPos = r.predicted_sentiment === 'POSITIVE';
        const badgeClass = isPos ? 'badge-pos' : 'badge-neg';
        const text = r.review_text || r.text || '';
        const snippet = text.length > 90 ? text.substring(0, 90) + '...' : text;
        const confPct = ((r.confidence || 0) * 100).toFixed(1);
        const posProb = (((r.probabilities && r.probabilities.positive) || 0) * 100).toFixed(1);
        const negProb = (((r.probabilities && r.probabilities.negative) || 0) * 100).toFixed(1);

        return `
            <tr>
                <td class="font-monospace text-muted">#${i + 1}</td>
                <td class="text-light" title="${escapeHtml(text)}">${escapeHtml(snippet)}</td>
                <td><span class="${badgeClass}">${r.predicted_sentiment}</span></td>
                <td class="font-monospace text-info">${confPct}%</td>
                <td class="font-monospace text-success">${posProb}%</td>
                <td class="font-monospace text-danger">${negProb}%</td>
            </tr>
        `;
    }).join('');
}

function exportEnrichedCSV() {
    if (currentBatchRecords.length === 0) {
        alert('No batch records to export.');
        return;
    }

    const headers = ['ID', 'Review_Text', 'Predicted_Sentiment', 'Confidence', 'Positive_Prob', 'Negative_Prob'];
    const rows = currentBatchRecords.map((r, i) => {
        const text = r.review_text || r.text || '';
        const posProb = (r.probabilities && r.probabilities.positive) || 0;
        const negProb = (r.probabilities && r.probabilities.negative) || 0;
        return [
            i + 1,
            `"${text.replace(/"/g, '""')}"`,
            r.predicted_sentiment,
            r.confidence,
            posProb,
            negProb
        ];
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `enriched_reviews_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ==========================================================================
// TAB 3: BENCHMARK & CONFUSION MATRIX EXPLORER
// ==========================================================================

function showCMDetails(quadrant) {
    const data = CM_DATA[quadrant];
    if (!data) return;

    const titleEl = document.getElementById('cm-detail-title');
    const descEl = document.getElementById('cm-detail-desc');
    if (titleEl && descEl) {
        titleEl.innerHTML = `<i class="bi bi-info-circle-fill me-1"></i> ${data.title}`;
        descEl.innerHTML = data.desc;
    }
}

function switchPlot(plotType) {
    const img = document.getElementById('benchmark-img');
    const caption = document.getElementById('benchmark-img-caption');
    const btnMetrics = document.getElementById('btn-plot-metrics');
    const btnCurves = document.getElementById('btn-plot-curves');

    if (plotType === 'metrics') {
        img.src = '/static/img/metrics_comparison.png';
        caption.textContent = 'Figure 1: Side-by-side empirical performance comparison across Accuracy, Precision, Recall, F1, Specificity, and Sensitivity.';
        btnMetrics.classList.add('active');
        btnCurves.classList.remove('active');
    } else {
        img.src = '/static/img/training_curves.png';
        caption.textContent = 'Figure 2: Training loss convergence dynamics across AdamW optimization steps on Apple Silicon MPS.';
        btnCurves.classList.add('active');
        btnMetrics.classList.remove('active');
    }
}

// ==========================================================================
// TAB 4: VIVA QUESTIONS SEARCH FILTER
// ==========================================================================

function searchVivaQuestions() {
    const query = (document.getElementById('viva-search-input')?.value || '').toLowerCase().trim();
    const items = document.querySelectorAll('.viva-item');

    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        const keywords = item.getAttribute('data-keywords') || '';
        if (!query || text.includes(query) || keywords.includes(query)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}

// ==========================================================================
// TAB 5: TOKENIZER SIMULATOR
// ==========================================================================

function simulateTokens() {
    const input = document.getElementById('token-test-input');
    const container = document.getElementById('token-stream-box');
    const countChip = document.getElementById('token-count-chip');
    if (!input || !container) return;

    const rawText = input.value.trim();
    if (!rawText) {
        container.innerHTML = '<span class="text-muted small">Type above to view sub-word tokens...</span>';
        if (countChip) countChip.textContent = '0 tokens';
        return;
    }

    // Heuristic WordPiece decomposition simulator
    const words = rawText.split(/\s+/);
    const tokens = ['[CLS]'];

    words.forEach(word => {
        const clean = word.replace(/[^a-zA-Z0-9]/g, '');
        if (!clean) return;

        if (clean.length > 7) {
            const half = Math.floor(clean.length / 2);
            tokens.push(clean.substring(0, half).toLowerCase());
            tokens.push('##' + clean.substring(half).toLowerCase());
        } else {
            tokens.push(clean.toLowerCase());
        }
    });
    tokens.push('[SEP]');

    container.innerHTML = tokens.map(tok => {
        if (tok === '[CLS]' || tok === '[SEP]') {
            return `<span class="token-chip token-cls">${tok}</span>`;
        }
        return `<span class="token-chip">${tok}</span>`;
    }).join('');

    if (countChip) countChip.textContent = `${tokens.length} tokens`;
}

// Helper: Escape HTML strings for table rendering
function escapeHtml(string) {
    if (!string) return '';
    return String(string)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
