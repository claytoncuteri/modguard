/**
 * ModGuard Dashboard Application
 *
 * Manages WebSocket connection, chart rendering, live feed updates,
 * and detail modal display for the content moderation dashboard.
 */

// State
let ws = null;
let decisionChart = null;
let confidenceChart = null;
let feedItems = [];
const MAX_FEED_ITEMS = 100;

// Stats tracking
let stats = {
    totalProcessed: 0,
    approved: 0,
    flagged: 0,
    rejected: 0,
    totalConfidence: 0,
};

// Confidence histogram buckets
let confidenceBuckets = new Array(10).fill(0);

/**
 * Initialize the dashboard on page load.
 */
document.addEventListener('DOMContentLoaded', function () {
    initCharts();
    connectWebSocket();
    loadStats();
    loadHistory();
});

/**
 * Initialize Chart.js pie and bar charts.
 */
function initCharts() {
    const decisionCtx = document.getElementById('decisionChart').getContext('2d');
    decisionChart = new Chart(decisionCtx, {
        type: 'doughnut',
        data: {
            labels: ['Approved', 'Flagged', 'Rejected'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#22C55E', '#F59E0B', '#EF4444'],
                borderColor: '#1E293B',
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94A3B8', font: { size: 12 } },
                },
            },
        },
    });

    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'bar',
        data: {
            labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                     '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
            datasets: [{
                label: 'Count',
                data: new Array(10).fill(0),
                backgroundColor: '#3B82F6',
                borderColor: '#2563EB',
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: '#64748B', font: { size: 10 } },
                    grid: { color: '#334155' },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: '#64748B', stepSize: 1 },
                    grid: { color: '#334155' },
                },
            },
        },
    });
}

/**
 * Establish a WebSocket connection for real-time updates.
 */
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + window.location.host + '/ws';

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
        updateConnectionStatus(true);
    };

    ws.onmessage = function (event) {
        try {
            const result = JSON.parse(event.data);
            handleNewResult(result);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    ws.onclose = function () {
        updateConnectionStatus(false);
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = function () {
        updateConnectionStatus(false);
    };
}

/**
 * Update the connection status indicator.
 * @param {boolean} connected - Whether the WebSocket is connected.
 */
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connectionStatus');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('span:last-child');

    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Connected';
    } else {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Disconnected';
    }
}

/**
 * Load current stats from the API.
 */
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        stats.totalProcessed = data.total_processed;
        stats.approved = data.approved;
        stats.flagged = data.flagged;
        stats.rejected = data.rejected;
        stats.totalConfidence = data.avg_confidence * data.total_processed;
        updateStatsDisplay();
        updateDecisionChart();
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

/**
 * Load recent history from the API.
 */
async function loadHistory() {
    try {
        const response = await fetch('/history?page=1&page_size=50');
        const data = await response.json();
        if (data.items && data.items.length > 0) {
            feedItems = data.items;
            renderFeed();

            // Rebuild confidence buckets from history
            data.items.forEach(function (item) {
                const bucket = Math.min(9, Math.floor(item.confidence * 10));
                confidenceBuckets[bucket]++;
            });
            updateConfidenceChart();
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

/**
 * Handle a new moderation result from WebSocket or API.
 * @param {Object} result - The moderation result object.
 */
function handleNewResult(result) {
    // Update stats
    stats.totalProcessed++;
    stats.totalConfidence += result.confidence;

    if (result.decision === 'APPROVE') {
        stats.approved++;
    } else if (result.decision === 'FLAG_FOR_REVIEW') {
        stats.flagged++;
    } else if (result.decision === 'REJECT') {
        stats.rejected++;
    }

    // Update confidence histogram
    const bucket = Math.min(9, Math.floor(result.confidence * 10));
    confidenceBuckets[bucket]++;

    // Add to feed
    feedItems.unshift(result);
    if (feedItems.length > MAX_FEED_ITEMS) {
        feedItems.pop();
    }

    // Update UI
    updateStatsDisplay();
    updateDecisionChart();
    updateConfidenceChart();
    renderFeed();
}

/**
 * Update the stats cards with current values.
 */
function updateStatsDisplay() {
    document.getElementById('totalProcessed').textContent =
        stats.totalProcessed.toLocaleString();

    const approvalRate = stats.totalProcessed > 0
        ? ((stats.approved / stats.totalProcessed) * 100).toFixed(1)
        : '0.0';
    document.getElementById('approvalRate').textContent = approvalRate + '%';

    const avgConfidence = stats.totalProcessed > 0
        ? ((stats.totalConfidence / stats.totalProcessed) * 100).toFixed(1)
        : '0.0';
    document.getElementById('avgConfidence').textContent = avgConfidence + '%';

    const flaggedRate = stats.totalProcessed > 0
        ? ((stats.flagged / stats.totalProcessed) * 100).toFixed(1)
        : '0.0';
    document.getElementById('flaggedRate').textContent = flaggedRate + '%';
}

/**
 * Update the decision distribution chart.
 */
function updateDecisionChart() {
    if (decisionChart) {
        decisionChart.data.datasets[0].data = [
            stats.approved,
            stats.flagged,
            stats.rejected,
        ];
        decisionChart.update();
    }
}

/**
 * Update the confidence histogram chart.
 */
function updateConfidenceChart() {
    if (confidenceChart) {
        confidenceChart.data.datasets[0].data = [...confidenceBuckets];
        confidenceChart.update();
    }
}

/**
 * Render the live feed list.
 */
function renderFeed() {
    const feedList = document.getElementById('feedList');

    if (feedItems.length === 0) {
        feedList.innerHTML =
            '<div class="feed-empty">No moderation results yet. ' +
            'Submit text above or send via API.</div>';
        return;
    }

    let html = '';
    feedItems.forEach(function (item, index) {
        const badgeClass = getBadgeClass(item.decision);
        const displayText = item.text.length > 120
            ? item.text.substring(0, 120) + '...'
            : item.text;
        const confidence = (item.confidence * 100).toFixed(0) + '%';
        const timeStr = item.processing_time_ms
            ? item.processing_time_ms.toFixed(0) + 'ms'
            : '';

        html += '<div class="feed-item" onclick="showDetail(' + index + ')">';
        html += '<span class="decision-badge ' + badgeClass + '">';
        html += formatDecision(item.decision);
        html += '</span>';
        html += '<span class="feed-text">' + escapeHtml(displayText) + '</span>';
        html += '<span class="feed-meta">' + confidence + ' | ' + timeStr + '</span>';
        html += '</div>';
    });

    feedList.innerHTML = html;
}

/**
 * Get the CSS badge class for a decision.
 * @param {string} decision - The decision string.
 * @returns {string} CSS class name.
 */
function getBadgeClass(decision) {
    switch (decision) {
        case 'APPROVE': return 'badge-approve';
        case 'FLAG_FOR_REVIEW': return 'badge-flag';
        case 'REJECT': return 'badge-reject';
        default: return 'badge-flag';
    }
}

/**
 * Format a decision string for display.
 * @param {string} decision - The raw decision string.
 * @returns {string} Formatted decision string.
 */
function formatDecision(decision) {
    switch (decision) {
        case 'APPROVE': return 'Approved';
        case 'FLAG_FOR_REVIEW': return 'Flagged';
        case 'REJECT': return 'Rejected';
        default: return decision;
    }
}

/**
 * Submit text for moderation via the API.
 */
async function submitModeration() {
    const input = document.getElementById('moderationInput');
    const btn = document.getElementById('moderateBtn');
    const text = input.value.trim();

    if (!text) return;

    btn.disabled = true;
    btn.textContent = 'Processing...';

    try {
        const response = await fetch('/moderate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text }),
        });
        const result = await response.json();
        // The WebSocket broadcast will handle the UI update,
        // but also handle it directly in case WS is disconnected
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            handleNewResult(result);
        }
        input.value = '';
    } catch (e) {
        console.error('Moderation request failed:', e);
        alert('Failed to process moderation request. Check server connection.');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Moderate';
    }
}

/**
 * Show the detail modal for a feed item.
 * @param {number} index - Index in the feedItems array.
 */
function showDetail(index) {
    const item = feedItems[index];
    if (!item) return;

    const modalBody = document.getElementById('modalBody');
    let html = '';

    // Decision summary
    const badgeClass = getBadgeClass(item.decision);
    html += '<div class="detail-section">';
    html += '<h4>Decision</h4>';
    html += '<div class="detail-value">';
    html += '<span class="decision-badge ' + badgeClass + '">';
    html += formatDecision(item.decision) + '</span>';
    html += ' with ' + (item.confidence * 100).toFixed(1) + '% confidence';
    html += ' in ' + (item.processing_time_ms || 0).toFixed(1) + 'ms';
    html += '</div></div>';

    // Original text
    html += '<div class="detail-section">';
    html += '<h4>Original Text</h4>';
    html += '<div class="detail-value">' + escapeHtml(item.text) + '</div>';
    html += '</div>';

    // Explanation
    html += '<div class="detail-section">';
    html += '<h4>Explanation</h4>';
    html += '<div class="detail-value">' + escapeHtml(item.explanation) + '</div>';
    html += '</div>';

    // Layer-by-layer breakdown
    if (item.layer_results) {
        html += '<div class="detail-section">';
        html += '<h4>Layer Breakdown</h4>';

        Object.keys(item.layer_results).forEach(function (layerName) {
            const layerData = item.layer_results[layerName];
            html += '<div class="layer-item">';
            html += '<h5>' + escapeHtml(layerName) + '</h5>';
            html += '<pre>' + escapeHtml(JSON.stringify(layerData, null, 2)) + '</pre>';
            html += '</div>';
        });

        html += '</div>';
    }

    modalBody.innerHTML = html;
    document.getElementById('modalOverlay').classList.add('active');
}

/**
 * Close the detail modal.
 */
function closeModal() {
    document.getElementById('modalOverlay').classList.remove('active');
}

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} text - The text to escape.
 * @returns {string} Escaped HTML string.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Allow Enter key to submit moderation
document.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        const input = document.getElementById('moderationInput');
        if (document.activeElement === input) {
            e.preventDefault();
            submitModeration();
        }
    }

    // Close modal with Escape key
    if (e.key === 'Escape') {
        closeModal();
    }
});
