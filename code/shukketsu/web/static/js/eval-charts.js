/**
 * Chart.js rendering for eval metric history.
 *
 * Fetches recent runs from /evals/history and renders a line chart
 * with 4 metric lines and threshold markers.
 */

var EVAL_COLORS = {
    faithfulness: '#2ecc71',
    answer_relevancy: '#3498db',
    trajectory_precision: '#f5c518',
    domain_accuracy: '#9b59b6',
};

var THRESHOLDS = {
    faithfulness: 0.8,
    answer_relevancy: 0.7,
    trajectory_precision: 0.7,
    domain_accuracy: 0.7,
};

function renderEvalHistory(canvasId, runs) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !runs.length) return;

    var labels = runs.map(function(r) {
        return r.run_name || r.created_at || '';
    });

    var datasets = Object.keys(EVAL_COLORS).map(function(metric) {
        return {
            label: metric.replace(/_/g, ' '),
            data: runs.map(function(r) {
                var meta = r.metadata || {};
                return meta['avg_' + metric] || null;
            }),
            borderColor: EVAL_COLORS[metric],
            backgroundColor: EVAL_COLORS[metric] + '20',
            tension: 0.3,
            pointRadius: 3,
        };
    });

    // Add threshold lines
    Object.keys(THRESHOLDS).forEach(function(metric) {
        datasets.push({
            label: metric.replace(/_/g, ' ') + ' threshold',
            data: new Array(labels.length).fill(THRESHOLDS[metric]),
            borderColor: EVAL_COLORS[metric] + '60',
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 1,
            fill: false,
        });
    });

    new Chart(canvas, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            scales: {
                y: {
                    min: 0, max: 1,
                    ticks: { color: '#a89b8c' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                x: {
                    ticks: { color: '#a89b8c', maxRotation: 45 },
                    grid: { display: false },
                },
            },
            plugins: {
                legend: {
                    labels: { color: '#e8dcc4', font: { size: 11 }, filter: function(item) {
                        return !item.text.includes('threshold');
                    }},
                },
            },
        },
    });
}

// Auto-load chart on page load
document.addEventListener('DOMContentLoaded', function() {
    fetch('/evals/history')
        .then(function(r) { return r.json(); })
        .then(function(data) { renderEvalHistory('eval-history-chart', data.runs || []); })
        .catch(function(e) { console.warn('Failed to load eval history:', e); });
});
