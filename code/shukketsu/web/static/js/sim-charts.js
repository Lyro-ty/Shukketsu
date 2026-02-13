/**
 * Chart.js rendering functions for DPS simulation results.
 *
 * Used by HTMX partials returned from /sim/run to visualise ability
 * breakdowns, stat weights, and DPS distributions.
 */

/** WoW-themed colour palette for chart segments. */
var SIM_CHART_COLORS = [
    '#f5c518', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
    '#e67e22', '#1abc9c', '#34495e', '#f39c12', '#d35400',
];

/** Render a pie chart showing ability DPS breakdown. */
function renderAbilityPie(canvasId, breakdownData) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;

    var labels = Object.keys(breakdownData).map(function (k) {
        return k.replace(/_/g, ' ');
    });
    var values = Object.values(breakdownData);

    new Chart(canvas, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: SIM_CHART_COLORS.slice(0, labels.length),
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#e8dcc4', font: { size: 11 } },
                },
            },
        },
    });
}

/** Render a horizontal bar chart for stat weights. */
function renderStatWeightBar(canvasId, weightData) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;

    var labels = Object.keys(weightData).map(function (k) {
        return k.replace(/_/g, ' ');
    });
    var values = Object.values(weightData);

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: '#f5c518',
                borderWidth: 0,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#e8dcc4' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    ticks: { color: '#e8dcc4' },
                    grid: { display: false },
                },
            },
        },
    });
}

/** Render a histogram for DPS distribution. */
function renderDpsHistogram(canvasId, distributionData) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: distributionData.map(function (_, i) { return i; }),
            datasets: [{
                data: distributionData,
                backgroundColor: '#f5c518',
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#e8dcc4' },
                    grid: { display: false },
                    title: { display: true, text: 'DPS', color: '#a89b8c' },
                },
                y: {
                    ticks: { color: '#e8dcc4' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'Frequency', color: '#a89b8c' },
                },
            },
        },
    });
}
