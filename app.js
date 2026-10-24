/* ================================================
   CampusGrid AI — Energy Intelligence System
   Main application JavaScript
   Loads real GBR model results from energy_forecast_results.json
   ================================================ */

'use strict';

// ── Chart.js Global Defaults ──────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(129,140,248,0.1)';
Chart.defaults.font.family = 'Inter, -apple-system, sans-serif';
Chart.defaults.font.size = 11;

// ── Color Palette ──────────────────────────────────
const C = {
  indigo:  '#818cf8', indigoA: (a) => `rgba(129,140,248,${a})`,
  emerald: '#34d399', emeraldA: (a) => `rgba(52,211,153,${a})`,
  amber:   '#f59e0b', amberA:  (a) => `rgba(245,158,11,${a})`,
  rose:    '#ef4444', roseA:   (a) => `rgba(239,68,68,${a})`,
  sky:     '#38bdf8', skyA:    (a) => `rgba(56,189,248,${a})`,
  purple:  '#c084fc', purpleA: (a) => `rgba(192,132,252,${a})`,
};

// ── State ───────────────────────────────────────────
let DATA = null;
let charts = {};
let testPeriodFullData = null;

// ── Load Real Data ─────────────────────────────────
async function loadData() {
  try {
    const res = await fetch('./energy_forecast_results.json');
    DATA = await res.json();
    console.log('✅ Real GBR data loaded:', DATA.gbr_mae, 'MAE');
    init();
  } catch (e) {
    console.error('Failed to load JSON, using fallback', e);
    useFallback();
  }
}

function useFallback() {
  DATA = {
    gbr_mae: 0.8981, gbr_rmse: 2.0552,
    baseline_mae: 1.0125, baseline_rmse: 3.324,
    improvement_pct: 11.3, high_threshold: 16.473, mean_consumption: 5.326,
    train_size: 11472, test_size: 744,
    feature_importances: { lag_1:0.8967, lag_24:0.0306, hour:0.014, lag_48:0.0094, lag_168:0.0083, lag_2:0.0083, lag_3:0.007, roll_24h_mean:0.0057, roll_24h_std:0.005, temp_proxy:0.0049, roll_7d_mean:0.0045, is_occupied:0.0019, day_of_week:0.0017, month:0.0014, is_weekend:0.0006 },
    hourly_avg: [6.663,6.091,5.619,5.265,5.012,5.005,6.259,5.921,6.459,7.374,5.907,4.992,4.935,4.872,4.889,4.614,4.518,4.603,4.526,5.223,5.965,6.539,6.867,6.986],
    weekday_avg: [5.864,5.504,5.698,5.516,5.348,5.605,5.866],
    monthly_avg: [2.788,3.248,4.475,3.4,7.698,10.469,9.241,9.151,6.274,6.773,6.289,3.237],
    forecast_timestamps: Array.from({length:24},(_,i)=>`2018-05-31 ${String(i).padStart(2,'0')}:00:00`),
    forecast_actual: [32.275,29.555,25.658,21.927,20.284,20.174,17.696,12.834,14.673,14.477,12.564,12.586,14.651,13.601,17.09,15.884,12.887,12.077,12.624,18.074,21.026,25.38,35.987,38.856],
    forecast_predicted: [23.781,31.694,31.647,25.742,24.725,16.547,17.624,16.251,15.031,17.239,12.584,12.274,13.274,11.039,12.613,13.928,13.028,10.169,11.593,12.659,17.376,17.473,22.555,22.39],
    forecast_baseline: [27.625,26.572,25.756,24.073,21.57,17.177,14.901,13.193,13.961,13.42,10.349,9.914,13.068,16.228,15.907,13.59,11.165,9.398,10.448,11.832,15.829,21.296,29.752,34.879],
    test_timestamps: [], test_actual: [], test_predicted: [], test_baseline: [],
  };
  init();
}

// ── Navigation ─────────────────────────────────────
document.querySelectorAll('.nav-pill').forEach(btn => {
  btn.addEventListener('click', () => {
    const sec = btn.dataset.section;
    document.querySelectorAll('.nav-pill').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`section-${sec}`).classList.add('active');
    setTimeout(() => initSectionCharts(sec), 50);
  });
});

// ── Number Counter Animation ───────────────────────
function animateNumber(el, from, to, decimals = 2, suffix = '', duration = 900) {
  const start = performance.now();
  function update(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (to - from) * ease).toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── Hero Metrics ───────────────────────────────────
function updateMetrics(alertCount) {
  animateNumber(document.getElementById('val-mae'), 0, DATA.gbr_mae, 4, ' kWh');
  animateNumber(document.getElementById('val-rmse'), 0, DATA.gbr_rmse, 4, ' kWh');
  animateNumber(document.getElementById('val-improvement'), 0, DATA.improvement_pct, 1, '%');
  setTimeout(() => {
    const el = document.getElementById('val-alerts');
    el.textContent = alertCount;
  }, 600);
  document.getElementById('sub-mae').textContent =
    `Baseline: ${DATA.baseline_mae.toFixed(4)} kWh — ${DATA.improvement_pct}% better`;
}

// ── Build Forecast Chart (real 24-hour data) ───────
function buildForecastChart() {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  if (charts.forecast) charts.forecast.destroy();

  // Real 24-hour forecast from GBR model (May 31, 2018)
  const hours = DATA.forecast_timestamps.map(ts => ts.split(' ')[1].substring(0,5));
  const actual    = DATA.forecast_actual;
  const predicted = DATA.forecast_predicted;
  const baseline  = DATA.forecast_baseline;

  const gradIndigo = ctx.createLinearGradient(0, 0, 0, 360);
  gradIndigo.addColorStop(0, C.indigoA(0.28));
  gradIndigo.addColorStop(1, C.indigoA(0.02));

  const gradEmerald = ctx.createLinearGradient(0, 0, 0, 360);
  gradEmerald.addColorStop(0, C.emeraldA(0.18));
  gradEmerald.addColorStop(1, C.emeraldA(0.01));

  charts.forecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels: hours,
      datasets: [
        {
          label: 'AI Forecast (GBR)',
          data: predicted,
          borderColor: C.indigo,
          backgroundColor: gradIndigo,
          borderWidth: 2.5,
          pointRadius: 5,
          pointHoverRadius: 8,
          pointBackgroundColor: C.indigo,
          pointBorderColor: '#060914',
          pointBorderWidth: 2,
          fill: true, tension: 0.4, order: 1,
        },
        {
          label: 'Actual Consumption',
          data: actual,
          borderColor: C.emerald,
          backgroundColor: gradEmerald,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointBackgroundColor: C.emerald,
          pointBorderColor: '#060914',
          pointBorderWidth: 1.5,
          fill: true, tension: 0.4, order: 2,
        },
        {
          label: 'Naive Baseline (Prev Day)',
          data: baseline,
          borderColor: C.amber,
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          pointHoverRadius: 5,
          fill: false, tension: 0.4, order: 3,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2.8,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(6,9,20,0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: C.indigoA(0.3),
          borderWidth: 1, padding: 12, cornerRadius: 10,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} kWh`,
            afterBody: (items) => {
              const pred = items.find(i => i.datasetIndex === 0)?.parsed.y;
              if (pred && pred > DATA.high_threshold) return ['⚡ HIGH-USE ALERT'];
              return [];
            }
          }
        },
        annotation: {
          annotations: {
            threshold: {
              type: 'line',
              yMin: DATA.high_threshold,
              yMax: DATA.high_threshold,
              borderColor: C.roseA(0.75),
              borderWidth: 1.5,
              borderDash: [8, 4],
              label: {
                display: true,
                content: `⚡ High-Use Threshold (${DATA.high_threshold.toFixed(1)} kWh)`,
                position: 'end',
                backgroundColor: C.roseA(0.15),
                color: C.rose,
                font: { size: 10, weight: '600' },
                padding: { x: 8, y: 4 },
                borderRadius: 4,
              }
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: C.indigoA(0.06) },
          ticks: { font: { size: 10 } },
        },
        y: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'Consumption (kWh)', color: '#64748b', font: { size: 10 } },
          ticks: { callback: v => v.toFixed(1) + ' kWh', font: { size: 10 } }
        }
      }
    }
  });

  // Count alerts
  const alertCount = predicted.filter(v => v > DATA.high_threshold).length;
  buildAlerts(hours, predicted, alertCount);
  updateMetrics(alertCount);
}

// ── Alert Generation (real data) ──────────────────
function buildAlerts(hours, predicted, alertCount) {
  const container = document.getElementById('alerts-list');
  container.innerHTML = '';

  const flagged = predicted.map((v, i) => ({ hour: hours[i], value: v, idx: i }))
    .filter(x => x.value > DATA.high_threshold)
    .sort((a, b) => b.value - a.value);

  if (flagged.length === 0) {
    container.innerHTML = `
      <div class="no-alerts">
        <span style="font-size:2rem">✅</span>
        <span style="color:#34d399;font-weight:600">All clear — No high-use periods predicted</span>
        <span style="color:#64748b;font-size:0.75rem">All 24 forecast hours below ${DATA.high_threshold.toFixed(1)} kWh</span>
      </div>`;
    return;
  }

  const recommendations = {
    '00': 'Audit overnight HVAC setback schedules. Server room cooling & backup systems may be over-provisioned.',
    '01': 'Check overnight AC setpoints — allow 2–3°C relaxation for unoccupied zones.',
    '02': 'Review HVAC pre-conditioning schedules. Consider delaying warm-up by 30 min.',
    '03': 'Unexpected pre-dawn surge. Inspect common-area lighting timers and laundry room equipment.',
    '04': 'Pre-dawn HVAC ramp-up detected. Verify optimum-start mode is active on boiler/chiller.',
    '05': 'Early HVAC warm-up. Consider demand-controlled ventilation and staggered equipment starts.',
    '06': 'Morning start-up phase. Confirm daylight harvesting is enabled in east-facing zones.',
    '07': 'Morning peak build-up. Verify lighting auto-dimming in corridors and unoccupied rooms.',
    '08': 'Business-hours peak. Activate demand-response; defer non-critical computing workloads.',
    '09': 'Mid-morning high demand. Check lab equipment scheduling, AHU duty cycles, elevator loads.',
    '10': 'Peak academic-hours load. Consider load-shedding auxiliary lighting in low-occupancy zones.',
    '11': 'Pre-lunch peak. Brief equipment shutdown windows before canteen demand surge.',
    '12': 'Midday peak — likely canteen + HVAC cooling. Raise cooling setpoints 1°C during lunch.',
    '13': 'Post-lunch cooling demand. Monitor chiller staging; consider free-cooling if outdoor temp allows.',
    '14': 'Afternoon academic peak. Verify AHU scheduling in lecture halls with low occupancy.',
    '15': 'High afternoon demand. Consider solar pre-shading strategies and LED dimming.',
    '16': 'Late-afternoon peak. Batch non-urgent computing to off-peak hours (after 22:00).',
    '17': 'Early-evening transition. Implement auto-off schedules for classroom equipment.',
    '18': 'Evening demand — check if building is vacated and HVAC is in setback mode.',
    '19': 'Evening load — verify auto-off systems active for common areas and office zones.',
    '20': 'Unexpected evening demand. Audit shared kitchens, printing stations & student areas.',
    '21': 'Late-evening — confirm auto wind-down is active. Check security lighting levels.',
    '22': 'Post-occupancy load — check server room cooling cycles and hallway lighting timers.',
    '23': 'Midnight spike — audit backup systems, fire panels, overnight equipment left powered on.',
  };

  flagged.slice(0, 8).forEach(({ hour, value }) => {
    const hourKey = hour.split(':')[0];
    const rec = recommendations[hourKey] || 'Inspect active loads and verify automated shutdown sequences are functioning.';
    const isCritical = value > DATA.high_threshold * 1.4;
    const overPct = (((value - DATA.high_threshold) / DATA.high_threshold) * 100).toFixed(0);
    const el = document.createElement('div');
    el.className = `alert-item${isCritical ? '' : ' medium'}`;
    el.innerHTML = `
      <div class="alert-header">
        <span class="alert-time">🕐 ${hour}</span>
        <span class="alert-badge${isCritical ? '' : ' medium'}">${isCritical ? '🔴 CRITICAL' : '🟡 HIGH'}</span>
      </div>
      <div class="alert-value${isCritical ? '' : ' medium'}">
        ${value.toFixed(2)} kWh
        <span style="font-size:0.7rem;font-weight:500;opacity:0.7"> (+${overPct}% over threshold)</span>
      </div>
      <div class="alert-rec"><strong>Recommended check:</strong> ${rec}</div>
    `;
    container.appendChild(el);
  });

  if (flagged.length > 8) {
    const more = document.createElement('div');
    more.style.cssText = 'text-align:center;color:#64748b;font-size:0.75rem;padding:0.5rem';
    more.textContent = `+${flagged.length - 8} more high-use hours in forecast period`;
    container.appendChild(more);
  }
}

// ── Error Comparison Chart ─────────────────────────
function buildErrorChart() {
  const ctx = document.getElementById('errorChart').getContext('2d');
  if (charts.error) charts.error.destroy();

  charts.error = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['MAE', 'RMSE'],
      datasets: [
        {
          label: 'GBR Model',
          data: [DATA.gbr_mae, DATA.gbr_rmse],
          backgroundColor: [C.indigoA(0.75), C.indigoA(0.75)],
          borderColor: [C.indigo, C.indigo],
          borderWidth: 1.5, borderRadius: 6, borderSkipped: false,
        },
        {
          label: 'Naive Baseline',
          data: [DATA.baseline_mae, DATA.baseline_rmse],
          backgroundColor: [C.amberA(0.55), C.amberA(0.55)],
          borderColor: [C.amber, C.amber],
          borderWidth: 1.5, borderRadius: 6, borderSkipped: false,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(6,9,20,0.95)',
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)} kWh` }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11, weight: '600' } } },
        y: {
          grid: { color: C.indigoA(0.06) },
          ticks: { callback: v => v.toFixed(2), font: { size: 10 } },
          title: { display: true, text: 'Error (kWh)', color: '#64748b', font: { size: 10 } }
        }
      }
    }
  });

  document.getElementById('error-table').innerHTML = `
    <div class="error-row">
      <span class="error-row-label">Metric</span>
      <div class="error-row-vals">
        <span class="error-val" style="color:#818cf8;font-weight:700">GBR Model</span>
        <span class="error-val" style="color:#f59e0b;font-weight:700">Baseline</span>
      </div>
    </div>
    <div class="error-row">
      <span class="error-row-label">MAE (kWh)</span>
      <div class="error-row-vals">
        <span class="error-val model">${DATA.gbr_mae.toFixed(4)}</span>
        <span class="error-val baseline">${DATA.baseline_mae.toFixed(4)}</span>
      </div>
    </div>
    <div class="error-row">
      <span class="error-row-label">RMSE (kWh)</span>
      <div class="error-row-vals">
        <span class="error-val model">${DATA.gbr_rmse.toFixed(4)}</span>
        <span class="error-val baseline">${DATA.baseline_rmse.toFixed(4)}</span>
      </div>
    </div>
    <div class="error-row">
      <span class="error-row-label">MAE Improvement</span>
      <div class="error-row-vals">
        <span class="error-val" style="color:#34d399">+${DATA.improvement_pct.toFixed(1)}%</span>
        <span class="error-val" style="color:#64748b">—</span>
      </div>
    </div>
    <div class="error-row">
      <span class="error-row-label">RMSE Improvement</span>
      <div class="error-row-vals">
        <span class="error-val" style="color:#34d399">+${((1-DATA.gbr_rmse/DATA.baseline_rmse)*100).toFixed(1)}%</span>
        <span class="error-val" style="color:#64748b">—</span>
      </div>
    </div>
    <div class="error-row">
      <span class="error-row-label">Training Hours</span>
      <div class="error-row-vals">
        <span class="error-val model">${(DATA.train_size||11472).toLocaleString()}</span>
        <span class="error-val" style="color:#64748b">—</span>
      </div>
    </div>
    <div class="error-row">
      <span class="error-row-label">Test Hours</span>
      <div class="error-row-vals">
        <span class="error-val model">${(DATA.test_size||744).toLocaleString()}</span>
        <span class="error-val" style="color:#64748b">—</span>
      </div>
    </div>
  `;
}

// ── Full Test Period Chart (real 744-hr data) ──────
function buildTestPeriodChart() {
  const ctx = document.getElementById('testPeriodChart')?.getContext('2d');
  if (!ctx) return;
  if (charts.testPeriod) charts.testPeriod.destroy();

  // Use real test timestamps → extract day labels for every 24th point
  const ts = DATA.test_timestamps;
  const labels = ts.map((t, i) => {
    const h = parseInt(t.split(' ')[1].split(':')[0]);
    return h === 0 ? t.split(' ')[0].slice(5) : (h === 12 ? '↕' : '');
  });

  testPeriodFullData = {
    labels,
    actual: DATA.test_actual,
    predicted: DATA.test_predicted,
    baseline: DATA.test_baseline,
    timestamps: ts,
  };

  charts.testPeriod = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Actual',
          data: DATA.test_actual,
          borderColor: C.emerald,
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        },
        {
          label: 'AI Forecast (GBR)',
          data: DATA.test_predicted,
          borderColor: C.indigo,
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        },
        {
          label: 'Naive Baseline',
          data: DATA.test_baseline,
          borderColor: C.amber,
          backgroundColor: 'transparent',
          borderWidth: 1, borderDash: [4, 3],
          pointRadius: 0, tension: 0.2,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true, aspectRatio: 3,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true, position: 'top', align: 'end',
          labels: { boxWidth: 14, boxHeight: 2, padding: 16, usePointStyle: true, pointStyleWidth: 14 }
        },
        tooltip: {
          backgroundColor: 'rgba(6,9,20,0.95)',
          callbacks: {
            title: (items) => {
              const idx = items[0]?.dataIndex;
              return ts[idx] || '';
            },
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} kWh`
          }
        },
        annotation: {
          annotations: {
            threshold: {
              type: 'line',
              yMin: DATA.high_threshold, yMax: DATA.high_threshold,
              borderColor: C.roseA(0.45), borderWidth: 1, borderDash: [6, 4],
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: C.indigoA(0.04) },
          ticks: { maxTicksLimit: 16, maxRotation: 0, font: { size: 9 } }
        },
        y: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'kWh', color: '#64748b', font: { size: 10 } },
          ticks: { callback: v => v.toFixed(1), font: { size: 10 } }
        }
      }
    }
  });
}

window.filterTestChart = function(filter) {
  document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + filter)?.classList.add('active');
  if (!charts.testPeriod || !testPeriodFullData) return;

  const td = testPeriodFullData;
  const n = td.actual.length;
  let start = 0, end = n;

  if (filter === 'week1') { start = 0; end = Math.min(168, n); }
  else if (filter === 'week2') { start = 168; end = Math.min(336, n); }
  else if (filter === 'week3') { start = 336; end = Math.min(504, n); }

  charts.testPeriod.data.labels = td.labels.slice(start, end);
  charts.testPeriod.data.datasets[0].data = td.actual.slice(start, end);
  charts.testPeriod.data.datasets[1].data = td.predicted.slice(start, end);
  charts.testPeriod.data.datasets[2].data = td.baseline.slice(start, end);
  charts.testPeriod.update('active');
};

// ── Hourly Load Profile ───────────────────────────
function buildHourlyChart() {
  const ctx = document.getElementById('hourlyChart')?.getContext('2d');
  if (!ctx || charts.hourly) return;

  const hours = Array.from({length:24}, (_,i) => `${String(i).padStart(2,'0')}:00`);
  const vals = DATA.hourly_avg;
  const maxV = Math.max(...vals);

  charts.hourly = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hours,
      datasets: [{
        label: 'Avg. kWh',
        data: vals,
        backgroundColor: vals.map(v => v >= maxV * 0.9 ? C.roseA(0.75) : v >= maxV * 0.75 ? C.indigoA(0.8) : C.skyA(0.55)),
        borderColor: vals.map(v => v >= maxV * 0.9 ? C.rose : v >= maxV * 0.75 ? C.indigo : C.sky),
        borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true, aspectRatio: 1.8,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(6,9,20,0.95)', callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(3)} kWh avg` } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, font: { size: 9 }, maxTicksLimit: 8 } },
        y: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'Avg kWh', color: '#64748b', font: { size: 10 } },
          ticks: { callback: v => v.toFixed(1), font: { size: 10 } }
        }
      }
    }
  });
}

// ── Weekly Pattern (Radar) ────────────────────────
function buildWeeklyChart() {
  const ctx = document.getElementById('weeklyChart')?.getContext('2d');
  if (!ctx || charts.weekly) return;

  charts.weekly = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
      datasets: [{
        label: 'Avg. kWh/hr',
        data: DATA.weekday_avg,
        borderColor: C.purple, backgroundColor: C.purpleA(0.15),
        pointBackgroundColor: C.purple, pointBorderColor: '#060914',
        pointBorderWidth: 2, pointRadius: 5, borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true, aspectRatio: 1.8,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(6,9,20,0.95)', callbacks: { label: ctx => ` ${ctx.parsed.r.toFixed(3)} kWh avg/hr` } }
      },
      scales: {
        r: {
          grid: { color: C.purpleA(0.15) },
          pointLabels: { color: '#94a3b8', font: { size: 10 } },
          ticks: { backdropColor: 'transparent', color: '#64748b', font: { size: 8 } },
        }
      }
    }
  });
}

// ── Scatter: Predicted vs Actual (real 744-pt data) ─
function buildScatterChart() {
  const ctx = document.getElementById('scatterChart')?.getContext('2d');
  if (!ctx || charts.scatter) return;

  const actual = DATA.test_actual;
  const predicted = DATA.test_predicted;
  const baseline = DATA.test_baseline;

  // Sample every 4th point for readability
  const modelPts = [], basePts = [];
  for (let i = 0; i < actual.length; i += 4) {
    modelPts.push({ x: actual[i], y: predicted[i] });
    basePts.push({ x: actual[i], y: baseline[i] });
  }

  const allVals = [...actual, ...predicted, ...baseline];
  const minV = Math.min(...allVals.filter(v => v > 0));
  const maxV = Math.max(...allVals);

  charts.scatter = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'GBR Model',
          data: modelPts,
          backgroundColor: C.indigoA(0.45),
          borderColor: C.indigoA(0.7),
          borderWidth: 0.5,
          pointRadius: 3.5, pointHoverRadius: 6,
        },
        {
          label: 'Naive Baseline',
          data: basePts,
          backgroundColor: C.amberA(0.3),
          borderColor: C.amberA(0.6),
          borderWidth: 0.5,
          pointRadius: 3, pointHoverRadius: 5,
        },
        {
          label: 'Perfect Prediction',
          data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
          borderColor: C.roseA(0.65),
          borderWidth: 1.5,
          borderDash: [5, 5],
          showLine: true, pointRadius: 0, type: 'line',
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true, aspectRatio: 2.2,
      plugins: {
        legend: {
          display: true, position: 'top', align: 'end',
          labels: { boxWidth: 10, boxHeight: 10, padding: 16, usePointStyle: true }
        },
        tooltip: {
          backgroundColor: 'rgba(6,9,20,0.95)',
          callbacks: {
            label: ctx => ctx.dataset.label !== 'Perfect Prediction'
              ? ` Actual: ${ctx.parsed.x.toFixed(2)} → Pred: ${ctx.parsed.y.toFixed(2)} kWh`
              : null
          }
        }
      },
      scales: {
        x: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'Actual Consumption (kWh)', color: '#64748b', font: { size: 11 } },
          ticks: { callback: v => v.toFixed(1), font: { size: 10 } }
        },
        y: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'Predicted (kWh)', color: '#64748b', font: { size: 11 } },
          ticks: { callback: v => v.toFixed(1), font: { size: 10 } }
        }
      }
    }
  });
}

// ── Feature Importance ────────────────────────────
function buildFeatureChart() {
  const ctx = document.getElementById('featureChart')?.getContext('2d');
  if (!ctx || charts.feature) return;

  const fi = DATA.feature_importances;
  const sorted = Object.entries(fi).sort((a,b) => b[1]-a[1]);
  const labelMap = {
    lag_1:'Lag 1h',lag_24:'Lag 24h',hour:'Hour',lag_48:'Lag 48h',lag_168:'Lag 168h',
    lag_2:'Lag 2h',lag_3:'Lag 3h',roll_24h_mean:'Roll 24h Mean',roll_24h_std:'Roll 24h Std',
    temp_proxy:'Temp Proxy',roll_7d_mean:'Roll 7d Mean',is_occupied:'Is Occupied',
    day_of_week:'Day of Week',month:'Month',is_weekend:'Is Weekend'
  };
  const labels = sorted.map(([k]) => labelMap[k] || k);
  const vals = sorted.map(([,v]) => parseFloat((v*100).toFixed(2)));
  const colors = vals.map((v, i) => {
    if (i === 0) return C.indigoA(0.9);
    if (i < 3) return C.indigoA(0.65);
    if (i < 7) return C.skyA(0.55);
    return C.purpleA(0.45);
  });

  charts.feature = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Feature Importance (%)',
        data: vals,
        backgroundColor: colors,
        borderWidth: 1, borderRadius: 4, borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: true, aspectRatio: 1.4,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(6,9,20,0.95)', callbacks: { label: ctx => ` Importance: ${ctx.parsed.x}%` } }
      },
      scales: {
        x: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'Importance (%)', color: '#64748b', font: { size: 10 } },
          ticks: { callback: v => v + '%', font: { size: 10 } }
        },
        y: { grid: { display: false }, ticks: { font: { size: 10, family: "'JetBrains Mono', monospace" } } }
      }
    }
  });
}

// ── Monthly Profile ───────────────────────────────
function buildMonthlyChart() {
  const ctx = document.getElementById('monthlyChart')?.getContext('2d');
  if (!ctx || charts.monthly) return;

  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const vals = Array.isArray(DATA.monthly_avg) ? DATA.monthly_avg : [];
  // monthly_avg from Python has 12 values (index 1-12 mapped to Jan-Dec)

  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, C.amberA(0.5));
  gradient.addColorStop(1, C.amberA(0.02));

  charts.monthly = new Chart(ctx, {
    type: 'line',
    data: {
      labels: months,
      datasets: [{
        label: 'Monthly Avg. kWh/hr',
        data: vals,
        borderColor: C.amber,
        backgroundColor: gradient,
        borderWidth: 2.5,
        pointRadius: 6, pointHoverRadius: 9,
        pointBackgroundColor: C.amber, pointBorderColor: '#060914', pointBorderWidth: 2,
        fill: true, tension: 0.4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true, aspectRatio: 2.5,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(6,9,20,0.95)', callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(2)} kWh avg/hr` } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: {
          grid: { color: C.indigoA(0.06) },
          title: { display: true, text: 'Avg kWh / hr', color: '#64748b', font: { size: 10 } },
          ticks: { callback: v => v.toFixed(1), font: { size: 10 } }, min: 0,
        }
      }
    }
  });
}

// ── Insights Panel ────────────────────────────────
function buildInsights() {
  const container = document.getElementById('insights-grid');
  if (!container || container.children.length > 0) return;

  const insights = [
    { emoji:'🌙', title:'Overnight Load Reduction', priority:'high', text:`The data shows sustained consumption during 00:00–06:00. Audit HVAC setback schedules and server room cooling. Nighttime setpoints can be relaxed 2–3°C in unoccupied residential zones — estimated 5–8% energy saving.` },
    { emoji:'⚡', title:'Peak Demand Shaving', priority:'high', text:`The model flags evening hours (20:00–23:00) with consistent high-use alerts. Implement demand-response: stagger non-critical equipment starts and defer batch computing jobs to off-peak windows (after 22:00).` },
    { emoji:'🌡️', title:'Temperature-Driven Efficiency', priority:'medium', text:`Monthly data shows strong seasonal variation (2.8→10.5 kWh/hr from winter to summer). Each 1°C increase in HVAC setpoint during summer peak hours reduces cooling load by approximately 3–5%.` },
    { emoji:'📅', title:'Weekend Scheduling Optimization', priority:'low', text:`Weekend consumption (~5.6 kWh/hr avg) runs slightly different from weekdays. Ensure automated scheduling deactivates HVAC zones in unoccupied departments and lecture halls every weekend.` },
    { emoji:'🔮', title:'Lag-1h Momentum Insight', priority:'medium', text:`The previous hour's consumption (lag_1) accounts for ${(DATA.feature_importances.lag_1*100).toFixed(1)}% of prediction power. Short-term momentum dominates — reinforcing the value of real-time monitoring and 15-minute sub-hourly alerts.` },
    { emoji:'📊', title:'Sub-Metering Recommendation', priority:'medium', text:`Current model aggregates building-level consumption. Zone-level sub-metering (HVAC vs lighting vs equipment) would enable targeted interventions and could reduce model error by an estimated 15–25%.` },
    { emoji:'🤖', title:'Model Retraining Cadence', priority:'low', text:`To prevent distributional drift, retrain the GBR model quarterly. Add academic calendar events (exam periods, holidays, special events) as binary features to capture occupancy-driven demand spikes.` },
    { emoji:'💡', title:'Daylight Harvesting Opportunity', priority:'medium', text:`Hours 07:00–09:00 show rapid consumption ramp-up. Daylight harvesting systems in east-facing classrooms and corridors can defer this morning peak and reduce demand by 8–12%.` },
    { emoji:'🏭', title:'Equipment Audit Trigger', priority:'high', text:`Hours predicted above ${DATA.high_threshold.toFixed(1)} kWh trigger an alert. A systematic audit of equipment timers (AHUs, pumps, chillers, kitchen) against occupancy schedules is recommended within 30 days.` },
  ];

  const accentMap = { high: 'linear-gradient(90deg,#ef4444,#f97316)', medium: 'linear-gradient(90deg,#f59e0b,#84cc16)', low: 'linear-gradient(90deg,#34d399,#38bdf8)' };

  container.innerHTML = insights.map(ins => `
    <div class="insight-card" style="--accent-grad:${accentMap[ins.priority]}">
      <span class="insight-emoji">${ins.emoji}</span>
      <div class="insight-title">${ins.title}</div>
      <div class="insight-text">${ins.text}</div>
      <span class="insight-priority priority-${ins.priority}">${ins.priority.toUpperCase()} PRIORITY</span>
    </div>
  `).join('');
}

// ── Model Specs ───────────────────────────────────
function buildSpecs() {
  const container = document.getElementById('specs-grid');
  if (!container || container.children.length > 0) return;

  const rmseImprovement = ((1 - DATA.gbr_rmse/DATA.baseline_rmse)*100).toFixed(1);

  const specs = [
    { label:'Algorithm', value:'GBR', desc:'Gradient Boosting Regressor' },
    { label:'Estimators', value:'200', desc:'Weak learner trees' },
    { label:'Max Depth', value:'5', desc:'Per-tree depth limit' },
    { label:'Learning Rate', value:'0.08', desc:'Shrinkage parameter' },
    { label:'Features', value:'15', desc:'Input feature count' },
    { label:'Train Hours', value:(DATA.train_size||11472).toLocaleString(), desc:'Jan 2017 – Apr 2018' },
    { label:'Test Hours', value:(DATA.test_size||744).toLocaleString(), desc:'May 2018 (31 days)' },
    { label:'Loss Function', value:'MSE', desc:'Mean Squared Error' },
    { label:'Test MAE', value:DATA.gbr_mae.toFixed(4)+' kWh', desc:'Out-of-sample error' },
    { label:'Test RMSE', value:DATA.gbr_rmse.toFixed(4)+' kWh', desc:'Root mean sq error' },
    { label:'MAE Gain', value:'+'+DATA.improvement_pct+'%', desc:'vs. naive baseline' },
    { label:'RMSE Gain', value:'+'+rmseImprovement+'%', desc:'vs. naive baseline' },
  ];

  container.innerHTML = specs.map(s => `
    <div class="spec-item">
      <div class="spec-label">${s.label}</div>
      <div class="spec-value">${s.value}</div>
      <div class="spec-desc">${s.desc}</div>
    </div>
  `).join('');
}

// ── Section Router ────────────────────────────────
function initSectionCharts(sec) {
  if (sec === 'analysis') {
    buildTestPeriodChart();
    buildHourlyChart();
    buildWeeklyChart();
    buildScatterChart();
  } else if (sec === 'insights') {
    buildInsights();
  } else if (sec === 'model') {
    buildFeatureChart();
    buildMonthlyChart();
    buildSpecs();
  }
}

// ── Init ──────────────────────────────────────────
function init() {
  buildForecastChart();
  buildErrorChart();
}

document.addEventListener('DOMContentLoaded', loadData);
