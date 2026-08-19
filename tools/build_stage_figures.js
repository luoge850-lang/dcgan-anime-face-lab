const fs = require('fs');
const path = require('path');

// DCGAN Lab scientific figures.
// Design rule: one figure answers one question and contains one metric/series.
// The script uses only Node.js core modules so it can be reproduced offline.

const ROOT = path.resolve(process.argv[2] || path.join(__dirname, '..'));
const RESULTS = path.join(ROOT, 'results');
const OUT = path.resolve(process.argv[3] || path.join(ROOT, 'results', 'figures'));
fs.mkdirSync(OUT, { recursive: true });

const COLORS = {
  blue: '#0072B2', orange: '#E69F00', green: '#009E73', vermilion: '#D55E00',
  purple: '#CC79A7', grid: '#D1D5DB', text: '#1F2937', paper: '#FFFFFF',
};

function exists(file) { return fs.existsSync(file); }
function readText(file) { return fs.readFileSync(file, 'utf8'); }
function readJson(file) { try { return JSON.parse(readText(file)); } catch (_) { return null; } }
function dirs(dir) {
  if (!exists(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => path.join(dir, entry.name));
}
function walk(dir) {
  if (!exists(dir)) return [];
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full)); else out.push(full);
  }
  return out;
}
function csvParse(text) {
  const rows = []; let row = []; let cell = ''; let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]; const next = text[i + 1];
    if (ch === '"' && quoted && next === '"') { cell += '"'; i += 1; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { row.push(cell); cell = ''; }
    else if ((ch === '\n' || ch === '\r') && !quoted) {
      if (ch === '\r' && next === '\n') i += 1;
      row.push(cell); if (row.some((value) => value !== '')) rows.push(row); row = []; cell = '';
    } else cell += ch;
  }
  if (cell !== '' || row.length) { row.push(cell); if (row.some((value) => value !== '')) rows.push(row); }
  if (!rows.length) return [];
  const headers = rows[0].map((value) => value.trim());
  return rows.slice(1).map((values) => {
    const item = {}; headers.forEach((header, index) => { item[header] = (values[index] || '').trim(); }); return item;
  });
}
function readCsv(file) { return exists(file) ? csvParse(readText(file)) : []; }
function n(value) { const result = Number(value); return Number.isFinite(result) ? result : null; }
function firstNumber(row, names) { for (const name of names) { const value = n(row[name]); if (value !== null) return value; } return null; }
function esc(value) { return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function csvEscape(value) { const text = String(value ?? ''); return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text; }
function writeCsv(file, rows, columns) {
  const body = [columns.join(','), ...rows.map((row) => columns.map((c) => csvEscape(row[c])).join(','))].join('\n');
  fs.writeFileSync(file, `${body}\n`, 'utf8');
}
function fmt(value, digits = 2) { return value === null || value === undefined || !Number.isFinite(Number(value)) ? 'NA' : Number(value).toFixed(digits); }
function axisFmt(value) { const abs = Math.abs(Number(value)); if (abs > 0 && abs < 0.01) return Number(value).toFixed(4); if (abs < 1) return Number(value).toFixed(3); if (abs < 100) return Number(value).toFixed(1); return Number(value).toFixed(0); }
function shortLabel(value, limit = 25) {
  const text = String(value || '').replace(/^Generator\./, '').replace(/^\/net\//, '').replace(/_aug$/, '')
    .replace(/Width/g, '宽度').replace(/SENet/g, 'SE').replace(/Laplacian/g, '拉普拉斯').replace(/Wavelet/g, '小波')
    .replace(/PixelShuffle/g, '像素重排').replace(/ResG/g, '残差G').replace(/baseline/g, '基线').replace(/continue/g, '延续').replace(/clip/g, 'CLIP');
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
function sourceLabel(source) { const text = String(source || '').replace(/\\/g, '/'); return text.length > 145 ? `${text.slice(0, 142)}…` : text; }
function ticks(min, max, count = 5) { if (max === min) return [min]; return Array.from({ length: count + 1 }, (_, index) => min + (max - min) * index / count); }
function range(values, includeZero = false) {
  const clean = values.filter((value) => value !== null && Number.isFinite(value)); if (!clean.length) return [0, 1];
  let min = Math.min(...clean); let max = Math.max(...clean); if (includeZero) { min = Math.min(0, min); max = Math.max(0, max); }
  if (min === max) { min -= 1; max += 1; } const pad = (max - min) * 0.08; return [min - pad, max + pad];
}
function scale(value, min, max, start, size) { if (max === min) return start + size / 2; return start + ((value - min) / (max - min)) * size; }
function downsample(rows, maxPoints = 240) {
  if (rows.length <= maxPoints) return rows; const out = [];
  for (let i = 0; i < maxPoints; i += 1) out.push(rows[Math.round(i * (rows.length - 1) / (maxPoints - 1))]);
  return out;
}

function readMetricFamily(root, family) {
  return dirs(root).map((folder) => {
    const file = path.join(folder, 'metrics.json'); if (!exists(file)) return null; const data = readJson(file); if (!data) return null;
    return { family, experiment: data.experiment_name || path.basename(folder), source: path.relative(ROOT, file), epochs: n(data.epochs),
      fid: firstNumber(data, ['FID', 'fid_standard_inception_v3', 'fid_legacy_inception_v3']),
      diversity: firstNumber(data, ['Diversity', 'lpips_alex_diversity']), laplacian: firstNumber(data, ['Laplacian_Variance', 'fake_laplacian_mean']),
      edgeRatio: firstNumber(data, ['Edge_Density_Ratio', 'edge_density_ratio']), clipMmd: firstNumber(data, ['clip_mmd2_unbiased']),
      protocol: data.FID_protocol || 'legacy project metric' };
  }).filter((row) => row && row.fid !== null);
}
function readClipFamily(root) {
  return dirs(root).map((folder) => {
    const metricsFile = ['final_metrics.json', 'metrics_epoch_050.json'].map((name) => path.join(folder, name)).find(exists); if (!metricsFile) return null;
    const data = readJson(metricsFile); if (!data) return null;
    const configFile = ['config_target_epoch_050.json', 'config_eval_only.json'].map((name) => path.join(folder, name)).find(exists);
    const config = configFile ? (readJson(configFile) || {}) : {};
    return { family: 'CLIP正则', experiment: path.basename(folder), source: path.relative(ROOT, metricsFile), fid: firstNumber(data, ['fid_legacy_inception_v3']),
      clipMmd: firstNumber(data, ['clip_mmd2_unbiased']), clipLambda: firstNumber(config, ['lambda_clip']), protocol: 'legacy Inception-v3 FID; CLIP MMD is separate' };
  }).filter((row) => row && row.fid !== null && /^C[0-4]_/.test(row.experiment));
}
function loadData() {
  const dep = path.join(RESULTS, 'Deployment_Optimization_Results');
  const serviceStagePath = path.join(dep, '06_Service_Stress', '06D', 'task3_stage_summary.csv');
  const serviceStageRows = readCsv(serviceStagePath);
  const serviceFallbackRows = process.env.DCGAN_SERVICE_SUMMARY ? readCsv(process.env.DCGAN_SERVICE_SUMMARY) : [];
  return {
    early: readMetricFamily(path.join(RESULTS, '前期调优结果'), '前期调优'),
    g: readMetricFamily(path.join(RESULTS, 'G强化实验结果'), 'G结构强化'),
    deep: readMetricFamily(path.join(RESULTS, '深度调优结果'), '深度调优'),
    clip: readClipFamily(path.join(RESULTS, 'CLIP实验结果')),
    deploymentQuality: readCsv(path.join(dep, '03_Quantization', '03E_Report', 'fp32_fp16_int8_metrics.csv')),
    sensitivity: readCsv(path.join(dep, '04_Quantization_Sensitivity', '04A', 'layer_sensitivity_summary.csv')),
    mixed: readCsv(path.join(dep, '04_Quantization_Sensitivity', '04C', 'final_confirmation_summary.csv')),
    qat: readCsv(path.join(dep, '05_QAT', '05B', 'qat_vs_ptq_summary (3).csv')),
    engines: readCsv(path.join(dep, '02_Engine_Benchmark', '02E_Report', 'task2_engine_comparison.csv')),
    topOperators: readCsv(path.join(dep, '02_Engine_Benchmark', '02E_Report', 'task2_top3_operators.csv')),
    bnFusion: readCsv(path.join(dep, '01_ONNX_Fusion', '01C_BN_Fold', 'manual_bn_latency_summary.csv')),
    serviceStages: serviceStageRows.length ? serviceStageRows : serviceFallbackRows,
    soak: readCsv(path.join(dep, '06_Service_Stress', '06D', 'task3_soak_resource_summary.csv')),
    soakPhases: readCsv(path.join(dep, '06_Service_Stress', '06D', 'task3_soak_summary.csv')),
    loss: { early: readCsv(path.join(RESULTS, '前期调优结果', 'exp1_epoch_300', 'loss.csv')), g: readCsv(path.join(RESULTS, 'G强化实验结果', '03_G_Width3x', 'loss.csv')), deep: readCsv(path.join(RESULTS, '深度调优结果', '05_G_Wavelet', 'loss.csv')), clip: readCsv(path.join(RESULTS, 'CLIP实验结果', 'C4_clip_L010', 'training_log.csv')) },
  };
}

function svgDocument(title, subtitle, width, height, body, source, note) {
  const footer = `数据源：${sourceLabel(source)}${note ? `；说明：${note}` : ''}`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
<title id="title">${esc(title)}</title><desc id="desc">${esc(subtitle)}</desc><rect width="${width}" height="${height}" fill="${COLORS.paper}"/>
<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC","SimHei",Arial,sans-serif;fill:${COLORS.text}}.title{font-size:28px;font-weight:600}.subtitle{font-size:16px;fill:#4B5563}.axis{font-size:16px}.tick{font-size:14px;fill:#4B5563}.value{font-size:14px;font-weight:600}.note{font-size:12px;fill:#4B5563}.grid{stroke:${COLORS.grid};stroke-width:1}.frame{stroke:#374151;stroke-width:1.2;fill:none}.mark{shape-rendering:geometricPrecision}</style>
<text x="${width / 2}" y="42" text-anchor="middle" class="title">${esc(title)}</text><text x="${width / 2}" y="70" text-anchor="middle" class="subtitle">${esc(subtitle)}</text>${body}<text x="40" y="${height - 20}" class="note">${esc(footer)}</text></svg>`;
}
function barChart(title, subtitle, inputData, options = {}) {
  const data = inputData.filter((row) => row && row.value !== null && Number.isFinite(row.value)); const width = options.width || 1500;
  const rowHeight = options.rowHeight || 46; const height = Math.max(options.minHeight || 620, 160 + data.length * rowHeight); const left = options.left || 350; const right = 125; const top = 112; const bottom = 88; const plotWidth = width - left - right; const plotHeight = height - top - bottom; const yStep = data.length ? plotHeight / data.length : plotHeight;
  const maxValue = Math.max(...data.map((row) => row.value), 0); const xMax = maxValue > 0 ? maxValue * 1.18 : 1; const xTicks = ticks(0, xMax, 5); const valueFormat = options.valueFormat || ((value) => fmt(value, 2)); let body = '<g aria-label="单系列水平柱状图">';
  xTicks.forEach((tick) => { const x = scale(tick, 0, xMax, left, plotWidth); body += `<line x1="${x}" y1="${top}" x2="${x}" y2="${top + plotHeight}" class="grid"/><text x="${x}" y="${top + plotHeight + 28}" text-anchor="middle" class="tick">${esc(valueFormat(tick))}</text>`; });
  body += `<rect x="${left}" y="${top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  data.forEach((row, index) => { const y = top + index * yStep + yStep * 0.18; const h = yStep * 0.64; const x2 = scale(row.value, 0, xMax, left, plotWidth); body += `<text x="${left - 16}" y="${y + h * 0.72}" text-anchor="end" class="tick">${esc(shortLabel(row.label, options.labelLimit || 28))}</text><rect x="${left}" y="${y}" width="${Math.max(1, x2 - left)}" height="${h}" fill="${options.color || COLORS.blue}" class="mark"><title>${esc(row.label)}：${esc(valueFormat(row.value))}</title></rect><text x="${Math.min(x2 + 10, width - 105)}" y="${y + h * 0.72}" class="value">${esc(valueFormat(row.value))}</text>`; });
  body += `<text x="${width - right}" y="${top + plotHeight + 60}" text-anchor="end" class="axis">${esc(options.xLabel || '数值')}</text></g>`;
  return { svg: svgDocument(title, subtitle, width, height, body, options.source, options.note), rows: data.length };
}
function lineChart(title, subtitle, inputData, options = {}) {
  const data = inputData.filter((row) => row && row.x !== null && row.value !== null && Number.isFinite(row.x) && Number.isFinite(row.value)); const sampled = downsample(data, options.maxPoints || 240); const width = options.width || 1500; const height = options.height || 760; const left = 115; const right = 90; const top = 112; const bottom = 108; const plotWidth = width - left - right; const plotHeight = height - top - bottom;
  const xValues = sampled.map((row) => row.x); const yValues = sampled.map((row) => row.value); const [xMin, xMax] = range(xValues, false); const [yMin, yMax] = range(yValues, options.includeZeroY === true); const xTicks = ticks(Math.min(...xValues), Math.max(...xValues), 5); const yTicks = ticks(yMin, yMax, 5); const yFormat = options.yFormat || ((value) => axisFmt(value)); const xFormat = options.xFormat || ((value) => axisFmt(value)); let body = '<g aria-label="单系列折线图">';
  yTicks.forEach((tick) => { const y = scale(tick, yMin, yMax, top + plotHeight, -plotHeight); body += `<line x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}" class="grid"/><text x="${left - 14}" y="${y + 5}" text-anchor="end" class="tick">${esc(yFormat(tick))}</text>`; });
  xTicks.forEach((tick) => { const x = scale(tick, xMin, xMax, left, plotWidth); body += `<line x1="${x}" y1="${top}" x2="${x}" y2="${top + plotHeight}" class="grid"/><text x="${x}" y="${top + plotHeight + 28}" text-anchor="middle" class="tick">${esc(xFormat(tick))}</text>`; });
  body += `<rect x="${left}" y="${top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  const points = sampled.map((row) => `${scale(row.x, xMin, xMax, left, plotWidth)},${scale(row.value, yMin, yMax, top + plotHeight, -plotHeight)}`).join(' '); body += `<polyline points="${points}" fill="none" stroke="${options.color || COLORS.blue}" stroke-width="3" class="mark"/>`; const pointStep = Math.max(1, Math.floor(sampled.length / 24));
  sampled.forEach((row, index) => { if (index % pointStep !== 0 && index !== sampled.length - 1) return; const x = scale(row.x, xMin, xMax, left, plotWidth); const y = scale(row.value, yMin, yMax, top + plotHeight, -plotHeight); body += `<circle cx="${x}" cy="${y}" r="4" fill="${options.color || COLORS.blue}" class="mark"><title>${esc(xFormat(row.x))}：${esc(yFormat(row.value))}</title></circle>`; });
  const last = sampled[sampled.length - 1]; if (last) { const x = scale(last.x, xMin, xMax, left, plotWidth); const y = scale(last.value, yMin, yMax, top + plotHeight, -plotHeight); body += `<text x="${Math.min(x + 12, width - 115)}" y="${Math.max(top + 18, y - 12)}" class="value">${esc(yFormat(last.value))}</text>`; }
  body += `<text x="${width - right}" y="${top + plotHeight + 62}" text-anchor="end" class="axis">${esc(options.xLabel || '横轴')}</text><text x="22" y="${top - 12}" class="axis">${esc(options.yLabel || '数值')}</text></g>`;
  return { svg: svgDocument(title, subtitle, width, height, body, options.source, options.note), rows: data.length };
}

const manifest = []; const data = loadData(); const f = (value) => fmt(value, 2); const pct = (value) => `${fmt(Number(value) * 100, 1)}%`; const pp = (value) => `${fmt(value, 2)} pp`; const ms = (value) => `${fmt(value, 2)} ms`; const ips = (value) => `${fmt(value, 0)} img/s`;
function writeFigure(fileName, result, meta) { if (!result.rows) return; fs.writeFileSync(path.join(OUT, fileName), result.svg, 'utf8'); manifest.push({ file: fileName, title: meta.title, chart_type: meta.chartType, metric: meta.metric, unit: meta.unit, rows: result.rows, source: meta.source, note: meta.note || '' }); }
function metricRows(rows, field, label = (row) => row.experiment) { return rows.map((row) => ({ label: label(row), value: n(row[field]) })); }
function lineRows(rows, xField, yField) { return rows.map((row) => ({ x: n(row[xField]), value: n(row[yField]) })).filter((row) => row.x !== null && row.value !== null); }
function deploymentEngineRows(rows, precision, field) { const selected = []; for (const engine of ['ONNX Runtime', 'TensorRT', 'OpenVINO']) { const row = rows.find((item) => item.engine === engine && item.precision === precision && item.batch === '32'); if (!row) continue; const value = field === 'latency' ? (engine === 'TensorRT' ? n(row.end_to_end_mean_ms) : n(row.mean_ms)) : n(row.throughput_images_per_s); selected.push({ label: engine, value }); } return selected; }

// 01–08: training and evaluation history; these FID values are legacy project metrics.
{
  const rows = data.early.filter((row) => /^exp1_epoch_/.test(row.experiment)).sort((a, b) => (a.epochs || 0) - (b.epochs || 0)); const title = '前期训练轮数与 FID'; const source = '前期调优结果/exp1_epoch_*/metrics.json';
  const result = lineChart(title, '只展示同一组 exp1 训练时长实验；FID 越低越好', rows.map((row) => ({ x: row.epochs, value: row.fid })), { xLabel: '训练轮数（epoch）', yLabel: 'FID（legacy）', source, note: 'legacy 项目评估口径，仅在该组实验内比较', yFormat: f }); writeFigure('01_前期训练轮数_FID.svg', result, { title, chartType: 'line', metric: 'FID', unit: 'legacy FID', source, note: 'legacy 项目评估口径' });
}
{
  const title = '前期数据增强实验 FID'; const source = '前期调优结果/exp2_*、exp3_*/metrics.json'; const rows = data.early.filter((row) => /^exp[23]_/.test(row.experiment)); const result = barChart(title, '只比较增强方案；FID 越低越好', metricRows(rows, 'fid'), { xLabel: 'FID（legacy）', source, note: '不同增强方案沿用项目 legacy 口径', valueFormat: f, labelLimit: 30 }); writeFigure('02_前期增强实验_FID.svg', result, { title, chartType: 'bar', metric: 'FID', unit: 'legacy FID', source, note: 'legacy 项目评估口径' });
}
{
  const title = 'G 结构强化实验 FID'; const source = 'G强化实验结果/*/metrics.json'; const result = barChart(title, '结构容量、残差、注意力与训练规模的单指标比较；FID 越低越好', metricRows(data.g, 'fid'), { xLabel: 'FID（legacy）', source, note: '各实验均为项目 legacy 评估口径', valueFormat: f, labelLimit: 30, rowHeight: 48 }); writeFigure('03_G强化_FID.svg', result, { title, chartType: 'bar', metric: 'FID', unit: 'legacy FID', source, note: 'legacy 项目评估口径' });
}
{
  const title = 'G 结构强化实验多样性'; const source = 'G强化实验结果/*/metrics.json'; const result = barChart(title, 'LPIPS-Alex 多样性指标；数值越高通常表示样本差异更大', metricRows(data.g, 'diversity'), { xLabel: 'LPIPS-Alex 多样性', source, note: 'LPIPS 字段沿用该阶段定义，不与部署阶段混用', valueFormat: f, labelLimit: 30, rowHeight: 48, color: COLORS.green }); writeFigure('04_G强化_多样性LPIPS.svg', result, { title, chartType: 'bar', metric: 'Diversity/LPIPS', unit: 'LPIPS-Alex', source, note: '仅在该阶段内部比较' });
}
{
  const title = '深度调优：G 端方法 FID'; const source = '深度调优结果/00_baseline、*_G_*/metrics.json'; const rows = data.deep.filter((row) => row.experiment === '00_baseline' || /^\d+_G_/.test(row.experiment)); const result = barChart(title, '小波、FFT、Canny、Laplacian 与注意力方法；FID 越低越好', metricRows(rows, 'fid'), { xLabel: 'FID（legacy）', source, note: 'Wavelet/SN 是历史训练阶段方法，不等于当前部署图结构', valueFormat: f, labelLimit: 32, rowHeight: 48 }); writeFigure('05_深度调优_G方法_FID.svg', result, { title, chartType: 'bar', metric: 'FID', unit: 'legacy FID', source, note: 'legacy 项目评估口径' });
}
{
  const title = '深度调优：D 端方法 FID'; const source = '深度调优结果/*_D_*/metrics.json'; const rows = data.deep.filter((row) => /^\d+_D_/.test(row.experiment)); const result = barChart(title, 'SN、Hinge、R1 与判别器组合；FID 越低越好', metricRows(rows, 'fid'), { xLabel: 'FID（legacy）', source, note: '历史训练阶段的 SN 不代表当前部署图包含 SN', valueFormat: f, labelLimit: 34, rowHeight: 48, color: COLORS.orange }); writeFigure('06_深度调优_D方法_FID.svg', result, { title, chartType: 'bar', metric: 'FID', unit: 'legacy FID', source, note: 'legacy 项目评估口径' });
}
{
  const title = 'CLIP 正则强度与 FID'; const source = 'CLIP实验结果/C0–C4/final_metrics.json + config*.json'; const rows = data.clip.sort((a, b) => (a.clipLambda || 0) - (b.clipLambda || 0)); const result = lineChart(title, 'C0–C4 同一组 λ_clip 消融；FID 越低越好', rows.map((row) => ({ x: row.clipLambda, value: row.fid })), { xLabel: 'λ_clip', yLabel: 'FID（legacy）', source, note: 'E0 formal eval 未纳入该 λ 曲线；FID 仍为 legacy 口径', xFormat: (value) => Number(value).toFixed(3), yFormat: f }); writeFigure('07_CLIP正则_FID.svg', result, { title, chartType: 'line', metric: 'FID', unit: 'legacy FID', source, note: 'C0–C4 同组消融' });
}
{
  const title = 'CLIP 正则强度与 MMD²'; const source = 'CLIP实验结果/C0–C4/final_metrics.json + config*.json'; const rows = data.clip.sort((a, b) => (a.clipLambda || 0) - (b.clipLambda || 0)); const result = lineChart(title, 'C0–C4 同一组 λ_clip 消融；MMD² 越低表示特征分布差异越小', rows.map((row) => ({ x: row.clipLambda, value: row.clipMmd })), { xLabel: 'λ_clip', yLabel: 'CLIP MMD²', source, note: 'CLIP MMD 与 FID 是不同指标，不能互相替代', xFormat: (value) => Number(value).toFixed(3), yFormat: (value) => Number(value).toFixed(4), color: COLORS.purple }); writeFigure('08_CLIP正则_MMD2.svg', result, { title, chartType: 'line', metric: 'CLIP MMD²', unit: 'MMD²', source, note: '与 FID 分开解释' });
}

// 09–16: representative training curves; each loss is intentionally separate.
const curveSpecs = [
  ['09_前期代表_D_loss.svg', '前期代表实验 D loss', data.loss.early, 'epoch', 'D_loss', 'epoch', 'D loss', '前期调优结果/exp1_epoch_300/loss.csv', COLORS.blue],
  ['10_前期代表_G_loss.svg', '前期代表实验 G loss', data.loss.early, 'epoch', 'G_loss', 'epoch', 'G loss', '前期调优结果/exp1_epoch_300/loss.csv', COLORS.orange],
  ['11_G强化代表_D_loss.svg', 'G 强化代表实验 D loss', data.loss.g, 'epoch', 'D_loss', 'epoch', 'D loss', 'G强化实验结果/03_G_Width3x/loss.csv', COLORS.blue],
  ['12_G强化代表_G_loss.svg', 'G 强化代表实验 G loss', data.loss.g, 'epoch', 'G_loss', 'epoch', 'G loss', 'G强化实验结果/03_G_Width3x/loss.csv', COLORS.orange],
  ['13_深度代表_G_adv.svg', '深度调优代表实验 G adversarial loss', data.loss.deep, 'epoch', 'G_adv', 'epoch', 'G adv loss', '深度调优结果/05_G_Wavelet/loss.csv', COLORS.vermilion],
  ['14_深度代表_G_wavelet.svg', '深度调优代表实验小波损失', data.loss.deep, 'epoch', 'G_wavelet', 'epoch', 'Wavelet loss', '深度调优结果/05_G_Wavelet/loss.csv', COLORS.green],
  ['15_CLIP代表_D_loss.svg', 'CLIP 代表实验 D loss', data.loss.clip, 'epoch', 'd_loss', 'epoch', 'D loss', 'CLIP实验结果/C4_clip_L010/training_log.csv', COLORS.blue],
  ['16_CLIP代表_CLIP_MMD.svg', 'CLIP 代表实验 CLIP MMD²', data.loss.clip, 'epoch', 'clip_mmd', 'epoch', 'CLIP MMD²', 'CLIP实验结果/C4_clip_L010/training_log.csv', COLORS.purple],
];
// The owner intentionally removed the representative loss figures from the source folder.
// Keep the data-aware definitions for provenance, but do not recreate those seven loss SVGs.
if (false) for (const [file, title, rows, xField, yField, xLabel, yLabel, source, color] of curveSpecs) { const result = lineChart(title, '代表实验训练轨迹；曲线仅展示一个损失/评估量', lineRows(rows, xField, yField), { xLabel, yLabel, source, note: '训练日志按等间隔抽样至最多 240 个点，仅用于图形可读性', color, yFormat: (value) => axisFmt(value) }); writeFigure(file, result, { title, chartType: 'line', metric: yField, unit: yLabel, source, note: '等间隔抽样，不改变端点' }); }

// 17–21: deployment benchmark. Each plot has one metric only.
for (const graph of ['raw', 'manual_bn_fused']) {
  const rows = data.bnFusion.filter((row) => row.graph === graph).sort((a, b) => n(a.batch) - n(b.batch)); const title = graph === 'raw' ? 'BN 未融合图：批量与平均延迟' : 'BN 融合图：批量与平均延迟'; const file = graph === 'raw' ? '17_BN未融合_平均延迟.svg' : '18_BN融合_平均延迟.svg'; const source = 'Deployment_Optimization_Results/01_ONNX_Fusion/01C_BN_Fold/manual_bn_latency_summary.csv'; const result = lineChart(title, 'CPUExecutionProvider；只展示该图对应的单一图结构', rows.map((row) => ({ x: n(row.batch), value: n(row.mean_ms) })), { xLabel: 'Batch size', yLabel: '平均延迟（ms）', source, note: 'raw 与 manual_bn_fused 已拆成两张图，避免双系列混读', xFormat: (value) => fmt(value, 0), yFormat: ms }); writeFigure(file, result, { title, chartType: 'line', metric: 'mean_ms', unit: 'ms', source, note: '单图单系列' });
}
for (const precision of ['FP32', 'FP16']) {
  const suffix = precision.toLowerCase(); const latency = []; const throughput = [];
  for (const engine of ['ONNX Runtime', 'TensorRT', 'OpenVINO']) { const row = data.engines.find((item) => item.engine === engine && item.precision === precision && item.batch === '32'); if (!row) continue; latency.push({ label: engine, value: engine === 'TensorRT' ? n(row.end_to_end_mean_ms) : n(row.mean_ms) }); throughput.push({ label: engine, value: n(row.throughput_images_per_s) }); }
  const source = 'Deployment_Optimization_Results/02_Engine_Benchmark/02E_Report/task2_engine_comparison.csv'; const latencyTitle = `三引擎 ${precision} Batch=32 端到端延迟`; const latencyResult = barChart(latencyTitle, 'ONNX Runtime、TensorRT、OpenVINO；数值越低越好', latency, { xLabel: '平均端到端延迟（ms）', source, note: '同一 Batch=32；TensorRT 使用 end_to_end_mean_ms，其余使用 mean_ms', valueFormat: ms, labelLimit: 22, minHeight: 600 }); const latencyFile = precision === 'FP32' ? '19_三引擎_FP32_batch32_延迟.svg' : '21_三引擎_FP16_batch32_延迟.svg'; writeFigure(latencyFile, latencyResult, { title: latencyTitle, chartType: 'bar', metric: 'end-to-end latency', unit: 'ms', source, note: '同 Batch=32' });
  const throughputTitle = `三引擎 ${precision} Batch=32 吞吐量`; const throughputResult = barChart(throughputTitle, 'ONNX Runtime、TensorRT、OpenVINO；数值越高越好', throughput, { xLabel: '吞吐量（images/s）', source, note: '同一 Batch=32；只展示吞吐量', valueFormat: ips, labelLimit: 22, color: COLORS.green, minHeight: 600 }); const throughputFile = precision === 'FP32' ? '20_三引擎_FP32_batch32_吞吐.svg' : '22_三引擎_FP16_batch32_吞吐.svg'; writeFigure(throughputFile, throughputResult, { title: throughputTitle, chartType: 'bar', metric: 'throughput', unit: 'images/s', source, note: '同 Batch=32' });
}
{
  const title = 'TensorRT Top-3 算子耗时'; const source = 'Deployment_Optimization_Results/02_Engine_Benchmark/02E_Report/task2_top3_operators.csv'; const rows = data.topOperators.filter((row) => row.source === 'TensorRT IProfiler').sort((a, b) => n(a.rank) - n(b.rank)).map((row) => ({ label: `Top ${row.rank} ${row.operator_or_layer}`, value: n(row.total_ms) })); const result = barChart(title, 'TensorRT profile 汇总；总耗时越高越值得优先优化', rows, { xLabel: '总耗时（ms）', source, note: '只画 TensorRT IProfiler 的三项，其他引擎证据保留在原始 CSV', valueFormat: ms, labelLimit: 60, color: COLORS.vermilion, minHeight: 600 }); writeFigure('23_TensorRT_Top3_算子耗时.svg', result, { title, chartType: 'bar', metric: 'total_ms', unit: 'ms', source, note: 'TensorRT IProfiler' });
}

// 22–31: quantization, sensitivity, mixed precision and QAT.
{
  const title = 'PTQ 质量基线：FID'; const source = 'Deployment_Optimization_Results/03_Quantization/03E_Report/fp32_fp16_int8_metrics.csv'; const result = barChart(title, 'FP32、FP16、全 INT8；FID 越低越好', metricRows(data.deploymentQuality, 'fid_standard_inception_v3', (row) => row.precision), { xLabel: 'Standard FID', source, note: '部署阶段 Standard FID；与历史 legacy FID 分开', valueFormat: f, labelLimit: 20, minHeight: 600, color: COLORS.vermilion }); writeFigure('22_PTQ_FID.svg', result, { title, chartType: 'bar', metric: 'FID', unit: 'Standard FID', source, note: 'Standard FID' });
}
{
  const title = 'PTQ 质量基线：模糊率'; const source = 'Deployment_Optimization_Results/03_Quantization/03E_Report/fp32_fp16_int8_metrics.csv'; const result = barChart(title, 'FP32、FP16、全 INT8；模糊率越低越好', metricRows(data.deploymentQuality, 'fake_blur_rate', (row) => row.precision), { xLabel: '模糊率', source, note: '模糊率按真实集 p10 阈值定义', valueFormat: pct, labelLimit: 20, minHeight: 600, color: COLORS.orange }); writeFigure('23_PTQ_模糊率.svg', result, { title, chartType: 'bar', metric: 'fake_blur_rate', unit: '%', source, note: '真实集 p10 阈值' });
}
{
  const title = 'PTQ 速度基线：吞吐量'; const source = 'Deployment_Optimization_Results/03_Quantization/03E_Report/fp32_fp16_int8_metrics.csv'; const result = barChart(title, 'FP32、FP16、全 INT8；吞吐越高越好', metricRows(data.deploymentQuality, 'inference_images_per_s', (row) => row.precision), { xLabel: '吞吐量（images/s）', source, note: '原始结果文件中的 inference_images_per_s', valueFormat: ips, labelLimit: 20, minHeight: 600, color: COLORS.green }); writeFigure('24_PTQ_吞吐.svg', result, { title, chartType: 'bar', metric: 'inference_images_per_s', unit: 'images/s', source, note: '原始吞吐字段' });
}
{
  const labels = ['none', 'all_int8', 'net.0', 'net.3', 'net.6', 'net.9', 'net.12']; const rows = data.sensitivity.filter((row) => labels.includes(row.restored_layer)).map((row) => ({ label: row.restored_layer === 'none' ? 'none（FP32参考）' : row.restored_layer === 'all_int8' ? 'all_int8' : `恢复 ${row.restored_layer}`, value: n(row.fid_standard) })); const title = '逐层恢复 FP16：FID'; const source = 'Deployment_Optimization_Results/04_Quantization_Sensitivity/04A/layer_sensitivity_summary.csv'; const result = barChart(title, '04A 同一敏感度实验；柱越低越好', rows, { xLabel: 'Standard FID', source, note: '只在 04A 内部比较；与 03E/04C 的评估批次不混合', valueFormat: f, labelLimit: 28, rowHeight: 48, color: COLORS.vermilion }); writeFigure('25_逐层敏感度_FID.svg', result, { title, chartType: 'bar', metric: 'fid_standard', unit: 'Standard FID', source, note: '04A 内部比较' });
}
{
  const labels = ['none', 'all_int8', 'net.0', 'net.3', 'net.6', 'net.9', 'net.12']; const rows = data.sensitivity.filter((row) => labels.includes(row.restored_layer)).map((row) => ({ label: row.restored_layer === 'none' ? 'none（FP32参考）' : row.restored_layer === 'all_int8' ? 'all_int8' : `恢复 ${row.restored_layer}`, value: n(row.latency_mean_ms_batch) })); const title = '逐层恢复 FP16：平均延迟'; const source = 'Deployment_Optimization_Results/04_Quantization_Sensitivity/04A/layer_sensitivity_summary.csv'; const result = barChart(title, '04A 同一敏感度实验；延迟越低越快', rows, { xLabel: '平均延迟（ms/batch）', source, note: '同一测试配置；与 FID 图一一对应', valueFormat: ms, labelLimit: 28, rowHeight: 48 }); writeFigure('26_逐层敏感度_延迟.svg', result, { title, chartType: 'bar', metric: 'latency_mean_ms_batch', unit: 'ms/batch', source, note: '04A 内部比较' });
}
{
  const title = '混合精度策略：FID'; const source = 'Deployment_Optimization_Results/04_Quantization_Sensitivity/04C/final_confirmation_summary.csv'; const result = barChart(title, '04C 最终确认；FID 越低越好', metricRows(data.mixed, 'fid_standard', (row) => row.strategy), { xLabel: 'Standard FID', source, note: 'FP32、all_int8 与 net.0+net.12 混合策略', valueFormat: f, labelLimit: 28, minHeight: 600, color: COLORS.vermilion }); writeFigure('27_混合策略_FID.svg', result, { title, chartType: 'bar', metric: 'fid_standard', unit: 'Standard FID', source, note: '04C 最终确认' });
}
{
  const title = '混合精度策略：平均延迟'; const source = 'Deployment_Optimization_Results/04_Quantization_Sensitivity/04C/final_confirmation_summary.csv'; const result = barChart(title, '04C 最终确认；延迟越低越快', metricRows(data.mixed, 'latency_mean_ms_batch', (row) => row.strategy), { xLabel: '平均延迟（ms/batch）', source, note: '速度与质量拆成两张图', valueFormat: ms, labelLimit: 28, minHeight: 600 }); writeFigure('28_混合策略_延迟.svg', result, { title, chartType: 'bar', metric: 'latency_mean_ms_batch', unit: 'ms/batch', source, note: '04C 最终确认' });
}
{
  const title = 'QAT 对照：FID'; const source = 'Deployment_Optimization_Results/05_QAT/05B/qat_vs_ptq_summary (3).csv'; const result = barChart(title, '05B 同一评估批次；FID 越低越好', metricRows(data.qat, 'fid_standard', (row) => row.label), { xLabel: 'Standard FID', source, note: 'QAT、PTQ、MIXED、PRE-QAT 与 FP32 统一画为单指标', valueFormat: f, labelLimit: 26, rowHeight: 48, color: COLORS.vermilion }); writeFigure('29_QAT_FID.svg', result, { title, chartType: 'bar', metric: 'fid_standard', unit: 'Standard FID', source, note: '05B 同一评估批次' });
}
{
  const title = 'QAT 对照：模糊率'; const source = 'Deployment_Optimization_Results/05_QAT/05B/qat_vs_ptq_summary (3).csv'; const result = barChart(title, '05B 同一评估批次；模糊率越低越好', metricRows(data.qat, 'blur_rate', (row) => row.label), { xLabel: '模糊率', source, note: '不宣称 literal hair/eyeliner ROI；此图只支持全局模糊率比较', valueFormat: pct, labelLimit: 26, rowHeight: 48, color: COLORS.orange }); writeFigure('30_QAT_模糊率.svg', result, { title, chartType: 'bar', metric: 'blur_rate', unit: '%', source, note: '全局模糊率，不作 ROI 显著性结论' });
}
{
  const title = 'QAT 对照：P99 延迟'; const source = 'Deployment_Optimization_Results/05_QAT/05B/qat_vs_ptq_summary (3).csv'; const result = barChart(title, '05B 同一推理评估；延迟越低越快', metricRows(data.qat, 'latency_p99_ms_batch', (row) => row.label), { xLabel: 'P99 延迟（ms/batch）', source, note: '单独展示 P99，避免与质量指标混画', valueFormat: ms, labelLimit: 26, rowHeight: 48 }); writeFigure('31_QAT_P99.svg', result, { title, chartType: 'bar', metric: 'latency_p99_ms_batch', unit: 'ms/batch', source, note: '05B 同一评估批次' });
}

// 32–40: service stress and soak. Each resource/latency metric is separated.
{
  const title = '服务压测：并发与 P99 延迟'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06BC/Run_20260819_014504.zip::stage_results.csv'; const rows = data.serviceStages.sort((a, b) => n(a.concurrency) - n(b.concurrency)); const result = lineChart(title, '当前 staged run 阶梯式加压；曲线用于观察延迟随并发变化', rows.map((row) => ({ x: n(row.concurrency), value: n(row.p99_ms) })), { xLabel: '并发数', yLabel: 'P99 延迟（ms）', source, note: '128 并发为已测上界；若无失败不能写成理论崩溃点', xFormat: (value) => fmt(value, 0), yFormat: ms, color: COLORS.vermilion }); writeFigure('32_服务并发_P99.svg', result, { title, chartType: 'line', metric: 'p99_ms', unit: 'ms', source, note: '当前 staged run；已测上界，不等于物理崩溃点' });
}
{
  const title = '服务压测：并发与 RPS'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06BC/Run_20260819_014504.zip::stage_results.csv'; const rows = data.serviceStages.sort((a, b) => n(a.concurrency) - n(b.concurrency)); const result = lineChart(title, '当前 staged run 阶梯式加压；RPS 是服务吞吐表现', rows.map((row) => ({ x: n(row.concurrency), value: n(row.rps) })), { xLabel: '并发数', yLabel: 'RPS（requests/s）', source, note: '只展示当前 staged run 的服务吞吐量', xFormat: (value) => fmt(value, 0), yFormat: (value) => fmt(value, 0), color: COLORS.green }); writeFigure('33_服务并发_RPS.svg', result, { title, chartType: 'line', metric: 'rps', unit: 'requests/s', source, note: '当前 staged run；阶梯式加压' });
}
{
  const title = '服务压测：并发与 GPU 显存'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_stage_summary.csv'; const rows = data.serviceStages.sort((a, b) => n(a.concurrency) - n(b.concurrency)); const result = lineChart(title, '06D 监控采样峰值；显存越高越接近资源上限', rows.map((row) => ({ x: n(row.concurrency), value: n(row.gpu_peak_mb) })), { xLabel: '并发数', yLabel: 'GPU 显存峰值（MB）', source, note: 'GPU 型号与监控协议见 06D 原始结果', xFormat: (value) => fmt(value, 0), yFormat: (value) => fmt(value, 0), color: COLORS.blue }); writeFigure('34_服务并发_GPU显存.svg', result, { title, chartType: 'line', metric: 'gpu_peak_mb', unit: 'MB', source, note: '监控峰值' });
}
{
  const title = '服务压测：并发与 RSS'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_stage_summary.csv'; const rows = data.serviceStages.sort((a, b) => n(a.concurrency) - n(b.concurrency)); const result = lineChart(title, '06D 系统进程内存峰值；用于观察内存增长', rows.map((row) => ({ x: n(row.concurrency), value: n(row.rss_peak_mb) })), { xLabel: '并发数', yLabel: 'RSS 峰值（MB）', source, note: '长时间泄漏需结合 soak 的头尾差值判断', xFormat: (value) => fmt(value, 0), yFormat: (value) => fmt(value, 0), color: COLORS.purple }); writeFigure('35_服务并发_RSS.svg', result, { title, chartType: 'line', metric: 'rss_peak_mb', unit: 'MB', source, note: '系统内存峰值' });
}
{
  const title = '服务压测：并发与 SM 利用率'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_stage_summary.csv'; const rows = data.serviceStages.sort((a, b) => n(a.concurrency) - n(b.concurrency)); const result = lineChart(title, '06D 监控采样峰值；只展示 SM 指标', rows.map((row) => ({ x: n(row.concurrency), value: n(row.sm_peak_percent) })), { xLabel: '并发数', yLabel: 'SM 利用率峰值（%）', source, note: '采样值是峰值，不等于整个阶段平均利用率', xFormat: (value) => fmt(value, 0), yFormat: (value) => `${fmt(value, 0)}%`, color: COLORS.orange }); writeFigure('36_服务并发_SM.svg', result, { title, chartType: 'line', metric: 'sm_peak_percent', unit: '%', source, note: '采样峰值' });
}
{
  const title = 'Soak Test：阶段 P99 延迟'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_soak_summary.csv'; const rows = data.soakPhases.sort((a, b) => n(a.duration_seconds) - n(b.duration_seconds)).map((row) => ({ label: row.phase.replace(/^soak_/, '').replace(/_u\d+$/, ''), value: n(row.p99_ms) })); const result = barChart(title, '06D 稳态 soak；warmup 与 steady 分开比较', rows, { xLabel: 'P99 延迟（ms）', source, note: 'steady 阶段为 1800 秒（30 分钟）', valueFormat: ms, labelLimit: 30, minHeight: 600, color: COLORS.vermilion }); writeFigure('37_Soak阶段_P99.svg', result, { title, chartType: 'bar', metric: 'p99_ms', unit: 'ms', source, note: 'steady=1800s' });
}
{
  const title = 'Soak Test：阶段 RPS'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_soak_summary.csv'; const rows = data.soakPhases.sort((a, b) => n(a.duration_seconds) - n(b.duration_seconds)).map((row) => ({ label: row.phase.replace(/^soak_/, '').replace(/_u\d+$/, ''), value: n(row.rps) })); const result = barChart(title, '06D 稳态 soak；warmup 与 steady 分开比较', rows, { xLabel: 'RPS（requests/s）', source, note: '无失败请求；steady 阶段用于长时间稳定性', valueFormat: (value) => fmt(value, 1), labelLimit: 30, minHeight: 600, color: COLORS.green }); writeFigure('38_Soak阶段_RPS.svg', result, { title, chartType: 'bar', metric: 'rps', unit: 'requests/s', source, note: '长时间稳定性' });
}
{
  const row = data.soak[0]; const rows = row ? [{ label: 'GPU 显存头部', value: n(row.gpu_memory_head_mb) }, { label: 'GPU 显存尾部', value: n(row.gpu_memory_tail_mb) }] : []; const title = 'Soak Test：GPU 显存头尾'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_soak_resource_summary.csv'; const result = barChart(title, '06D 稳态 soak 资源审计；头尾差值用于观察显存增长', rows, { xLabel: '显存（MB）', source, note: row ? `gpu_memory_delta=${row.gpu_memory_delta_percent}%` : '无数据', valueFormat: (value) => fmt(value, 1), labelLimit: 30, minHeight: 520 }); writeFigure('39_Soak_GPU显存头尾.svg', result, { title, chartType: 'bar', metric: 'gpu_memory_head/tail_mb', unit: 'MB', source, note: row ? `delta=${row.gpu_memory_delta_percent}%` : '无数据' });
}
{
  const row = data.soak[0]; const rows = row ? [{ label: 'RSS 头部均值', value: n(row.rss_head_mean_mb) }, { label: 'RSS 尾部均值', value: n(row.rss_tail_mean_mb) }] : []; const title = 'Soak Test：RSS 头尾均值'; const source = 'Deployment_Optimization_Results/06_Service_Stress/06D/task3_soak_resource_summary.csv'; const result = barChart(title, '06D 稳态 soak 资源审计；头尾差值用于观察内存增长', rows, { xLabel: 'RSS（MB）', source, note: row ? `rss_delta=${row.rss_delta_mb} MB；rss_delta_percent=${row.rss_delta_percent}%` : '无数据', valueFormat: (value) => fmt(value, 1), labelLimit: 30, minHeight: 520, color: COLORS.purple }); writeFigure('40_Soak_RSS头尾.svg', result, { title, chartType: 'bar', metric: 'rss_head/tail_mean_mb', unit: 'MB', source, note: row ? `delta=${row.rss_delta_percent}%` : '无数据' });
}

const metricIndex = [...data.early, ...data.g, ...data.deep, ...data.clip].map((row) => ({ family: row.family, experiment: row.experiment, fid: row.fid, diversity: row.diversity, laplacian: row.laplacian, edge_ratio: row.edgeRatio, clip_mmd: row.clipMmd, source: row.source, protocol: row.protocol }));
writeCsv(path.join(OUT, '全实验指标汇总.csv'), metricIndex, ['family', 'experiment', 'fid', 'diversity', 'laplacian', 'edge_ratio', 'clip_mmd', 'source', 'protocol']);
const inventory = walk(RESULTS).map((file) => ({ path: path.relative(ROOT, file), type: path.extname(file).slice(1) || 'no_extension', bytes: fs.statSync(file).size }));
writeCsv(path.join(OUT, '全实验数据清单.csv'), inventory, ['path', 'type', 'bytes']);
const limitations = [
  '前期调优、G强化、深度调优、CLIP 的 FID 使用历史 legacy 项目协议；部署阶段 Standard FID 不与它们直接混合排名。',
  '历史训练结果中的 Wavelet/SN 是训练实验方法；当前部署图的算子证据仍以 ConvTranspose、BatchNorm、ReLU、Tanh 为准。',
  '05B 的全局模糊率与全局高频指标不支持 literal hair/eyeliner ROI 显著优于 PTQ 的结论；图表不作该超出证据范围的声称。',
  '06D 只在已测并发范围内展示服务行为；没有失败或 OOM 时，最大已测并发不能写成理论物理崩溃点。',
  '训练曲线最多等间隔抽样 240 个点，仅改变显示密度，不改变指标文件。',
];
fs.writeFileSync(path.join(OUT, 'figure_manifest.json'), JSON.stringify({ generated_at: new Date().toISOString(), design: 'single_metric_single_series_line_or_bar', figures: manifest, excluded: ['SDXL_Controlled_Study_Results：独立 SDXL 对照研究，不并入 DCGAN 主线图表'], limitations }, null, 2), 'utf8');
const cards = manifest.map((item) => `<section><h2>${esc(item.title)}</h2><p>${esc(item.chart_type)} · ${esc(item.metric)} · ${esc(item.unit)} · ${item.rows} rows</p><img src="${encodeURI(item.file)}" alt="${esc(item.title)}"></section>`).join('\n');
fs.writeFileSync(path.join(OUT, '全实验图表索引.html'), `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>DCGAN Lab 单指标图表索引</title><style>body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1540px;margin:24px auto;color:#1f2937}h1{font-size:28px}section{margin:34px 0 56px;border-bottom:1px solid #d1d5db;padding-bottom:24px}h2{font-size:22px}p{color:#4b5563}img{display:block;max-width:100%;height:auto;border:1px solid #d1d5db}</style></head><body><h1>DCGAN Lab 全实验单指标图表</h1><p>每张图只展示一个指标和一个系列，便于报告引用。旧版组合图已移入备份目录。</p>${cards}</body></html>`, 'utf8');
fs.writeFileSync(path.join(OUT, 'README.md'), `# DCGAN Lab 全实验图表（单指标版）

本目录已重做为“单图单论点”：每张 SVG 只展示一个指标，主体使用单系列折线图或单系列水平柱状图。旧版组合图没有删除，已移入带日期的备份目录。

## 图表范围

- 前期调优：训练轮数 FID、增强实验 FID、代表实验 D/G loss；
- G 强化：FID、多样性 LPIPS、代表实验 D/G loss；
- 深度调优：G 端方法 FID、D 端方法 FID、代表实验 G adversarial / Wavelet loss；
- CLIP：λ_clip-FID、λ_clip-CLIP MMD²、代表实验 D loss；
- 部署：BN 融合延迟、三引擎 FP32/FP16 Batch=32 延迟与吞吐、TensorRT Top-3 算子；
- 量化：PTQ FID/模糊率/吞吐、逐层恢复 FID/延迟、混合策略 FID/延迟、QAT FID/模糊率/P99；
- 服务：并发-P99、并发-RPS、并发-GPU 显存、并发-RSS、并发-SM、soak 阶段 P99/RPS、soak 头尾 GPU/RSS。

## 文件

- 全实验图表索引.html：按实验顺序逐张查看；
- 全实验指标汇总.csv：训练调优和 CLIP 指标索引；
- 全实验数据清单.csv：results 下原始数据文件清单；
- figure_manifest.json：每张图的来源、指标、单位、行数和限制；
- ../../tools/build_simple_figures.js：Node.js 核心库重生成脚本。

## 口径限制

1. 前期调优、G 强化、深度调优、CLIP 的 FID 是历史 legacy 项目协议；部署阶段 Standard FID 不与它们直接混合排名。
2. 历史训练结果中的 Wavelet/SN 是训练实验方法；当前部署图的算子证据仍以 ConvTranspose、BatchNorm、ReLU、Tanh 为准。
3. 05B 的全局模糊率与全局高频指标不支持 literal hair/eyeliner ROI 显著优于 PTQ 的结论；相关图表只呈现可核实的全局指标。
4. 06D 的 128 并发是已测上界，不是理论物理崩溃点；无失败或 OOM 时不能绘制“已证实崩溃点”。
5. 训练曲线最多等间隔抽样 240 个点，仅为可读性处理，不改变原始 CSV。

生成命令：

node tools/build_simple_figures.js
`, 'utf8');
console.log(`[figures] generated ${manifest.length} single-series figures in ${OUT}`);
