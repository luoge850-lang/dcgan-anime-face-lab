param(
    [string]$TargetPath = 'C:\Users\32875\OneDrive\Desktop\DCGAN部署优化实验报告.docx'
)

$ErrorActionPreference = 'Stop'
$tmp = Join-Path $env:TEMP ('dcgan_report_ooxml_' + [guid]::NewGuid().ToString('N'))
$deploymentScriptRoot = Split-Path -Parent $PSScriptRoot
$labRoot = Split-Path $deploymentScriptRoot -Parent
$resultsRoot = Join-Path $labRoot 'results\Deployment_Optimization_Results'
$task1ManifestPath = Join-Path $resultsRoot '01_ONNX_Fusion\task1_manifest.json'
$task2ManifestPath = Join-Path $resultsRoot '02_Engine_Benchmark\02E_Report\task2_manifest.json'
$task3ManifestPath = Join-Path $resultsRoot '03_Quantization\03E_Report\task3_manifest.json'
$overallManifestPath = Join-Path $resultsRoot 'deployment_optimization_manifest.json'

function Read-JsonSafe([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json) } catch { return $null }
}

function Manifest-Status($Manifest) {
    if ($null -eq $Manifest) { return 'missing' }
    if ($null -eq $Manifest.status) { return 'unknown' }
    return [string]$Manifest.status
}

$task1Manifest = Read-JsonSafe $task1ManifestPath
$task2Manifest = Read-JsonSafe $task2ManifestPath
$task3Manifest = Read-JsonSafe $task3ManifestPath
$overallManifest = Read-JsonSafe $overallManifestPath
$task1Status = Manifest-Status $task1Manifest
$task2Status = Manifest-Status $task2Manifest
$task3Status = Manifest-Status $task3Manifest
$overallStatus = Manifest-Status $overallManifest
$task2ComparisonRows = @()
$task2Top3Rows = @()
$comparisonPath = Join-Path $resultsRoot '02_Engine_Benchmark\02E_Report\task2_engine_comparison.csv'
$top3Path = Join-Path $resultsRoot '02_Engine_Benchmark\02E_Report\task2_top3_operators.csv'
if (Test-Path -LiteralPath $comparisonPath) { $task2ComparisonRows = @(Import-Csv -LiteralPath $comparisonPath) }
if (Test-Path -LiteralPath $top3Path) { $task2Top3Rows = @(Import-Csv -LiteralPath $top3Path) }
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp '_rels') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp 'word') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp 'word\_rels') | Out-Null

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function E([string]$Text) {
    if ($null -eq $Text) { return '' }
    return [System.Security.SecurityElement]::Escape([string]$Text)
}

function Run([string]$Text, [int]$Size = 21, [string]$Color = '1F2937', [bool]$Bold = $false, [bool]$Italic = $false) {
    $b = if ($Bold) { '<w:b/>' } else { '' }
    $i = if ($Italic) { '<w:i/>' } else { '' }
    return '<w:r><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="' + $Size + '"/><w:szCs w:val="' + $Size + '"/><w:color w:val="' + $Color + '"/>' + $b + $i + '</w:rPr><w:t xml:space="preserve">' + (E $Text) + '</w:t></w:r>'
}

function P([string]$Text, [string]$Style = 'Normal', [int]$Before = 0, [int]$After = 120, [int]$Size = 21, [string]$Color = '1F2937', [bool]$Bold = $false, [string]$Align = 'left') {
    $jc = '<w:jc w:val="' + $Align + '"/>'
    $pr = '<w:pPr><w:pStyle w:val="' + $Style + '"/><w:spacing w:before="' + ($Before * 20) + '" w:after="' + ($After * 20) + '" w:line="276" w:lineRule="auto"/>' + $jc + '</w:pPr>'
    return '<w:p>' + $pr + (Run $Text $Size $Color $Bold $false) + '</w:p>'
}

function RichP([object[]]$Parts, [string]$Style = 'Normal', [int]$Before = 0, [int]$After = 120, [string]$Align = 'left') {
    $runs = ''
    foreach ($part in $Parts) {
        $runs += Run $part.Text $part.Size $part.Color $part.Bold $part.Italic
    }
    return '<w:p><w:pPr><w:pStyle w:val="' + $Style + '"/><w:spacing w:before="' + ($Before * 20) + '" w:after="' + ($After * 20) + '" w:line="276" w:lineRule="auto"/><w:jc w:val="' + $Align + '"/></w:pPr>' + $runs + '</w:p>'
}

function Bullet([string]$Text) {
    return '<w:p><w:pPr><w:pStyle w:val="Normal"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr><w:spacing w:after="80" w:line="276" w:lineRule="auto"/></w:pPr>' + (Run $Text 21 '1F2937' $false $false) + '</w:p>'
}

function Cell([string]$Text, [int]$Width, [bool]$Header = $false) {
    $fill = if ($Header) { 'E8EEF5' } else { 'FFFFFF' }
    $run = Run $Text 18 '1F2937' $Header $false
    return '<w:tc><w:tcPr><w:tcW w:w="' + $Width + '" w:type="dxa"/><w:shd w:fill="' + $fill + '"/><w:tcMar><w:top w:w="90" w:type="dxa"/><w:bottom w:w="90" w:type="dxa"/><w:start w:w="120" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tcMar><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>' + $run + '</w:p></w:tc>'
}

function Table([string[]]$Headers, [object[]]$Rows, [int[]]$Widths) {
    $grid = ''; foreach ($w in $Widths) { $grid += '<w:gridCol w:w="' + $w + '"/>' }
    $borders = '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="B7C3D0"/><w:left w:val="single" w:sz="4" w:color="B7C3D0"/><w:bottom w:val="single" w:sz="4" w:color="B7C3D0"/><w:right w:val="single" w:sz="4" w:color="B7C3D0"/><w:insideH w:val="single" w:sz="4" w:color="D8E0E8"/><w:insideV w:val="single" w:sz="4" w:color="D8E0E8"/></w:tblBorders>'
    $tbl = '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/><w:tblLayout w:type="fixed"/>' + $borders + '</w:tblPr><w:tblGrid>' + $grid + '</w:tblGrid>'
    $tbl += '<w:tr><w:trPr><w:tblHeader/></w:trPr>'
    for ($i = 0; $i -lt $Headers.Count; $i++) { $tbl += Cell $Headers[$i] $Widths[$i] $true }
    $tbl += '</w:tr>'
    foreach ($row in $Rows) {
        $tbl += '<w:tr>'
        for ($i = 0; $i -lt $Headers.Count; $i++) { $tbl += Cell ([string]$row[$i]) $Widths[$i] $false }
        $tbl += '</w:tr>'
    }
    return $tbl + '</w:tbl>'
}

function Caption([string]$Text) { return P $Text 'Caption' 80 80 18 '64748B' $false 'left' }
function PageBreak() { return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>' }

$body = ''
$body += RichP @(@{Text='DCGAN 部署优化实验阶段报告';Size=36;Color='0B2545';Bold=$true;Italic=$false}) 'Title' 0 80 'center'
$body += RichP @(@{Text='Exp11 Generator · ONNX 图导出、算子融合与多引擎推理性能评估';Size=23;Color='475569';Bold=$false;Italic=$false}) 'Subtitle' 0 240 'center'
$body += Table @('项目','内容') @(
    @('模型基线','Exp11 Generator EMA 权重；输出尺寸 64×64；latent 维度 128'),
    @('实验范围','任务一：ONNX 导出与算子融合；任务二：ORT / TensorRT / OpenVINO 部署 benchmark'),
    @('测试环境','由各阶段 manifest 和 CSV 记录实际运行环境；未记录的环境不会在本报告中猜测'),
    @('报告状态',"Task1=$task1Status；Task2=$task2Status；Task3=$task3Status；Overall=$overallStatus")
) @(1700,7660)
$body += P '摘要' 'Heading1' 300 120 31 '2E74B5' $true
$body += P '本阶段围绕 Exp11 Generator 完成三条部署实验链：任务一验证 ONNX 导出、计算图捕获、标准 ONNX 算子替换探针和 ConvTranspose-BatchNorm 融合；任务二比较 ONNX Runtime、TensorRT 和 OpenVINO 的 FP32/FP16 动态 Batch 性能；任务三使用 TensorRT ModelOpt INT8 PTQ，并在同一 latent_eval 上比较 FID、模糊率、LPIPS 和频域误差。' 'Normal' 0 120 21 '1F2937' $false 'left'
$body += P '总评：任务一至任务三的脚本链已补齐，但只有在实际运行并生成对应 CSV、Trace、engine 和质量指标后，才能把最终报告标记为 complete。当前报告生成器不应预填未验证的速度或质量结论。' 'Callout' 120 180 21 '1F3A5F' $true 'left'

$body += P '1. 实验目标与设计' 'Heading1' 240 120 31 '2E74B5' $true
$body += P '本阶段不重新训练 GAN，也不改变 Exp11 Generator 的权重。所有部署实验均以 01A 导出的 generator_fp32_raw.onnx 作为主比较图，确保不同引擎面对相同的网络结构和参数。任务一关注计算图合法性与融合；任务二关注部署后的推理效率。' 'Normal' 0 120 21 '1F2937' $false 'left'
$body += P '统一控制变量' 'Heading2' 160 80 26 '2E74B5' $true
$body += Bullet '主比较图：01A raw ONNX；不把 01B/01C 处理图混入三引擎主表。'
$body += Bullet '精度：FP32 与 FP16。'
$body += Bullet '动态 Batch：默认 1、4、8、16、32。'
$body += Bullet 'Benchmark：warmup=20，正式迭代=100，并记录均值、p50、p95 和吞吐量。'
$body += Bullet 'Profiler：预热阶段在正式 profiling 之外，避免把 CUDA 初始化误判为模型算子。'

$body += P '2. 任务一：ONNX 导出与算子融合' 'Heading1' 240 120 31 '2E74B5' $true
$body += P '2.1 01A 原始 ONNX 导出' 'Heading2' 160 80 26 '2E74B5' $true
$body += P "01A 的实际状态：$task1Status。运行后应以 01A_Export/onnx_check.json 为准，确认 generator.onnx、onnx.checker、FX 图捕获和标准 ONNX 替换探针；本报告不预填未验证的 checker 结论。" 'Normal' 0 100 21 '1F2937' $false 'left'
$body += Caption '表 1  原始 ONNX 图结构'
$body += Table @('算子','数量','说明') @(
    @('ConvTranspose','5','Generator 的主要上采样计算'),
    @('BatchNormalization','4','四个中间特征归一化层'),
    @('Relu','4','中间激活函数'),
    @('Tanh','1','输出范围映射')
) @(2500,1200,5660)

$body += P '2.2 01B ORT 自动优化' 'Heading2' 160 80 26 '2E74B5' $true
$body += P '01B/01C 的融合速度、节点数和数值等价必须从 fusion_check.json、manual_bn_fusion_check.json 及其 CSV 读取。速度未达门槛仍是有效测量结果，但不能在运行前写死为加速或减速。' 'Normal' 0 100 21 '1F2937' $false 'left'
$body += Caption '表 2  ORT 自动优化结果'
$body += Table @('版本','节点数','Batch 1 均值','结论') @(
    @('Raw ONNX','见 fusion_check.json','见 fusion_latency_summary.csv','以实际运行记录为准'),
    @('ORT optimized','见 fusion_check.json','见 fusion_latency_summary.csv','以实际运行记录为准')
) @(2200,1400,1900,3860)

$body += P '2.3 01C 手动 BatchNorm Folding' 'Heading2' 160 80 26 '2E74B5' $true
$body += P '01C 将 ConvTranspose 后接 BatchNorm 的结构折叠到卷积权重和偏置中。节点数、最大绝对误差和速度变化以 manual_bn_fusion_check.json 与 numerical_equivalence.csv 为准。' 'Normal' 0 100 21 '1F2937' $false 'left'
$body += Caption '表 3  手动融合的数值与速度结果'
$body += Table @('指标','结果','判断') @(
    @('节点数量','见 manual_bn_fusion_check.json','以实际图统计为准'),
    @('最大绝对误差','见 numerical_equivalence.csv','必须由运行结果确认'),
    @('Batch 延迟','见 manual_bn_latency_summary.csv','逐 batch 对比 raw/fused'),
    @('预设门槛','至少两个 Batch 提升 5%','由脚本自动判定')
) @(2600,3000,3760)
$body += P '任务一结论：01A 现在同时输出 FX 图、标准 ONNX 替换探针和 checker 证据；01B/01C 输出融合前后节点级耗时与数值等价。当前 Exp11 Generator 主图没有 wavelet 或动态 SN 推理节点，因此自定义算子部分应按“替换探针通过/主图不适用”记录，不能伪称模型已包含自定义算子。' 'Callout' 120 180 21 '1F3A5F' $true 'left'

$body += P '3. 任务二：三引擎部署与性能比较' 'Heading1' 240 120 31 '2E74B5' $true
$body += P '任务二使用同一个 01A raw ONNX，在 ONNX Runtime CPU、OpenVINO CPU 和 TensorRT GPU 上分别测试 FP32/FP16 与动态 Batch。三引擎的输入图一致，但硬件不同，因此结果应理解为“引擎与硬件组合的端到端性能”，不能简单写成纯软件算法倍数比较。' 'Normal' 0 100 21 '1F2937' $false 'left'
$body += Caption '表 4  三引擎关键性能（端到端均值）'
$benchmarkTableRows = @()
foreach ($row in $task2ComparisonRows) {
    $benchmarkTableRows += ,@(
        ("{0} / {1}" -f $row.engine, $row.device),
        $row.precision,
        ("B{0}: {1} ms" -f $row.batch, $row.end_to_end_mean_ms),
        ("{0} img/s" -f $row.throughput_images_per_s),
        ("{0} MB" -f $row.memory_mb)
    )
}
if ($benchmarkTableRows.Count -eq 0) {
    $benchmarkTableRows = @(,@('未运行','-','-','-','-'))
}
$body += Table @('引擎/设备','精度','Batch/端到端均值','吞吐','显存/RSS') $benchmarkTableRows @(2500,1200,2500,1500,1660)
$body += P "任务二实际状态：$task2Status。上表只读取 task2_engine_comparison.csv；没有 CSV 时显示未运行，不使用旧的固定 benchmark 数值。显存字段必须按各引擎 manifest 中的来源说明解释。" 'Normal' 0 120 21 '1F2937' $false 'left'

$body += P '3.1 Torch Profiler 算子瓶颈' 'Heading2' 160 80 26 '2E74B5' $true
$body += Caption '表 5  模型级 Top-3 瓶颈'
$top3TableRows = @()
foreach ($row in ($task2Top3Rows | Select-Object -First 12)) {
    $top3TableRows += ,@($row.rank, $row.operator_or_layer, $row.evidence_file, '见 task2_report.md 的图层优化方案')
}
if ($top3TableRows.Count -eq 0) {
    $top3TableRows = @(,@('-','未运行','-','运行 02E 后生成'))
}
$body += Table @('排名','算子/层','证据','优化方向') $top3TableRows @(700,2600,3000,3060)
$body += P 'Profiler 解释：dgrad_engine 是 ConvTranspose 的子 kernel，不能与父级 ConvTranspose 再相加；Exp11_Generator_Inference、cudaDeviceSynchronize、模块加载等属于外层或运行时开销，不应列为模型 Top-3。' 'Normal' 0 120 21 '475569' $false 'left'

$body += P '3.2 显存结果的边界' 'Heading2' 160 80 26 '2E74B5' $true
$body += P '新版 02B 已记录 cuda_peak_used_snapshot_mb 和 cuda_engine_delta_mb。它们是 GPU 设备级显存快照，包含同一 GPU 上的其他分配，不等同于 TensorRT 单独拥有的精确显存。报告中应保留字段并标注“设备级快照”，不能把它写成纯 engine 显存。' 'Normal' 0 120 21 '1F2937' $false 'left'

$body += P '4. 导师要求完成度审计' 'Heading1' 240 120 31 '2E74B5' $true
$body += Caption '表 6  两项任务完成度'
$body += Table @('要求','当前证据','状态','客观说明') @(
    @('导出 generator.onnx','01A_Export/onnx_check.json','由 manifest 判定',$task1Status),
    @('自定义算子替换与捕获','FX、custom_operator_probe.csv','由 01A 运行判定','主图是否含 custom domain 以 checker 记录为准'),
    @('算子融合','01B/01C manifest 与 latency CSV','由 manifest 判定','结构、数值和速度分开审计'),
    @('三引擎部署','02A/02B/02C benchmark/profile','由 manifest 判定','缺任一后端 profile 则 incomplete'),
    @('FP32/FP16 动态 Batch','task2_engine_comparison.csv','由 manifest 判定','默认覆盖 1/4/8/16/32'),
    @('Chrome Trace 与 Top-3','02D/02F trace 与 02E top3 CSV','由 manifest 判定','只接受保存的 profile 证据'),
    @('量化质量基线','03D metrics/error/frequency CSV','由 manifest 判定','FID/模糊率/LPIPS 与频域误差'),
    @('总体状态','deployment_optimization_manifest.json',$overallStatus,'必须三项任务均有真实运行证据')
) @(2400,2500,1300,3160)

$body += P '4. 任务三：量化破坏程度与 QAT 基线' 'Heading1' 240 120 31 '2E74B5' $true
$body += P '任务三不是重新训练 GAN，而是比较 FP32、FP16 和 INT8 PTQ 在速度与生成质量之间的权衡。三种精度必须使用同一组固定 latent noise，避免随机性影响结论。' 'Normal' 0 100 21 '1F2937' $false 'left'
$body += P '实验顺序' 'Heading2' 160 80 26 '2E74B5' $true
$body += Bullet '固定 5,000 个 latent，使用 FP32 raw ONNX 生成基准结果。'
$body += Bullet '使用现有 TensorRT FP16 engine 生成同一批 latent 的结果。'
$body += Bullet '用固定 latent 校准 INT8 PTQ，生成 INT8 engine。'
$body += Bullet '对 FP32/FP16/INT8 重新计算 FID、模糊率、LPIPS、多样性和边缘密度。'
$body += Bullet '分析 INT8 是否导致高频细节截断、色块、模糊或面部结构破坏。'
$body += Bullet '使用 frequency_band_error.csv 比较 LL/LH/HL/HH；只有高频误差明显高于 LL 且质量指标同步恶化时，才判定高频截断证据成立。'
$body += Bullet '03E 输出唯一的 task3_manifest.json、指标表、量化误差、频域诊断和 QAT 建议。'
$body += Bullet "03E 的实际状态：$task3Status；只有读取到 03A/03B/03C/03D 的真实 manifest 后才会标记 complete。"
$body += P '校准集注意事项：真实动漫头像不能直接作为 Generator 的输入，因为 Generator 接收的是 latent z。100 张真实图片应作为 FID/模糊率/分布参考；INT8 激活校准应使用 100 或 512 个固定 latent。报告中应明确区分“评价参考集”和“模型输入校准集”。' 'Callout' 120 180 21 '7A5A00' $true 'left'

$body += P '6. 需要掌握的知识' 'Heading1' 240 120 31 '2E74B5' $true
$body += Bullet 'ONNX 计算图、opset、动态 Batch 和标准算子域。'
$body += Bullet 'ConvTranspose 的尺寸变化、计算量与 GPU kernel 特性。'
$body += Bullet 'BatchNorm Folding 的数学原理及数值误差验证。'
$body += Bullet 'ORT、TensorRT、OpenVINO 的后端优化差异。'
$body += Bullet 'warmup、CUDA 同步、p50/p95、吞吐量和显存快照。'
$body += Bullet 'FP16/INT8、scale、zero-point、PTQ calibration 与 QAT。'
$body += Bullet 'FID、LPIPS、模糊率、边缘密度和生成质量损失分析。'

$body += PageBreak
$body += P '附录 A：文件与结果索引' 'Heading1' 240 120 31 '2E74B5' $true
$body += Table @('阶段','主要文件','用途') @(
    @('01A','01A_Export/generator.onnx、generator_fp32_raw.onnx；onnx_check.json','原始 ONNX、FX、替换探针与 checker 证据'),
    @('01B','01B_ORT_Optimize/fusion_check.json；fusion_latency_summary.csv','ORT 自动优化与算子级耗时'),
    @('01C','01C_BN_Fold/manual_bn_fusion_check.json；numerical_equivalence.csv','手动 BN Folding 与数值等价'),
    @('02A','02A_ORT/ort_benchmark.csv；profiles/ort_profile_*.json','ORT CPU FP32/FP16 benchmark'),
    @('02B','02B_TensorRT/tensorrt_benchmark.csv；tensorrt_layer_profile.csv','TensorRT GPU FP32/FP16 benchmark与显存快照'),
    @('02C','02C_OpenVINO/openvino_benchmark.csv；openvino_operator_profile.csv','OpenVINO CPU FP32/FP16 benchmark'),
    @('02D/02F','02D_Torch_Reference/torch_trace.json；02F_Layer_Profile/layer_profiler_trace.json','Chrome Trace 与算子/层瓶颈'),
    @('02E','02E_Report/task2_report.md；task2_engine_comparison.csv；task2_manifest.json','任务二汇总报告与对比表'),
    @('03A-03D','03A_Protocol；03B_FP32_FP16；03C_INT8_PTQ；03D_Evaluation','校准集、FP16/INT8 engine 和质量评估'),
    @('03E','03E_Report/task3_quantization_report.md；task3_manifest.json；deployment_optimization_manifest.json','任务三最终报告、频域诊断、QAT 基线和总审计状态')
) @(1200,4300,3860)
$body += P '附录 B：报告生成说明' 'Heading2' 160 80 26 '2E74B5' $true
$body += P '本报告依据本地保存的实验 CSV、JSON 和 Chrome Trace 整理。所有数值应以对应结果文件为准；任务二的主比较图是 01A raw ONNX。显存数值采用设备级 CUDA 快照，不能解释为 TensorRT 单一 engine 的精确占用。' 'Normal' 0 120 21 '1F2937' $false 'left'

$document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>' + $body + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/><w:headerReference w:type="default" r:id="rId2"/><w:footerReference w:type="default" r:id="rId3"/></w:sectPr></w:body></w:document>'

$styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:before="0" w:after="160"/><w:jc w:val="center"/></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:pPr><w:spacing w:before="0" w:after="240"/><w:jc w:val="center"/></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="0"/><w:spacing w:before="320" w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="31"/><w:b/><w:color w:val="2E74B5"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="26"/><w:b/><w:color w:val="2E74B5"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="80" w:after="80"/></w:pPr><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="18"/><w:color w:val="64748B"/><w:i/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Callout"><w:name w:val="Callout"/><w:basedOn w:val="Normal"/><w:pPr><w:shd w:fill="F4F6F9"/><w:ind w:left="180" w:right="180"/><w:spacing w:before="120" w:after="180"/></w:pPr><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/><w:color w:val="1F3A5F"/></w:rPr></w:style></w:styles>'

$numbering = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="360"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum><w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="360"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr><w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="0"/></w:num></w:numbering>'

$settings = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:updateFields w:val="true"/><w:compat/></w:settings>'
$header = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="left"/><w:spacing w:after="80"/></w:pPr>' + (Run 'DCGAN 部署优化实验报告 · Exp11 Generator' 18 '64748B' $false $false) + '</w:p></w:hdr>'
$footer = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="right"/></w:pPr>' + (Run 'Deployment Optimization Report' 16 '94A3B8' $false $false) + '</w:p></w:ftr>'
$rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
$docRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/><Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/></Relationships>'
$types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/><Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/><Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'

Write-Utf8NoBom (Join-Path $tmp '[Content_Types].xml') $types
Write-Utf8NoBom (Join-Path $tmp '_rels\.rels') $rels
Write-Utf8NoBom (Join-Path $tmp 'word\document.xml') $document
Write-Utf8NoBom (Join-Path $tmp 'word\styles.xml') $styles
Write-Utf8NoBom (Join-Path $tmp 'word\numbering.xml') $numbering
Write-Utf8NoBom (Join-Path $tmp 'word\settings.xml') $settings
Write-Utf8NoBom (Join-Path $tmp 'word\header1.xml') $header
Write-Utf8NoBom (Join-Path $tmp 'word\footer1.xml') $footer
Write-Utf8NoBom (Join-Path $tmp 'word\_rels\document.xml.rels') $docRels

Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $TargetPath) { Remove-Item -LiteralPath $TargetPath -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($tmp, $TargetPath)
Remove-Item -LiteralPath $tmp -Recurse -Force
Write-Output "[created] $TargetPath"
