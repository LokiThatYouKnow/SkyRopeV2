<#
  run_dpo_after_sft.ps1 —— 等 SFT 跑完自动接 DPO（独立进程，不受会话影响）
  日志: out/pipeline_dpo.log
#>
$ErrorActionPreference = 'Continue'
$root = 'D:\SkyRope\SkyRopeV2'
$sftLog = Join-Path $root 'out\sft24k_run2.log'
$pipeLog = Join-Path $root 'out\pipeline_dpo.log'
$dpoLog = Join-Path $root 'out\dpo24k.log'

function Log($m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m"
  Add-Content -Path $pipeLog -Value $line -Encoding utf8
}

Log '=== 等待 SFT 完成 (step 113215/113215) ==='
$deadline = (Get-Date).AddHours(10)
$done = $false
while ((Get-Date) -lt $deadline) {
  $tail = Get-Content $sftLog -Encoding utf8 -Tail 60 -ErrorAction SilentlyContinue
  if ($tail -match '\(113215/113215\)') { $done = $true; break }
  $proc = Get-Process python -ErrorAction SilentlyContinue
  if (-not $proc) { Log 'SFT 进程已不在（可能异常退出），继续检查日志标记'; Start-Sleep -Seconds 30
    $tail = Get-Content $sftLog -Encoding utf8 -Tail 60 -ErrorAction SilentlyContinue
    if ($tail -match '\(113215/113215\)') { $done = $true }
    break }
  Start-Sleep -Seconds 60
}
if (-not $done) { Log '未检测到 SFT 完成标记，放弃自动接 DPO'; exit 1 }
Log 'SFT 完成标记已出现，等待最终权重落盘 ...'

$w = Join-Path $root 'out\full_sft_768_moe.pth'
for ($i = 0; $i -lt 20; $i++) {
  $f = Get-Item $w -ErrorAction SilentlyContinue
  if ($f -and $f.LastWriteTime -gt (Get-Date).AddMinutes(-15)) { Log "SFT 权重已落盘: $($f.LastWriteTime)"; break }
  Start-Sleep -Seconds 30
}
$f = Get-Item $w -ErrorAction SilentlyContinue
if (-not $f) { Log '找不到 full_sft_768_moe.pth，放弃'; exit 1 }

# 换词表/MoE 形状的对齐参数（下游脚本靠环境变量读取）
$env:SKYROPE_TOKENIZER = '../model_tok24k'
$env:SKYROPE_VOCAB = '24000'
$env:SKYROPE_NUM_EXPERTS = '4'
$env:SKYROPE_TOP_K = '2'
$env:SKYROPE_MOE_INTERMEDIATE = '1216'
$env:SKYROPE_MOE_IMPL = 'permute'
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'

$dpoArgs = @('-X','utf8','-u','train_dpo.py',
  '--model_type','llm','--use_moe','1',
  '--data_path','../dataset/dpo.jsonl','--from_weight','full_sft','--from_resume','0',
  '--epochs','1','--batch_size','2','--accumulation_steps','4','--max_seq_len','1024',
  '--dpo_beta','0.1','--learning_rate','5e-7',
  '--save_weight','dpo','--save_interval','500','--log_interval','50','--num_workers','2')
Log '启动 DPO ...'
$p = Start-Process -FilePath 'python' -ArgumentList $dpoArgs -WorkingDirectory (Join-Path $root 'trainer') -RedirectStandardOutput $dpoLog -RedirectStandardError "$dpoLog.err" -WindowStyle Hidden -PassThru
Log "DPO 已启动 pid=$($p.Id)  log=$dpoLog"

Start-Sleep -Seconds 900
if ($p.HasExited) {
  Log "DPO 提前退出 exit=$($p.ExitCode)（约 15 分钟内）——日志尾部："
  (Get-Content $dpoLog -Encoding utf8 -Tail 25 -ErrorAction SilentlyContinue) | ForEach-Object { Log "  $_" }
  (Get-Content "$dpoLog.err" -Encoding utf8 -Tail 15 -ErrorAction SilentlyContinue) | ForEach-Object { Log "  ERR $_" }
} else {
  Log 'DPO 运行正常，日志尾部：'
  (Get-Content $dpoLog -Encoding utf8 -Tail 6 -ErrorAction SilentlyContinue) | ForEach-Object { Log "  $_" }
}
Log '=== 驱动脚本结束（DPO 继续在后台跑）==='
