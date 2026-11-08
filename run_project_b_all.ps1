# Run the complete Project B experiment: clustering, anomaly, then edge.
#
# Safe to rerun after an interruption (shutdown, crash): the clustering and
# anomaly runners resume from their checkpoints, and a part whose results
# file already exists is skipped. Logs are appended per session.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$resultsDir = Join-Path $PSScriptRoot 'results\project_b'
New-Item -ItemType Directory -Force $resultsDir | Out-Null

foreach ($part in 'clustering', 'anomaly', 'edge') {
    $log = Join-Path $resultsDir "$part.log"
    if (Test-Path (Join-Path $resultsDir "${part}_results.csv")) {
        "SKIP $part (results already written) $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }
    "START $part $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    & $python -P "run_project_b_$part.py" *>> $log
    $code = $LASTEXITCODE
    "EXIT $part code=$code $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -ne 0) { "ABORTED after $part (exit code $code)"; exit 1 }
}
"ALL DONE $(Get-Date -Format o)"
