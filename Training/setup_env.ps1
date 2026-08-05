# Sets the three env vars that the training and eval scripts need for each new
# PowerShell session. Dot-source it: `. .\setup_env.ps1`
#
# Expects:
#   - A file containing your Tinker API key. By default we look for `tinkerkey.md`
#     in the parent folder of this script (the repo root). Override with:
#         $env:TINKER_KEY_FILE = "C:\path\to\your\tinkerkey.md"
#     before dot-sourcing.
#   - `wandb login` has already been run once (writes to $env:USERPROFILE\_netrc).
#
# NEVER commit your tinkerkey.md — the top-level .gitignore excludes it.

if ($env:TINKER_KEY_FILE) {
    $keyFile = $env:TINKER_KEY_FILE
} else {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $keyFile = Join-Path (Split-Path -Parent $scriptDir) "tinkerkey.md"
}

if (-not (Test-Path $keyFile)) {
    Write-Host "ERROR: Tinker key file not found at $keyFile" -ForegroundColor Red
    Write-Host "Set `$env:TINKER_KEY_FILE to override, or place tinkerkey.md next to the repo root." -ForegroundColor Red
    return
}

$env:TINKER_API_KEY = (Get-Content -Path $keyFile -Raw).Trim()
$env:WANDB_API_KEY = (Select-String -Path "$env:USERPROFILE\_netrc" -Pattern "password" | Select-Object -First 1).Line.Trim().Split()[-1]
$env:PYTHONUTF8 = "1"

Write-Host "Env vars set:" -ForegroundColor Green
Write-Host "  TINKER_API_KEY (length $($env:TINKER_API_KEY.Length))"
Write-Host "  WANDB_API_KEY  (length $($env:WANDB_API_KEY.Length))"
Write-Host "  PYTHONUTF8     = $env:PYTHONUTF8"
