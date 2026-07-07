$flaskPort = 8080
$borePath = "C:\Program Files (x86)\cloudflared\cloudflared.exe"

Write-Host "=== Starting Flask server ===" -ForegroundColor Cyan
$flask = Start-Process -FilePath "pythonw" -ArgumentList "gui.py" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3

Write-Host "=== Starting Cloudflare Tunnel ===" -ForegroundColor Cyan
Write-Host "(waiting for trycloudflare URL...)" -ForegroundColor Yellow
Write-Host ""

$logFile = Join-Path $PSScriptRoot "tunnel.log"
$urlFile = Join-Path $PSScriptRoot "tunnel_url.txt"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $borePath
$psi.Arguments = "tunnel --url http://localhost:$flaskPort --no-autoupdate"
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null

$urlFound = $false
$timeout = 30
$elapsed = 0

while (-not $proc.HasExited) {
    $line = $proc.StandardError.ReadLine()
    if ($line) {
        Add-Content -Path $logFile -Value $line
        if ($line -match 'https://[a-zA-Z0-9-]+\.trycloudflare\.com') {
            $url = $matches[0]
            if (-not $urlFound) {
                $urlFound = $true
                $url | Out-File -FilePath $urlFile -Encoding UTF8
                $host.ui.RawUI.WindowTitle = "MIMII Tunnel: $url"
                Write-Host ""
                Write-Host "============================================" -ForegroundColor Cyan
                Write-Host "  TUNNEL URL: $url" -ForegroundColor Green
                Write-Host "============================================" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "  URL saved to: tunnel_url.txt" -ForegroundColor Yellow
                Write-Host "  Open this URL on your phone" -ForegroundColor Yellow
                Write-Host "  Close this window to stop the tunnel" -ForegroundColor Yellow
                Write-Host ""
            }
        } elseif (-not $urlFound) {
            # Show progress dots for initial connection
            $elapsed++
            if ($elapsed -le 20) {
                Write-Host "." -NoNewline -ForegroundColor DarkYellow
            }
        }
    }

    $outLine = $proc.StandardOutput.ReadLine()
    if ($outLine) {
        Add-Content -Path $logFile -Value $outLine
    }
}

Write-Host ""
Write-Host "Tunnel stopped" -ForegroundColor Red
Stop-Process -Id $flask.Id -Force -ErrorAction SilentlyContinue
