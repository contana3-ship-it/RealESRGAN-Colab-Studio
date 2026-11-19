param(
    [string]$Repo = "RealESRGAN-Colab-Studio",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

try {
    Set-Location -LiteralPath $PSScriptRoot

    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        throw "Chua cai Git. Chay: winget install --id Git.Git -e"
    }
    if (-not (Get-Command gh.exe -ErrorAction SilentlyContinue)) {
        throw "Chua cai GitHub CLI. Chay: winget install --id GitHub.cli -e"
    }

    if (-not (Test-Path -LiteralPath "colab/start.sh")) {
        throw "Khong tim thay colab/start.sh. Hay dat file PS1 trong thu muc goc du an."
    }
    if (-not (Test-Path -LiteralPath "app/server.py")) {
        throw "Khong tim thay app/server.py. Hay giai nen day du du an."
    }

    Write-Host "=== REAL-ESRGAN COLAB STUDIO / GITHUB PUBLISH ===" -ForegroundColor Cyan

    # PowerShell 5.1 may treat stderr from native tools as a terminating error.
    # cmd.exe hides the expected error from 'gh auth status' or 'gh api' checks.
    & cmd.exe /d /c 'gh auth status >NUL 2>NUL'
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Dang nhap GitHub..." -ForegroundColor Yellow
        & gh.exe auth login --hostname github.com --git-protocol https --web
        if ($LASTEXITCODE -ne 0) { throw "Dang nhap GitHub that bai." }
    }

    $owner = (& gh.exe api user --jq .login | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($owner)) {
        throw "Khong lay duoc ten tai khoan GitHub."
    }
    if ($Repo -notmatch '^[A-Za-z0-9_.-]+$') {
        throw "Ten repository khong hop le: $Repo"
    }

    $full = "$owner/$Repo"
    $url = "https://github.com/$full.git"
    Write-Host "Tai khoan : $owner"
    Write-Host "Repository: $full"

    if (-not (Test-Path -LiteralPath ".git")) {
        & git.exe init
        if ($LASTEXITCODE -ne 0) { throw "git init that bai." }
    }

    # Configure author only in this local repository when needed.
    $gitName = (& git.exe config --local user.name | Out-String).Trim()
    if (-not $gitName) { & git.exe config --local user.name $owner }
    $gitEmail = (& git.exe config --local user.email | Out-String).Trim()
    if (-not $gitEmail) {
        & git.exe config --local user.email "$owner@users.noreply.github.com"
    }

    & git.exe add -A
    if ($LASTEXITCODE -ne 0) { throw "git add that bai." }

    # A nonzero exit code from --quiet means there are staged changes.
    & git.exe diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        & git.exe commit -m "Publish RealESRGAN Colab Studio"
        if ($LASTEXITCODE -ne 0) { throw "git commit that bai." }
    }

    & git.exe branch -M main
    if ($LASTEXITCODE -ne 0) { throw "Khong doi duoc branch sang main." }

    # Check repo existence without causing PowerShell to stop on expected 404.
    # Using only a validated owner/repo avoids shell injection.
    & cmd.exe /d /c "gh api repos/$full --silent >NUL 2>NUL"
    $repoExists = ($LASTEXITCODE -eq 0)

    if (-not $repoExists) {
        Write-Host "Dang tao repository $full ..." -ForegroundColor Cyan
        & gh.exe repo create $full "--$Visibility"
        if ($LASTEXITCODE -ne 0) {
            throw "Khong tao duoc repository. Kiem tra dang nhap va quyen tao repo."
        }
    } else {
        Write-Host "Repository da ton tai, se cap nhat." -ForegroundColor Green
    }

    # 'git remote get-url origin' prints an error when origin is absent.
    # Inspect the remote list instead, so PowerShell 5.1 never receives that error.
    $remotes = @(& git.exe remote)
    if ($LASTEXITCODE -ne 0) { throw "Khong doc duoc danh sach git remote." }
    if ($remotes -contains 'origin') {
        & git.exe remote set-url origin $url
    } else {
        & git.exe remote add origin $url
    }
    if ($LASTEXITCODE -ne 0) { throw "Khong cau hinh duoc remote origin." }

    Write-Host "Dang push ma nguon..." -ForegroundColor Cyan
    & git.exe push -u origin main
    if ($LASTEXITCODE -ne 0) {
        throw "Push that bai. Neu GitHub da co commit, can kiem tra lich su branch truoc khi merge. Script khong force push."
    }

    Write-Host ""
    Write-Host "HOAN THANH!" -ForegroundColor Green
    Write-Host "Repo : https://github.com/$full"
    Write-Host "Colab: https://colab.research.google.com/github/$full/blob/main/colab/RealESRGAN_Studio.ipynb"
    Write-Host ""
    Write-Host "Mo notebook Colab trong lien ket phia tren va Run All."
    Write-Host "Notebook se mount Google Drive, chon video va tu phuc hoi x4."
    Write-Host "Khong can Cloudflare Tunnel."
}
catch {
    Write-Host ""
    Write-Host "[LOI] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
