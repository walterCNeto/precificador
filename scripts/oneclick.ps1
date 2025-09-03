param(
  [string]$Branch,          # se nao passar, detecta atual
  [switch]$SkipPull         # use -SkipPull para nao fazer pull --rebase
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Exec {
  param(
    [string]$Cmd,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
  )
  if (-not $Cmd) { throw "Exec: parametro Cmd vazio." }
  if (-not $Args -or $Args.Count -eq 0) {
    throw ("Exec: nenhum argumento passado para '" + $Cmd + "'.")
  }
  Write-Host (">> " + $Cmd + " " + ($Args -join " ")) -ForegroundColor Cyan
  & $Cmd @Args
  if ($LASTEXITCODE -ne 0) {
    throw ("Command failed: " + $Cmd + " " + ($Args -join " ") + " (exit " + $LASTEXITCODE + ")")
  }
}

# 0) garantir repositorio Git e posicionar no root
$repo = git rev-parse --show-toplevel 2>$null
if (-not $repo) { throw "Este diretorio nao e um repositorio Git." }
Set-Location $repo

# 0.1) branch atual (se nao veio por parametro)
if (-not $Branch -or $Branch.Trim() -eq "") {
  $Branch = (git rev-parse --abbrev-ref HEAD).Trim()
}

# 1) escolher python do venv (fallback para 'python')
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# 2) rodar testes (desativa plugins externos do pytest)
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
Write-Host ""
Write-Host "==> Rodando testes..." -ForegroundColor Yellow
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw ("Testes falharam (exit " + $LASTEXITCODE + "). Abortando.") }
Write-Host "Testes passaram." -ForegroundColor Green

# 3) commitar mudancas (se houver)
$dirty = (git status --porcelain)
if ($dirty) {
  Write-Host ""
  Write-Host "==> Commit das mudancas..." -ForegroundColor Yellow
  Exec -Cmd "git" -Args @("add","-A")
  $ts  = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $msg = "chore(ci): one-click - tests passing @ " + $ts
  Exec -Cmd "git" -Args @("commit","-m",$msg)
} else {
  Write-Host "Nenhuma mudanca local para commitar." -ForegroundColor DarkGray
}

# 4) sincronizar com remoto
if (-not $SkipPull.IsPresent) {
  Write-Host ""
  Write-Host ("==> Sincronizando com remoto (fetch + rebase " + $Branch + ")...") -ForegroundColor Yellow
  Exec -Cmd "git" -Args @("fetch","--all","--prune")
  Exec -Cmd "git" -Args @("pull","--rebase","origin",$Branch)
} else {
  Write-Host "Pulando pull --rebase por opcao (-SkipPull)." -ForegroundColor DarkGray
}

# 5) push
Write-Host ""
Write-Host "==> Fazendo push..." -ForegroundColor Yellow
Exec -Cmd "git" -Args @("push","origin","HEAD")

# 6) resumo
Write-Host ""
Write-Host "==> Status final" -ForegroundColor Yellow
git status -sb
Write-Host ""
Write-Host "Tudo certo." -ForegroundColor Green
