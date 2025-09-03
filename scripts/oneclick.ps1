param(
  [string]$Branch = $(git rev-parse --abbrev-ref HEAD 2>$null),
  [switch]$SkipPull  # use -SkipPull se não quiser fazer pull --rebase
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Exec($cmd, $args) {
  Write-Host ">> $cmd $args" -ForegroundColor Cyan
  & $cmd $args
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd $args (exit $LASTEXITCODE)" }
}

# 0) garantir que estamos num repo git
$repo = git rev-parse --show-toplevel 2>$null
if (-not $repo) { throw "Este diretório não é um repositório Git." }
Set-Location $repo

# 1) escolher python do venv (fallback para 'python')
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# 2) rodar testes com plugins desativados
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
Write-Host "`n==> Rodando testes..." -ForegroundColor Yellow
& $py -m pytest -q
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Testes falharam (exit $code). Abortando." }
Write-Host "✓ Testes passaram." -ForegroundColor Green

# 3) commitar mudanças (se houver)
$dirty = git status --porcelain
if ($dirty) {
  Write-Host "`n==> Commit das mudanças..." -ForegroundColor Yellow
  git add -A
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  git commit -m "chore(ci): one-click — tests passing @ $ts"
} else {
  Write-Host "Nenhuma mudança local para commitar." -ForegroundColor DarkGray
}

# 4) sincronizar com remoto
if (-not $SkipPull) {
  Write-Host "`n==> Sincronizando com remoto (fetch + rebase $Branch)..." -ForegroundColor Yellow
  Exec git "fetch --all --prune"
  Exec git "pull --rebase origin $Branch"
} else {
  Write-Host "Pulando pull --rebase por opção (-SkipPull)." -ForegroundColor DarkGray
}

# 5) push
Write-Host "`n==> Fazendo push..." -ForegroundColor Yellow
Exec git "push origin HEAD"

# 6) resumo
Write-Host "`n==> Status final" -ForegroundColor Yellow
git status -sb
Write-Host "`nTudo certo. 🎉" -ForegroundColor Green
