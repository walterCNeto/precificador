#derivx

[![CI](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml/badge.svg)](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml)

**Precificador geral de derivativos por blocos de construção**:

- **Modelo sob Q** (GBM multiativo com correlação; `r(t)`, `q_i(t)`, `σ_i(t)`)
- **Numerário** piecewise-flat (desconto determinístico exato por trechos)
- **Payoffs PF** (funcionais de trajetória): terminal, média, máx/mín, basket, barreira…
- **Exercício** Europeu, Bermudano, Americano (LSMC robusto)
- **PDE 1D** (Crank–Nicolson) para vanillas (inclui Americano via projeção)
- **FFT (Heston)** para europeias (Carr–Madan)
- **Analítico (Haug/Margrabe)**: Digitais, Gap, Exchange
- **DSL** declarativa (dict/JSON) para montar produto e precificar
- **Testes** e CI prontos (pytest + GitHub Actions)

---

## Índice

- [Instalação](#instalação)
- [Hello, World (Quick Start)](#hello-world-quick-start)
- [Visão de arquitetura](#visão-de-arquitetura)
- [Usando a DSL (`price_from_spec`)](#usando-a-dsl-price_from_spec)
- [Exemplos práticos](#exemplos-práticos)
- [Produtos suportados](#produtos-suportados)
- [Validação & testes](#validação--testes)
- [Dicas de precisão & performance](#dicas-de-precisão--performance)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Contribuindo](#contribuindo)
- [Referências](#referências)
- [Licença](#licença)

---

## Instalação

> Requisitos: Python 3.11+ (testado). A instalação em modo **dev** já traz dependências de testes.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q

**macOS / Linux (bash/zsh):**
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

## Hello, World (Quick Start)

from derivx import price_from_spec, bs_call_price

spec = {
  "engine": "mc",
  "model": {"name":"gbm","r":0.05,"q":0.0,"sigma":0.2},
  "grid": {"T": 1.0, "steps": 128},
  "S0": [100.0],
  "product": {"style":"european","type":"european_call","asset":0,"K":100.0},
  "n_paths": 80_000, "seed": 42,
}
p, se = price_from_spec(spec)
print("MC:", p, "±", 1.96*se)
print("BS:", bs_call_price(100, 100, r=0.05, q=0.0, sigma=0.2, T=1.0))