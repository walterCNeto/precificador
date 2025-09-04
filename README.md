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