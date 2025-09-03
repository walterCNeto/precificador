# derivx

![CI](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml/badge.svg)

Precificador geral de derivativos por blocos de construção:
- **Modelo sob Q** (GBM multiativo com correlação, r(t), q_i(t), sigma_i(t))
- **Numerário** piecewise-flat (desconto determinístico)
- **Payoffs diversos** (PF: terminal, média, máx/mín, basket, barreira, etc.)
- **Exercício** Europeu, Bermuda, Americano (LSMC)
- **DSL** declarativa (dict/JSON) para montar produto e precificar

## Instalação (dev)
```bash
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\Scripts\activate)
pip install -e .[dev]