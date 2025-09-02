# derivx
![CI](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml/badge.svg)

Precificador genérico de derivativos via *building blocks*:
- **Modelo sob Q** (GBM multiativo com correlação, r(t), q_i(t), sigma_i(t))
- **Numerário** piecewise-flat (desconto determinístico)
- **Payoffs componíveis** (PF: terminal, média, máx/mín, basket, barreira, etc.)
- **Exercício** Europeu ou Bermudano/Americano (LSMC)
- **DSL** declarativa (dict/JSON) para montar produto e precificar


## Instalação (dev)
```bash
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\Scripts\activate)
pip install -e .[dev]