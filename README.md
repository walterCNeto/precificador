# derivx

![CI](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml/badge.svg)

Precificador geral de derivativos por blocos de construção:
- **Modelo sob Q** (GBM multiativo com correlação, r(t), q_i(t), sigma_i(t))
- **Numerário** piecewise-flat (desconto determinístico)
- **Payoffs diversos** (PF: terminal, média, máx/mín, basket, barreira, etc.)
- **Exercício** Europeu, Bermuda, Americano (LSMC)
- **DSL** declarativa (dict/JSON) para montar produto e precificar

# Digitais (Haug)
{"engine":"analytic","model":{"r":0.05,"q":[0.0],"sigma":[0.2],"corr":[[1.0]]},
 "grid":{"T":1.0,"steps":64},"S0":[100.0],
 "product":{"style":"european","type":"cash_or_nothing_call","asset":0,"K":100.0,"cash":1.0}}

# Gap (Haug)
{"engine":"analytic","model":{"r":0.05,"q":[0.0],"sigma":[0.2],"corr":[[1.0]]},
 "grid":{"T":1.0},"S0":[100.0],
 "product":{"style":"european","type":"gap_call","asset":0,"K1":95.0,"K2":100.0}}

# Margrabe (exchange)
{"engine":"analytic",
 "model":{"r":0.05,"q":[0.01,0.03],"sigma":[0.20,0.30],"corr":[[1.0,0.5],[0.5,1.0]]},
 "grid":{"T":1.0},"S0":[100.0,120.0],
 "product":{"style":"european","type":"exchange_call","asset_long":0,"asset_short":1}}

## Instalação (dev)
```bash
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\Scripts\activate)
pip install -e .[dev]