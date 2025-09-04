# derivx

[![CI](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml/badge.svg)](https://github.com/walterCNeto/precificador/actions/workflows/ci.yml)

**Precificador geral de derivativos por blocos de construção**

- **Modelo sob Q** (GBM multiativo com correlação; `r(t)`, `q_i(t)`, `σ_i(t)`)
- **Numerário** *piecewise-flat* (desconto determinístico exato por trechos)
- **Payoffs PF** (funcionais de trajetória): terminal, média, máx/mín, basket, barreira…
- **Exercício** Europeu, Bermudano, Americano (LSMC robusto com *cross‑fit* e desconto correto por janela)
- **PDE 1D** (Crank–Nicolson) para vanillas (inclui Americano via projeção)
- **FFT (Heston)** para europeias (Carr–Madan)
- **Analítico (Haug/Margrabe)**: Digitais, Gap, Exchange
- **Juros (Black‑76)**: ZCB, FRA, Swap, Cap, Floor, Payer/Receiver Swaption (analítico)
- **DSL** declarativa (dict/JSON) para montar produto e precificar
- **Testes** e CI prontos (pytest + GitHub Actions)
- **One‑click** para rodar testes + commit + sync + push (Windows PowerShell)

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

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q
```

**macOS / Linux (bash/zsh)**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

---

## Hello, World (Quick Start)

```python
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
```

---

## Visão de arquitetura

- **Curva (numerário)** — `PiecewiseFlatCurve` implementa \( r(t) \) *piecewise‑flat*, com **desconto exato** por trechos  
  \( DF(t_0,t_1)=\exp\{-\int_{t_0}^{t_1} r(u)\,du\} \).
- **Modelos sob \(\mathbb{Q}\)**  
  `RiskNeutralGBM` (multiativo; `q_i(t)`, `σ_i(t)`, correlação via Cholesky) e **Heston** (europeias) — MC/FFT.
- **Payoffs PF (funcionais de trajetória)** — utilitários vetorizados: terminal, médias, running max/min, barreiras, cestas…
- **Motores numéricos**
  - **MC**: simulação GBM/Heston, antitético e (opcional) *control variate*;
  - **PDE CN 1D**: vanillas (europeias e americanas);
  - **FFT (Heston)**: precifica vários strikes de uma vez;
  - **Analítico**: fórmulas fechadas (digitais, gap, exchange; *Black‑76* em juros).
- **Exercício (LSMC)** — regressão do valor de continuação em *features* (padrão `log S`, base polinomial até grau 2 + cruzados).
- **DSL** — `price_from_spec(spec)` mapeia um dicionário JSON para engine, grade temporal e produto.

---

## Usando a DSL (`price_from_spec`)

### Chaves principais

- **`engine`**: `"mc"` | `"pde"` | `"fft"` | `"analytic"` | `"auto"`  
  *`auto` tenta analítico/PDE/FFT quando aplicável; senão cai para MC.*
- **`model`**
  - GBM: `{"name":"gbm","r":...,"q":...,"sigma":...,"corr":...}`
  - Heston: `{"name":"heston","r":...,"q":...,"kappa":...,"theta":...,"xi":...,"rho":...,"v0":...}`
  - Juros (Black‑76 analítico): `{"r_curve":{"times":[...],"rates":[...]}}`
- **`grid`**
  - MC: `{"T":..., "steps":...}` (uniforme)
  - PDE/FFT: usa `{"T":...}` (demais *knobs* são específicos do engine)
- **`S0`**: lista de preços iniciais (uma entrada por ativo)
- **`product`**
  - `style`: `"european"` | `"bermudan"` | `"american"` (LSMC ou PDE)
  - `type`: ver [Produtos suportados](#produtos-suportados)
  - Parametrização conforme o tipo (`asset`, `K`, `barrier`, `weights`, …)
  - Exercício (bermudan): `exercise_idx`, `exercise_times` **ou** `exercise_every`
- **Parâmetros do motor**
  - MC: `n_paths`, `seed`
  - PDE: `NS`, `NT`, `Smax_mult`
  - FFT: `alpha`, `N`, `eta`

---

## Exemplos práticos

### 1) Europeu (MC) vs Black–Scholes
```python
from derivx import price_from_spec, bs_call_price

S0=K=100.0; r=0.05; q=0.0; sigma=0.2; T=1.0
spec = {
  "engine":"mc",
  "model":{"name":"gbm","r":r,"q":q,"sigma":sigma},
  "grid":{"T":T,"steps":128},
  "S0":[S0],
  "product":{"style":"european","type":"european_call","asset":0,"K":K},
  "n_paths":80_000,"seed":42,
}
pmc,se = price_from_spec(spec)
pbs = bs_call_price(S0,K,r,q,sigma,T)
print(f"MC={pmc:.4f} ± {1.96*se:.4f} | BS={pbs:.4f}")
```

### 2) PDE (put europeu vs americano)
```python
from derivx import price_from_spec

euro = {"engine":"pde","model":{"name":"gbm","r":0.05,"q":0.0,"sigma":0.2},
        "grid":{"T":1.0},"S0":[100.0],
        "product":{"style":"european","type":"european_put","asset":0,"K":100.0},
        "NS":800,"NT":800,"Smax_mult":5.0}

amer = {**euro, "product":{"style":"american","type":"european_put","asset":0,"K":100.0}}

pe,_ = price_from_spec(euro)
pa,_ = price_from_spec(amer)
print(f"PDE euro={pe:.4f}  PDE amer={pa:.4f}  (amer >= euro)")
```

### 3) Barreira *up‑and‑out* (monotonia)
```python
base = {"engine":"mc","model":{"name":"gbm","r":0.05,"q":0.0,"sigma":0.2},
        "grid":{"T":1.0,"steps":128},"S0":[100.0],
        "n_paths":100_000,"seed":7}

van,_   = price_from_spec({**base,"product":{"style":"european","type":"european_call","asset":0,"K":100.0}})
uo130,_ = price_from_spec({**base,"product":{"style":"european","type":"up_and_out_call","asset":0,"K":100.0,"barrier":130.0}})
uo140,_ = price_from_spec({**base,"product":{"style":"european","type":"up_and_out_call","asset":0,"K":100.0,"barrier":140.0}})
print(f"UO130={uo130:.4f} <= UO140={uo140:.4f} <= Vanilla={van:.4f}")
```

### 4) Asiática aritmética (≤ vanilla)
```python
asian,_ = price_from_spec({**base,"product":{"style":"european","type":"asian_arith_call","asset":0,"K":100.0}})
print(f"Asian={asian:.4f} <= Vanilla={van:.4f}")
```

### 5) Bermudana (LSMC): efeito da frequência de exercício
```python
def berm(ex_every):
  spec={"engine":"mc","model":{"name":"gbm","r":0.05,"q":0.0,"sigma":0.2},
        "grid":{"T":1.0,"steps":256},"S0":[100.0],
        "product":{"style":"bermudan","type":"european_put","asset":0,"K":100.0,"exercise_every":ex_every},
        "n_paths":120_000,"seed":7}
  return price_from_spec(spec)

for ex in (8,16,32):
  p,se = berm(ex)
  print(f"ex_every={ex:>2}: {p:.4f} ± {1.96*se:.4f}")
```

### 6) Heston: FFT vs MC
```python
common={"name":"heston","r":0.05,"q":0.0,"kappa":1.5,"theta":0.04,"xi":0.5,"rho":-0.7,"v0":0.04}

fft={"engine":"fft","model":common,"grid":{"T":1.0},"S0":[100.0],
     "product":{"style":"european","type":"european_call","asset":0,"K":100.0},
     "alpha":1.5,"N":4096,"eta":0.25}

mc={"engine":"mc","model":common,"grid":{"T":1.0,"steps":512},"S0":[100.0],
    "product":{"style":"european","type":"european_call","asset":0,"K":100.0},
    "n_paths":200_000,"seed":7}

p_fft,_ = price_from_spec(fft)
p_mc,se = price_from_spec(mc)
print(f"FFT={p_fft:.4f} | MC={p_mc:.4f} ± {1.96*se:.4f}")
```

### 7) Basket 2D (GBM correlacionado)
```python
spec={"engine":"mc",
      "model":{"name":"gbm","r":0.05,"q":[0.01,0.03],"sigma":[0.20,0.30],
               "corr":[[1.0,0.5],[0.5,1.0]]},
      "grid":{"T":2.0,"steps":80},"S0":[100.0,120.0],
      "product":{"style":"european","type":"basket_call","weights":[0.5,0.5],"K":110.0},
      "n_paths":100_000,"seed":3}
p,se = price_from_spec(spec)
print(p, "±", 1.96*se)
```

### 8) Analítico (Haug/Margrabe) — Digitais, Gap, Exchange

**Digitais (Haug) — MC (paga no vencimento)**
```json
{
  "engine":"mc",
  "model":{"name":"gbm","r":0.05,"q":0.0,"sigma":0.2},
  "grid":{"T":1.0,"steps":256},
  "S0":[100.0],
  "product":{"style":"european","type":"cash_or_nothing_call","asset":0,"K":100.0,"cash":1.0},
  "n_paths":200000,"seed":11
}
```

**Digitais (Analítico)**
```json
{
  "engine":"analytic",
  "model":{"r":0.05,"q":[0.0],"sigma":[0.2],"corr":[[1.0]]},
  "grid":{"T":1.0},
  "S0":[100.0],
  "product":{"style":"european","type":"asset_or_nothing_call","asset":0,"K":100.0}
}
```

**Gap (Haug)**
```json
{
  "engine":"analytic",
  "model":{"r":0.05,"q":[0.0],"sigma":[0.2],"corr":[[1.0]]},
  "grid":{"T":1.0},
  "S0":[100.0],
  "product":{"style":"european","type":"gap_call","asset":0,"K1":95.0,"K2":100.0}
}
```

**Margrabe (exchange)**
```json
{
  "engine":"analytic",
  "model":{"r":0.05,"q":[0.01,0.03],"sigma":[0.20,0.30],"corr":[[1.0,0.5],[0.5,1.0]]},
  "grid":{"T":1.0},
  "S0":[100.0,120.0],
  "product":{"style":"european","type":"exchange_call","asset_long":0,"asset_short":1}
}
```

### 9) Juros (Black‑76 — analítico)

**ZCB**
```json
{
  "engine":"analytic",
  "model":{"r_curve":{"times":[0.5,1.0,2.0],"rates":[0.05,0.052,0.055]}},
  "product":{"style":"european","type":"zcb","T":2.0,"notional":1.0}
}
```

**FRA**
```json
{
  "engine":"analytic",
  "model":{"r_curve":{"times":[0.5,1.0,2.0],"rates":[0.05,0.052,0.055]}},
  "product":{"style":"european","type":"fra","T1":1.0,"T2":1.5,"tau":0.5,"K":0.055,"notional":1e6}
}
```

**Swap (PV de *fixed‑for‑float*)**
```json
{
  "engine":"analytic",
  "model":{"r_curve":{"times":[0.5,1.0,1.5,2.0,2.5,3.0],"rates":[0.05,0.052,0.053,0.054,0.055,0.056]}},
  "product":{"style":"european","type":"swap","T0":0.0,
             "payment_times":[0.5,1.0,1.5,2.0,2.5,3.0],"tau":0.5,
             "fixed_rate":0.055,"notional":1e6}
}
```

**Cap / Floor (soma de caplets/floorlets — Black‑76)**  
(*`sigma` aqui é a vol de Black‑76 do forward de cada período; pode ser constante.*)
```json
{
  "engine":"analytic",
  "model":{"r_curve":{"times":[0.5,1.0,1.5,2.0],"rates":[0.05,0.052,0.053,0.055]}},
  "product":{"style":"european","type":"cap",
             "payment_times":[0.5,1.0,1.5,2.0],"tau":0.5,
             "K":0.06,"sigma":0.25,"notional":1e6}
}
```

**Swaption Payer / Receiver (Black‑76 com PVBP como *numéraire*)**
```json
{
  "engine":"analytic",
  "model":{"r_curve":{"times":[0.5,1.0,1.5,2.0,2.5,3.0],"rates":[0.05,0.052,0.053,0.054,0.055,0.056]}},
  "product":{"style":"european","type":"payer_swaption","expiry":1.0,
             "payment_times":[1.5,2.0,2.5,3.0],"tau":0.5,
             "K":0.055,"sigma":0.20,"notional":1e6}
}
```

> Dica: `engine: "auto"` tenta analítico quando suportado; caso contrário, cai para MC/PDE/FFT.

---

## Produtos suportados

✔️ = implementado / OK

**Vanillas (GBM)**  
✔️ `european_call`, `european_put` (MC e PDE)  
✔️ Americano (put/call) via PDE 1D (Crank–Nicolson)  
✔️ Bermudano (LSMC) com `exercise_every`, `exercise_idx` ou `exercise_times`

**Path‑dependentes (GBM)**  
✔️ `asian_arith_call`  
✔️ Barreira: `up_and_out_call` (outros tipos na fila)

**Multi‑ativo (GBM)**  
✔️ `basket_call` (pesos arbitrários)  
✔️ `exchange_call` (Margrabe, 2 ativos)

**Analíticos (Haug/Margrabe)**  
✔️ Digitais: `cash_or_nothing_call|put`, `asset_or_nothing_call|put`  
✔️ Gap: `gap_call`, `gap_put`  
✔️ Exchange (Margrabe): `exchange_call`

**Juros (Black‑76 — analítico)**  
✔️ `zcb`, `fra`, `swap`, `cap`, `floor`, `payer_swaption`, `receiver_swaption`

**Heston**  
✔️ Europeias (FFT Carr–Madan; MC)

**Roadmap curto**  
Digitais/barreiras adicionais (*up/down, in/out, rebates*); Asians put, lookbacks, cliquets; Greeks *pathwise*/LRM; calibração Heston/GBM.

---

## Validação & testes

- **Consistência BS:** MC (GBM) ≈ Black–Scholes (dentro de *k·SE*)  
- **PDE vs BS:** put europeu CN ≈ BS; americano PDE ≥ europeu  
- **Monotonicidades:** *up&out* ≤ vanilla; Bermudano densificando → Americano  
- **Heston:** MC ≈ FFT (tolerância baseada em SE)  
- **Analíticos (Haug/Margrabe/Black‑76):** testes de referência com fórmulas fechadas

Rodar todos os testes (desabilitando plugins externos do PyTest):
```powershell
# Windows
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest -q
```

**One‑click (Windows)**  
```powershell
.\scripts\oneclick.ps1
# Roda testes, faz commit (se houver mudanças), puxa e dá push
```

---

## Dicas de precisão & performance

| Backend | Parâmetro   | Efeito |
|---|---|---|
| MC  | `n_paths` | SE ∝ 1/√N (custo linear) |
| MC  | `steps`   | Reduz viés temporal (path‑dep./LSMC); custo ∝ paths×steps |
| PDE | `NS`, `NT` | Convergência O(Δt + ΔS²); aumente até estabilizar |
| PDE | `Smax_mult` | Domínio \[0, S_max]; comece com 5–7×K |
| FFT | `alpha`   | *Damping* (1–2 típico); extremos podem instabilizar |
| FFT | `N`, `eta` | Resolução em frequência/strike |

**LSMC — boas práticas**  
Regressão apenas em ITM; desconto correto entre janelas (Δt múltiplos); *cross‑fit* A/B; checar monotonia (Bermudano ↑ com mais janelas).

---

## Estrutura do repositório

```
src/derivx/
  __init__.py
  curves.py               # PiecewiseFlatCurve
  models/gbm.py           # RiskNeutralGBM
  engine/montecarlo.py
  engine/pde.py           # Crank–Nicolson 1D (vanillas)
  engine/fft.py           # Heston (Carr–Madan)
  exercise/lsmc.py        # LSMC (com fix de desconto entre janelas)
  payoffs/core.py         # PF utilitários (terminal, média, etc.)
  payoffs/extra.py        # Digitais, Gap, Exchange, etc. (PF)
  analytic/bs.py          # Φ, d1d2, BS
  analytic/haug.py        # Digitais, Gap, Margrabe (closed-form)
  ir/black76.py           # Juros: ZCB, FRA, Swap, Cap/Floor, Swaptions
  dsl/spec.py             # Mapeia dict -> engine/produto
tests/
  ref_formulas/           # testes de referência (BS/Haug/Margrabe/Black-76)
  reference/
scripts/
  oneclick.ps1            # roda testes + commit + sync + push
README.md
```

---

## Contribuindo

1. Crie um branch: `git checkout -b feat/minha-feature`  
2. Adicione testes em `tests/`  
3. Garanta **`python -m pytest -q`** verde  
4. Abra um Pull Request (escopo, resultados esperados, validação)

**Padrões**  
Estilo PEP 8 / tipagem leve; funções puras e vetorização; documente decisões numéricas (PDE BCs, FFT *knobs*, etc.).

---

## Referências

- Black, F.; Scholes, M. (1973) *The Pricing of Options and Corporate Liabilities*
- Longstaff, F.; Schwartz, E. (2001) *Valuing American Options by Simulation*
- Carr, P.; Madan, D. (1999) *Option Valuation Using the FFT*
- Heston, S. (1993) *A Closed-Form Solution for Options with Stochastic Volatility*
- Margrabe, W. (1978) *The Value of an Option to Exchange One Asset for Another*
- Haug, E. G. (2006) *The Complete Guide to Option Pricing Formulas*

---

## Licença

MIT. **Uso acadêmico/educacional**; valide premissas, calibração e risco de modelo antes de produção.
