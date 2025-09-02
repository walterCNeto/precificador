# Validation Report

```

==========================
1) Black–Scholes analítico
==========================
[PASS] BS call derivx ~= ref  | derivx=10.450584 ref=10.450584
[PASS] Put via paridade dentro de faixa ~5.57  | put_paridade=5.573526

======================
2) Curva PiecewiseFlat
======================
[PASS] df(0,1) ~= e^{-0.05}  | df=0.951229425 ref=0.951229425

==============================================
3) MC (GBM) vs BS — tolerância por erro padrão
==============================================
[PASS] MC GBM ~= BS | MC=10.4531 ± 0.1024  BS=10.4506  tol=0.2500

=================================
4) PDE — Euro ≈ BS e Amer >= Euro
=================================
[PASS] PDE Euro put ~= BS put | PDE=5.5425 BS=5.5735
[PASS] PDE American >= Euro | Amer=6.1881 Euro=5.5425

=============================
5) Barrier e Asian — relações
=============================
[PASS] Up&Out <= Vanilla | UO130=3.6421 Vanilla=10.4292
[PASS] Monotonia na barreira (130<=140) | UO130=3.6421 UO140=6.0270
[PASS] Asian aritmética <= Vanilla | Asian=5.7878 Vanilla=10.4292

=====================
6) Heston — MC vs FFT
=====================
[PASS] Heston MC ~= FFT | MC=10.4216 ± 0.0830 FFT=10.4211 tol=0.7000

Resumo: 10/10 PASS

```
