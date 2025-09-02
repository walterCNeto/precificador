from derivx import price_from_spec

def test_dsl_european_put_runs():
    spec={"engine":"mc",
          "model":{"name":"gbm","r":0.03,"q":0.01,"sigma":0.25},
          "grid":{"T":0.75,"steps":96},
          "S0":[120.0],
          "product":{"style":"european","type":"european_put","asset":0,"K":110.0},
          "n_paths":40_000,"seed":123}
    p, se = price_from_spec(spec)
    assert p > 0 and se > 0

def test_dsl_bermudan_put_runs():
    spec={"engine":"mc",
          "model":{"name":"gbm","r":0.05,"q":0.0,"sigma":0.2},
          "grid":{"T":1.0,"steps":128},
          "S0":[100.0],
          "product":{"style":"bermudan","type":"european_put","asset":0,"K":100.0,"exercise_every":16},
          "n_paths":40_000,"seed":321}
    p, se = price_from_spec(spec)
    assert p > 0 and se > 0
