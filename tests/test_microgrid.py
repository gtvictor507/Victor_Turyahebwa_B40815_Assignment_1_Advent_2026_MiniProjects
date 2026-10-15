from pathlib import Path

import numpy as np

from src.microgrid import MicroGrid, generate_demands, get_demand, read_demands


def test_feasible_day_matches_equations():
    grid = MicroGrid()
    result = grid.solve_day(100, 120)
    assert not result['infeasible']
    assert np.allclose(grid.matrix @ [result['solar'], result['battery']], [100, 120])


def test_negative_solution_gets_nnls_flag():
    result = MicroGrid().solve_day(10, 20)
    assert result['infeasible']
    assert result['battery'] >= 0 and result['solar'] >= 0
    assert result['residual'] > 0


def test_csv_and_vectorized_solve(tmp_path):
    path = tmp_path / 'demand.csv'
    generate_demands(str(path))
    values = read_demands(str(path))
    assert values.shape == (30, 2)
    assert np.allclose(MicroGrid().matrix @ MicroGrid().solve_many(values).T, values.T)


def test_prompt_retries_invalid_values():
    inputs = iter(['', 'bad', '-1', '5.5'])
    assert get_demand(lambda _: next(inputs)) == 5.5
