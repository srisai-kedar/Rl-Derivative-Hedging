import numpy as np
import pandas as pd
import plotly.graph_objects as go
from types import SimpleNamespace

from src.evaluation.backtest import BSDeltaPolicy, BacktestResults, ZeroHedgePolicy, load_results, run_backtest, save_results
from src.evaluation.metrics import compare_metrics, compute_all_metrics
from src.evaluation.plots import (plot_cost_sensitivity, plot_episode_replay, plot_hedge_ratio_over_time, plot_metric_comparison, plot_pnl_distribution, plot_robustness_heatmap)


def _backtest(fast_eval_config, n_episodes):
    return run_backtest([BSDeltaPolicy(), ZeroHedgePolicy()], fast_eval_config, n_episodes, show_progress=False)


def test_backtest_returns_backtest_results(fast_eval_config):
    assert isinstance(_backtest(fast_eval_config, 2), BacktestResults)


def test_episode_df_required_columns(backtest_results_fixture):
    assert {"episode_id", "agent_type", "seed", "terminal_pnl", "total_cost"} <= set(backtest_results_fixture.episode_df.columns)


def test_step_df_required_columns(backtest_results_fixture):
    assert {"episode_id", "agent_type", "step", "price", "hedge_ratio", "bs_delta", "step_pnl", "step_cost"} <= set(backtest_results_fixture.step_df.columns)


def test_correct_episode_count(backtest_results_fixture, n_eval_episodes):
    counts = backtest_results_fixture.episode_df.groupby("agent_type").size()
    assert (counts == n_eval_episodes).all()


def test_shared_seeds_same_price_paths(fast_eval_config):
    results = _backtest(fast_eval_config, 10)
    for episode_id in range(10):
        bs = results.step_df[(results.step_df.agent_type == "bs_delta") & (results.step_df.episode_id == episode_id)].price.to_numpy()
        zero = results.step_df[(results.step_df.agent_type == "zero_hedge") & (results.step_df.episode_id == episode_id)].price.to_numpy()
        np.testing.assert_array_almost_equal(bs, zero, decimal=5)


def test_bs_baseline_lower_std_than_zero_hedge(backtest_results_fixture):
    assert compute_all_metrics(backtest_results_fixture, "bs_delta")["std_pnl"] < compute_all_metrics(backtest_results_fixture, "zero_hedge")["std_pnl"]


def test_backtest_deterministic(fast_eval_config):
    first, second = _backtest(fast_eval_config, 10), _backtest(fast_eval_config, 10)
    pd.testing.assert_frame_equal(first.episode_df, second.episode_df)


def test_zero_hedge_has_zero_cost(backtest_results_fixture):
    costs = backtest_results_fixture.episode_df.query("agent_type == 'zero_hedge'").total_cost
    assert (costs == 0.0).all()


def test_bs_delta_hedge_ratio_tracks_bs_delta(backtest_results_fixture):
    data = backtest_results_fixture.step_df.query("agent_type == 'bs_delta'")
    np.testing.assert_allclose(data.hedge_ratio, data.bs_delta, atol=1e-5)


def test_all_plots_return_figure(backtest_results_fixture):
    robustness = pd.DataFrame([
        {"kappa": kappa, "sigma": sigma, "agent_type": agent, "std_pnl": 1.0 + kappa + sigma}
        for kappa in [0.0, 0.001] for sigma in [0.2, 0.3] for agent in ["bs_delta", "zero_hedge"]
    ])
    metrics = compare_metrics(backtest_results_fixture)
    figures = [plot_pnl_distribution(backtest_results_fixture), plot_hedge_ratio_over_time(backtest_results_fixture), plot_metric_comparison(metrics), plot_episode_replay(backtest_results_fixture, 0), plot_cost_sensitivity(robustness), plot_robustness_heatmap(robustness, agent_type="bs_delta")]
    assert all(isinstance(figure, go.Figure) for figure in figures)


def test_results_save_and_load(backtest_results_fixture, tmp_path):
    metrics = compare_metrics(backtest_results_fixture)
    save_results(backtest_results_fixture, metrics, str(tmp_path))
    loaded, loaded_metrics = load_results(str(tmp_path))
    pd.testing.assert_frame_equal(backtest_results_fixture.episode_df, loaded.episode_df)
    assert loaded_metrics == metrics


def test_evaluate_orchestrates_and_saves_outputs(monkeypatch, backtest_results_fixture, fast_eval_config, tmp_path):
    import src.evaluation.evaluate as evaluation

    saved_figures = []
    vec_normalize = SimpleNamespace(close=lambda: None)
    robustness = pd.DataFrame([
        {"kappa": kappa, "sigma": sigma, "agent_type": agent, "std_pnl": 1.0 + kappa + sigma}
        for kappa in [0.0, 0.001] for sigma in [0.2, 0.3] for agent in ["rl_agent", "bs_delta", "zero_hedge"]
    ])
    config = SimpleNamespace(eval_environment=fast_eval_config, results_dir=str(tmp_path), run=SimpleNamespace(seed=0))
    monkeypatch.setattr(evaluation, "load_config", lambda _: config)
    monkeypatch.setattr(evaluation.SAC, "load", lambda _: SimpleNamespace())
    monkeypatch.setattr(evaluation, "load_env_for_inference", lambda *args, **kwargs: vec_normalize)
    monkeypatch.setattr(evaluation, "run_backtest", lambda *args, **kwargs: backtest_results_fixture)
    monkeypatch.setattr(evaluation, "run_robustness_sweep", lambda *args, **kwargs: robustness)
    monkeypatch.setattr(evaluation, "save_figure", lambda fig, directory, name: saved_figures.append(name))

    evaluation.evaluate("model.zip", "vecnorm.pkl", output_dir=str(tmp_path), n_episodes=2, run_robustness=True)

    assert (tmp_path / "episode_data.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "robustness.csv").exists()
    assert {"pnl_distribution", "cost_sensitivity", "robustness_heatmap_rl_agent"} <= set(saved_figures)


def test_evaluate_cli_parses_arguments(monkeypatch):
    import src.evaluation.evaluate as evaluation

    received = {}
    monkeypatch.setattr(evaluation, "evaluate", lambda *args: received.setdefault("args", args))
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--checkpoint", "model.zip", "--vecnorm", "stats.pkl", "--n-episodes", "3", "--skip-robustness"])
    evaluation.main()
    assert received["args"] == ("model.zip", "stats.pkl", "configs/training.yaml", 3, None, False)
