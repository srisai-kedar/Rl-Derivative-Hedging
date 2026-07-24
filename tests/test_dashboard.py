import ast
import os

import pytest


class _FakeColumn:
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *args, **kwargs):
        self.parent.calls.append(("column_markdown", args, kwargs))


class _FakeSpinner:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self):
        self.calls = []
        self.session_state = {}
        self.sidebar = self
        self.button_value = False
        self.radio_value = "Overview"
        self.toggle_value = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *args, **kwargs):
        self.calls.append(("markdown", args, kwargs))

    def info(self, *args, **kwargs):
        self.calls.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args, kwargs))

    def success(self, *args, **kwargs):
        self.calls.append(("success", args, kwargs))

    def caption(self, *args, **kwargs):
        self.calls.append(("caption", args, kwargs))

    def metric(self, *args, **kwargs):
        self.calls.append(("metric", args, kwargs))

    def plotly_chart(self, *args, **kwargs):
        self.calls.append(("plotly_chart", args, kwargs))

    def dataframe(self, *args, **kwargs):
        self.calls.append(("dataframe", args, kwargs))

    def download_button(self, *args, **kwargs):
        self.calls.append(("download_button", args, kwargs))

    def radio(self, *args, **kwargs):
        self.calls.append(("radio", args, kwargs))
        return self.radio_value

    def selectbox(self, *args, **kwargs):
        self.calls.append(("selectbox", args, kwargs))
        return kwargs["options"][0]

    def toggle(self, *args, **kwargs):
        self.calls.append(("toggle", args, kwargs))
        return self.toggle_value

    def button(self, *args, **kwargs):
        self.calls.append(("button", args, kwargs))
        return self.button_value

    def slider(self, *args, **kwargs):
        self.calls.append(("slider", args, kwargs))
        return kwargs["value"]

    def columns(self, spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeColumn(self) for _ in range(count)]

    def spinner(self, *args, **kwargs):
        self.calls.append(("spinner", args, kwargs))
        return _FakeSpinner()

    def rerun(self):
        self.calls.append(("rerun", (), {}))


def test_css_file_exists():
    """style.css must exist and be non-empty."""
    css_path = os.path.join("src", "dashboard", "style.css")
    assert os.path.exists(css_path), f"CSS file missing at {css_path}"
    with open(css_path, encoding="utf-8") as file:
        content = file.read()
    assert len(content) > 100, "CSS file is suspiciously short"


def test_css_contains_required_classes():
    """Required CSS classes must be present."""
    css_path = os.path.join("src", "dashboard", "style.css")
    with open(css_path, encoding="utf-8") as file:
        content = file.read()
    required_classes = [
        ".metric-card",
        ".metric-label",
        ".metric-rl-value",
        ".metric-delta-positive",
        ".metric-delta-negative",
        ".empty-state",
    ]
    for css_class in required_classes:
        assert css_class in content, f"CSS class '{css_class}' missing from style.css"


def test_discover_result_dirs_empty_for_nonexistent_base(tmp_path):
    """Returns empty list when base_dir does not exist."""
    from src.dashboard.app import discover_result_dirs

    result = discover_result_dirs(str(tmp_path / "nonexistent"))
    assert result == []


def test_discover_result_dirs_finds_valid_run(mock_results_dir):
    """Discovers directories that have both episode_data.csv and metrics.json."""
    from src.dashboard.app import discover_result_dirs

    discover_result_dirs.clear()
    results = discover_result_dirs(str(mock_results_dir / "results"))
    assert len(results) == 1
    assert "test_run_20240101_120000" in results[0]


def test_discover_result_dirs_ignores_incomplete_runs(tmp_path):
    """Directories missing either file are not returned."""
    from src.dashboard.app import discover_result_dirs

    incomplete = tmp_path / "results" / "incomplete_run"
    incomplete.mkdir(parents=True)
    (incomplete / "metrics.json").write_text("{}", encoding="utf-8")

    discover_result_dirs.clear()
    results = discover_result_dirs(str(tmp_path / "results"))
    assert len(results) == 0


def test_load_results_cached_returns_none_for_bad_path():
    """Returns (None, None) gracefully for a nonexistent directory."""
    from src.dashboard.app import load_results_cached

    load_results_cached.clear()
    results, metrics = load_results_cached("/nonexistent/path/to/results")
    assert results is None
    assert metrics is None


def test_metric_cards_html_no_crash(backtest_results_fixture):
    """render_metric_cards must not raise when called with valid metrics."""
    from streamlit.testing.v1 import AppTest

    def app(results):
        from src.dashboard.components.metric_cards import render_metric_cards
        from src.evaluation.metrics import compare_metrics

        metrics = compare_metrics(
            results,
            primary="bs_delta",
            baseline="zero_hedge",
        )
        render_metric_cards(metrics, primary="bs_delta", baseline="zero_hedge")

    at = AppTest.from_function(app, args=(backtest_results_fixture,))
    at.run()
    assert not at.exception, f"render_metric_cards raised: {at.exception}"


def test_episode_replay_component_no_crash(backtest_results_fixture):
    """render_episode_replay_page must not raise on valid BacktestResults."""
    from streamlit.testing.v1 import AppTest

    def app(results):
        from src.dashboard.components.episode_replay import render_episode_replay_page

        render_episode_replay_page(results)

    at = AppTest.from_function(app, args=(backtest_results_fixture,))
    at.run()
    assert not at.exception, f"render_episode_replay_page raised: {at.exception}"


def test_experiment_table_no_crash_no_results(tmp_path):
    """render_experiment_table must not crash when no results exist."""
    from streamlit.testing.v1 import AppTest

    def app(base_dir):
        from src.dashboard.components.experiment_table import render_experiment_table

        render_experiment_table(base_dir=base_dir)

    at = AppTest.from_function(app, args=(str(tmp_path / "empty_results"),))
    at.run()
    assert not at.exception, f"render_experiment_table raised: {at.exception}"


def test_experiment_table_with_results(mock_results_dir):
    """render_experiment_table shows data when results exist."""
    from streamlit.testing.v1 import AppTest

    def app(base_dir):
        from src.dashboard.components.experiment_table import render_experiment_table

        render_experiment_table(base_dir=base_dir)

    at = AppTest.from_function(app, args=(str(mock_results_dir / "results"),))
    at.run()
    assert not at.exception, f"render_experiment_table raised: {at.exception}"


def test_pnl_section_no_crash(backtest_results_fixture):
    """render_pnl_section must not raise on valid data."""
    from streamlit.testing.v1 import AppTest

    def app(results):
        from src.dashboard.components.pnl_chart import render_pnl_section
        from src.evaluation.metrics import compare_metrics

        metrics = compare_metrics(
            results,
            primary="bs_delta",
            baseline="zero_hedge",
        )
        agents = ["bs_delta", "zero_hedge"]
        render_pnl_section(results, agents, metrics)

    at = AppTest.from_function(app, args=(backtest_results_fixture,))
    at.run()
    assert not at.exception, f"render_pnl_section raised: {at.exception}"


def test_app_imports_without_error():
    """The main app module file must exist and be valid Python."""
    app_path = os.path.join("src", "dashboard", "app.py")
    assert os.path.exists(app_path), "app.py not found"
    with open(app_path, encoding="utf-8") as file:
        source = file.read()
    try:
        ast.parse(source)
    except SyntaxError as error:
        pytest.fail(f"app.py has a syntax error: {error}")


def test_filter_results_keeps_correct_agents(backtest_results_fixture):
    """_filter_results returns only the specified agent types."""
    from src.dashboard.components.pnl_chart import _filter_results

    filtered = _filter_results(backtest_results_fixture, agents=["bs_delta"])
    assert set(filtered.episode_df["agent_type"].unique()) == {"bs_delta"}
    assert set(filtered.step_df["agent_type"].unique()) == {"bs_delta"}


def test_collect_rows_returns_empty_for_empty_dir(tmp_path):
    """_collect_rows returns empty list when no metrics.json files exist."""
    from src.dashboard.components.experiment_table import _collect_rows

    (tmp_path / "results").mkdir()
    rows = _collect_rows(str(tmp_path / "results"))
    assert rows == []


def test_sidebar_with_result_dir(monkeypatch, mock_results_dir):
    """Sidebar returns selected result directory and display settings."""
    import src.dashboard.app as dashboard_app

    fake_st = _FakeStreamlit()
    fake_st.toggle_value = True
    run_dir = str(mock_results_dir / "results" / "test_run_20240101_120000")

    monkeypatch.setattr(dashboard_app, "st", fake_st)
    monkeypatch.setattr(dashboard_app, "discover_result_dirs", lambda _base: [run_dir])

    page, selected_dir, show_zero = dashboard_app._render_sidebar()

    assert page == "Overview"
    assert selected_dir == run_dir
    assert show_zero is True


def test_sidebar_without_results(monkeypatch):
    """Sidebar handles no discovered result directories."""
    import src.dashboard.app as dashboard_app

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(dashboard_app, "st", fake_st)
    monkeypatch.setattr(dashboard_app, "discover_result_dirs", lambda _base: [])

    page, selected_dir, show_zero = dashboard_app._render_sidebar()

    assert page == "Overview"
    assert selected_dir == ""
    assert show_zero is False
    assert any(call[0] == "info" for call in fake_st.calls)


def test_quick_eval_stores_session_state(monkeypatch, backtest_results_fixture):
    """Quick eval stores transient results and metrics in session_state."""
    import src.dashboard.app as dashboard_app
    import src.evaluation.backtest as backtest

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(dashboard_app, "st", fake_st)
    monkeypatch.setattr(backtest, "run_backtest", lambda *args, **kwargs: backtest_results_fixture)

    dashboard_app._run_quick_eval_into_session_state()

    assert fake_st.session_state["quick_eval_results"] is backtest_results_fixture
    assert "quick_eval_metrics" in fake_st.session_state
    assert any(call[0] == "rerun" for call in fake_st.calls)


def test_main_routes_all_pages(monkeypatch, backtest_results_fixture):
    """main routes sidebar selections to the matching page renderers."""
    import src.dashboard.app as dashboard_app

    routed = []
    metrics = {"bs_delta": {"std_pnl": 1.0}, "zero_hedge": {"std_pnl": 2.0}}

    monkeypatch.setattr(
        dashboard_app,
        "load_results_cached",
        lambda _selected_dir: (backtest_results_fixture, metrics),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_overview_page",
        lambda results, metrics, show_zero: routed.append(("overview", show_zero)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_replay_page",
        lambda results: routed.append(("replay", results is backtest_results_fixture)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_experiments_page",
        lambda: routed.append(("experiments", True)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_training_logs_page",
        lambda selected_dir: routed.append(("logs", selected_dir)),
    )

    for page in ["Overview", "Episode Replay", "Experiments", "Training Logs"]:
        monkeypatch.setattr(dashboard_app, "_render_sidebar", lambda page=page: (page, "run", True))
        dashboard_app.main()

    assert routed == [
        ("overview", True),
        ("replay", True),
        ("experiments", True),
        ("logs", "run"),
    ]


def test_main_uses_quick_eval_when_no_selected_dir(monkeypatch, backtest_results_fixture):
    """main falls back to session_state quick-eval data."""
    import src.dashboard.app as dashboard_app

    fake_st = _FakeStreamlit()
    fake_st.session_state["quick_eval_results"] = backtest_results_fixture
    fake_st.session_state["quick_eval_metrics"] = {"bs_delta": {}}
    routed = []

    monkeypatch.setattr(dashboard_app, "st", fake_st)
    monkeypatch.setattr(dashboard_app, "_render_sidebar", lambda: ("Overview", "", False))
    monkeypatch.setattr(
        dashboard_app,
        "render_overview_page",
        lambda results, metrics, show_zero: routed.append((results, metrics, show_zero)),
    )

    dashboard_app.main()

    assert routed == [(backtest_results_fixture, {"bs_delta": {}}, False)]


def test_overview_page_empty_and_loaded(monkeypatch, backtest_results_fixture):
    """Overview renders empty state and delegates loaded sections."""
    import src.dashboard.app as dashboard_app
    import src.dashboard.components.metric_cards as metric_cards
    import src.dashboard.components.pnl_chart as pnl_chart

    fake_st = _FakeStreamlit()
    delegated = []
    metrics = {"bs_delta": {}, "zero_hedge": {}}

    monkeypatch.setattr(dashboard_app, "st", fake_st)
    monkeypatch.setattr(metric_cards, "render_metric_cards", lambda m: delegated.append(("cards", m)))
    monkeypatch.setattr(
        pnl_chart,
        "render_pnl_section",
        lambda results, agents, m: delegated.append(("pnl", tuple(agents))),
    )

    dashboard_app.render_overview_page(None, None, False)
    dashboard_app.render_overview_page(backtest_results_fixture, metrics, False)

    assert any(call[0] == "markdown" for call in fake_st.calls)
    assert ("cards", metrics) in delegated
    assert ("pnl", ("bs_delta",)) in delegated


def test_replay_and_experiments_delegate(monkeypatch, backtest_results_fixture):
    """Replay and experiments pages delegate to component renderers."""
    import src.dashboard.app as dashboard_app
    import src.dashboard.components.episode_replay as episode_replay
    import src.dashboard.components.experiment_table as experiment_table

    fake_st = _FakeStreamlit()
    delegated = []

    monkeypatch.setattr(dashboard_app, "st", fake_st)
    monkeypatch.setattr(
        episode_replay,
        "render_episode_replay_page",
        lambda results: delegated.append(("replay", results is backtest_results_fixture)),
    )
    monkeypatch.setattr(
        experiment_table,
        "render_experiment_table",
        lambda base_dir="results": delegated.append(("experiments", base_dir)),
    )

    dashboard_app.render_replay_page(None)
    dashboard_app.render_replay_page(backtest_results_fixture)
    dashboard_app.render_experiments_page()

    assert ("replay", True) in delegated
    assert ("experiments", "results") in delegated


def test_training_logs_empty_missing_and_monitor(monkeypatch, tmp_path):
    """Training logs page handles empty, missing, and valid Monitor CSV states."""
    import src.dashboard.app as dashboard_app

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(dashboard_app, "st", fake_st)

    dashboard_app.render_training_logs_page("")

    run_dir = tmp_path / "results" / "run"
    monitor_dir = tmp_path / "results" / "monitor"
    run_dir.mkdir(parents=True)
    dashboard_app.render_training_logs_page(str(run_dir))

    monitor_dir.mkdir()
    (monitor_dir / "worker.monitor.csv").write_text(
        '#{"t_start": 0}\nr,l,t\n1.0,30,0.1\n2.0,30,0.2\n',
        encoding="utf-8",
    )
    dashboard_app.render_training_logs_page(str(run_dir))

    assert any(call[0] == "plotly_chart" for call in fake_st.calls)


def test_component_branches_with_fake_streamlit(monkeypatch, backtest_results_fixture):
    """Exercise component branches not reached by AppTest smoke tests."""
    import pandas as pd

    from src.evaluation.backtest import BacktestResults
    import src.dashboard.components.episode_replay as episode_replay
    import src.dashboard.components.experiment_table as experiment_table

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(episode_replay, "st", fake_st)
    monkeypatch.setattr(experiment_table, "st", fake_st)

    empty = BacktestResults(
        episode_df=backtest_results_fixture.episode_df.iloc[0:0].copy(),
        step_df=backtest_results_fixture.step_df.iloc[0:0].copy(),
    )
    episode_replay.render_episode_replay_page(empty)

    rl_episode = backtest_results_fixture.episode_df.iloc[[0]].copy()
    rl_episode["agent_type"] = "rl_agent"
    rl_step = backtest_results_fixture.step_df[
        backtest_results_fixture.step_df["episode_id"] == 0
    ].copy()
    rl_step = rl_step[rl_step["agent_type"] == "bs_delta"].copy()
    rl_step["agent_type"] = "rl_agent"
    rl_results = BacktestResults(
        episode_df=pd.concat([backtest_results_fixture.episode_df, rl_episode], ignore_index=True),
        step_df=pd.concat([backtest_results_fixture.step_df, rl_step], ignore_index=True),
    )
    episode_replay.render_episode_replay_page(rl_results)

    df = pd.DataFrame(
        [
            {"Run": "a", "RL P&L Std": 2.0, "BS P&L Std": 1.0, "RL Mean Cost": 0.2, "Improvement %": -1.0},
            {"Run": "b", "RL P&L Std": 1.0, "BS P&L Std": 1.2, "RL Mean Cost": 0.1, "Improvement %": 5.0},
        ]
    )
    styled = experiment_table._style_dataframe(df)

    assert styled is not None
    assert any(call[0] == "warning" for call in fake_st.calls)
