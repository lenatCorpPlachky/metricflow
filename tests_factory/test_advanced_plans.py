"""Advanced business metric plans use MetricFlow's SQL AST and renderer."""

from __future__ import annotations

import logging

import duckdb
from metricflow_semantics.sql.sql_exprs import SqlColumnReferenceExpression
from metricflow_semantics.sql.sql_table import SqlTable

from metricflow.engine.factory_advanced_plans import FactoryAdvancedPlanCompiler
from metricflow.sql.render.duckdb_renderer import DuckDbSqlPlanRenderer
from metricflow.sql.sql_plan import SqlPlan, SqlSelectColumn
from metricflow.sql.sql_select_node import SqlSelectStatementNode
from metricflow.sql.sql_table_node import SqlTableNode

logger = logging.getLogger(__name__)


def _base(columns: tuple[str, ...]) -> SqlSelectStatementNode:
    return SqlSelectStatementNode.create(
        description="MetricFlow source metrics",
        select_columns=tuple(SqlSelectColumn(
            SqlColumnReferenceExpression.from_column_reference("facts", column), column)
            for column in columns),
        from_source=SqlTableNode.create(SqlTable(None, "facts")),
        from_source_alias="facts",
    )


def test_page_weekly_share_uses_project_week_denominator() -> None:
    """Key-event shares partition by project and ISO week, including NULL zero totals."""
    columns = ("project__project_id", "page__page_url", "page__segment",
               "metric_time__week", "page_clicks", "page_impressions", "page_ctr",
               "page_avg_position", "organic_sessions", "key_events")
    source = _base(columns)
    plan = FactoryAdvancedPlanCompiler().compile("mart_page_weekly", source)
    sql = DuckDbSqlPlanRenderer().render_sql_plan(SqlPlan(plan)).sql
    with duckdb.connect() as con:
        con.execute("CREATE TABLE facts AS SELECT * FROM (VALUES "
                    "('alpha','/a','All','2026-10-05',2,100,.02,3.8,10,3),"
                    "('alpha','/b','All','2026-10-05',2,100,.02,3.8,10,2),"
                    "('bravo','/a','All','2026-10-05',2,100,.02,3.8,10,10),"
                    "('zero','/a','All','2026-10-05',2,100,.02,3.8,10,0)) "
                    "t(project__project_id,page__page_url,page__segment,metric_time__week,"
                    "page_clicks,page_impressions,page_ctr,page_avg_position,organic_sessions,key_events)")
        rows = con.execute(sql).fetchall()
    by_key = {(row[0], row[1]): row for row in rows}
    assert by_key["alpha", "/a"][-1] == .6
    assert by_key["alpha", "/b"][-1] == .4
    assert by_key["bravo", "/a"][-1] == 1
    assert by_key["zero", "/a"][-1] is None
    assert "fct_" not in sql  # This isolated AST test has no warehouse source.


def test_ctr_curve_bucket_edges_and_minimum_sample_count() -> None:
    """Bucket thresholds include 5.0/10.4 and suppress medians below 30 samples."""
    columns = ("project__project_id", "page__page_url", "search_page_query_daily__query",
               "query_page_clicks", "query_page_impressions", "query_page_ctr",
               "query_page_avg_position")
    plan = FactoryAdvancedPlanCompiler().compile("mart_ctr_curve", _base(columns))
    sql = DuckDbSqlPlanRenderer().render_sql_plan(SqlPlan(plan)).sql
    with duckdb.connect() as con:
        con.execute("CREATE TABLE facts (project__project_id VARCHAR, page__page_url VARCHAR, "
                    "search_page_query_daily__query VARCHAR, query_page_clicks DOUBLE, "
                    "query_page_impressions DOUBLE, query_page_ctr DOUBLE, "
                    "query_page_avg_position DOUBLE)")
        rows = [("alpha", f"/a{i}", "term", 1, 10, .1, 5.0) for i in range(29)]
        rows += [("alpha", f"/b{i}", "term", 2, 10, .2, 10.4) for i in range(30)]
        rows += [("bravo", f"/b{i}", "term", 8, 10, .8, 10.4) for i in range(30)]
        con.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        results = con.execute(sql).fetchall()
    by_key = {(row[0], row[1]): row for row in results}
    assert by_key["alpha", "4-5"] == ("alpha", "4-5", None, 29)
    assert by_key["alpha", "6-10"] == ("alpha", "6-10", .2, 30)
    assert by_key["bravo", "6-10"] == ("bravo", "6-10", .8, 30)


def test_striking_distance_population_sd_and_primary_url_tie() -> None:
    """Position SD is population SD and equal-impression URLs tie lexically."""
    columns = ("project__project_id", "page__page_url", "search_page_query_daily__query",
               "metric_time", "query_page_clicks", "query_page_impressions",
               "query_page_weighted_position", "query_page_avg_position")
    plan = FactoryAdvancedPlanCompiler().compile("mart_striking_distance", _base(columns))
    sql = DuckDbSqlPlanRenderer().render_sql_plan(SqlPlan(plan)).sql
    with duckdb.connect() as con:
        con.execute("CREATE TABLE facts (project__project_id VARCHAR, page__page_url VARCHAR, "
                    "search_page_query_daily__query VARCHAR, metric_time DATE, "
                    "query_page_clicks DOUBLE, query_page_impressions DOUBLE, "
                    "query_page_weighted_position DOUBLE, query_page_avg_position DOUBLE)")
        con.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            ("alpha", "/a", "term", "2026-10-01", 2, 100, 400, 4),
            ("alpha", "/a", "term", "2026-10-02", 1, 100, 600, 6),
            ("alpha", "/b", "term", "2026-10-01", 3, 200, 1000, 5),
            ("bravo", "/z", "term", "2026-10-01", 4, 300, 900, 3),
        ])
        results = con.execute(sql).fetchall()
    by_key = {(row[0], row[1]): row for row in results}
    assert by_key["alpha", "/a"] == ("alpha", "/a", "term", 200, 3, 5, 1, True)
    assert by_key["alpha", "/b"][-1] is False
    assert by_key["bravo", "/z"][-1] is True


def test_cannibalization_two_urls_zero_denominator_and_project_isolation() -> None:
    """Cannibalization needs two URLs and denominators remain project-isolated."""
    columns = ("project__project_id", "page__page_url", "search_page_query_daily__query",
               "query_page_impressions")
    plan = FactoryAdvancedPlanCompiler().compile("mart_cannibalization", _base(columns))
    sql = DuckDbSqlPlanRenderer().render_sql_plan(SqlPlan(plan)).sql
    with duckdb.connect() as con:
        con.execute("CREATE TABLE facts (project__project_id VARCHAR, page__page_url VARCHAR, "
                    "search_page_query_daily__query VARCHAR, query_page_impressions DOUBLE)")
        con.executemany("INSERT INTO facts VALUES (?, ?, ?, ?)", [
            ("alpha", "/a", "term", 60), ("alpha", "/b", "term", 40),
            ("alpha", "/only", "unique", 10), ("bravo", "/a", "term", 200),
            ("alpha", "/zero-a", "zero", 0), ("alpha", "/zero-b", "zero", 0),
        ])
        results = con.execute(sql).fetchall()
    by_key = {(row[0], row[1], row[2]): row for row in results}
    assert by_key["alpha", "term", "/a"] == ("alpha", "term", "/a", .6, 60)
    assert by_key["alpha", "term", "/b"] == ("alpha", "term", "/b", .4, 40)
    assert by_key["alpha", "zero", "/zero-a"][3] is None
    assert all(key[0] != "bravo" and key[1] != "unique" for key in by_key)
