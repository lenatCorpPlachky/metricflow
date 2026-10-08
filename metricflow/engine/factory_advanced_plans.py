"""Typed second-stage SQL plans for grain-aware Factory semantic queries.

Each input is a first-stage query planned from dbt semantic models by the
ordinary MetricFlow dataflow planner. This module extends MetricFlow's native
SqlPlan; the same engine renderer and read-only SqlClient execute the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from metricflow_semantics.sql.sql_exprs import (
    SqlAggregateFunctionExpression,
    SqlCaseExpression,
    SqlColumnReferenceExpression,
    SqlComparison,
    SqlComparisonExpression,
    SqlFunction,
    SqlIntegerExpression,
    SqlLogicalExpression,
    SqlLogicalOperator,
    SqlNullExpression,
    SqlPercentileExpression,
    SqlPercentileExpressionArgument,
    SqlRatioComputationExpression,
    SqlStringLiteralExpression,
    SqlWindowFunction,
    SqlWindowFunctionExpression,
    SqlWindowOrderByArgument,
)

from metricflow.sql.sql_plan import SqlPlanNode, SqlSelectColumn
from metricflow.sql.sql_select_node import SqlSelectStatementNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvancedQueryDefinition:
    """Validated manifest recipe for a MetricFlow first-stage query."""

    name: str
    base_metrics: tuple[str, ...]
    group_by: tuple[str, ...]
    where: tuple[str, ...]


class FactoryAdvancedPlanCompiler:
    """Build additional native MetricFlow SQL plan stages from typed recipes."""

    def compile(self, name: str, source: SqlPlanNode) -> SqlSelectStatementNode:
        """Create a named second-stage business metric plan."""
        if name == "mart_page_weekly":
            return self._page_weekly(source)
        if name == "mart_ctr_curve":
            return self._ctr_curve(source)
        if name == "mart_striking_distance":
            return self._striking_distance(source)
        if name == "mart_cannibalization":
            return self._cannibalization(source)
        raise ValueError(f"Unsupported advanced semantic plan: {name}")

    @staticmethod
    def _col(alias: str, name: str) -> SqlColumnReferenceExpression:
        return SqlColumnReferenceExpression.from_column_reference(alias, name)

    @staticmethod
    def _sum(expression):
        return SqlAggregateFunctionExpression.create(SqlFunction.SUM, (expression,))

    @staticmethod
    def _nonempty_page(alias: str):
        return SqlComparisonExpression.create(
            FactoryAdvancedPlanCompiler._col(alias, "page__page_url"),
            SqlComparison.NOT_EQUALS, SqlStringLiteralExpression.create(""),
        )

    @staticmethod
    def _page_weekly(source: SqlPlanNode) -> SqlSelectStatementNode:
        src = "metricflow_source"

        def col(name: str) -> SqlColumnReferenceExpression:
            return SqlColumnReferenceExpression.from_column_reference(src, name)

        project = col("project__project_id")
        page = col("page__page_url")
        week = col("metric_time__week")
        numerator = col("key_events")
        denominator = SqlWindowFunctionExpression.create(
            SqlWindowFunction.SUM, sql_function_args=(numerator,),
            partition_by_args=(project, week),
        )
        output_names = {
            "project_id": "project__project_id",
            "page_url": "page__page_url",
            "segment": "page__segment",
            "iso_week": "metric_time__week",
            "clicks": "page_clicks",
            "impressions": "page_impressions",
            "ctr": "page_ctr",
            "avg_position_weighted": "page_avg_position",
            "organic_sessions": "organic_sessions",
            "key_events": "key_events",
        }
        columns = tuple(SqlSelectColumn(col(source_name), output_name)
                        for output_name, source_name in output_names.items())
        columns += (SqlSelectColumn(
            SqlRatioComputationExpression.create(numerator, denominator),
            "key_events_share"),)
        return SqlSelectStatementNode.create(
            description="Organic page key-event share within project and ISO week",
            select_columns=columns,
            from_source=source,
            from_source_alias=src,
            where=SqlComparisonExpression.create(page, SqlComparison.NOT_EQUALS,
                                                 SqlStringLiteralExpression.create("")),
        )

    @staticmethod
    def _ctr_curve(source: SqlPlanNode) -> SqlSelectStatementNode:
        col = FactoryAdvancedPlanCompiler._col
        src = "metricflow_source"
        position = col(src, "query_page_avg_position")
        buckets = ((1, 2, "1"), (2, 3, "2"), (3, 4, "3"), (4, 6, "4-5"),
                   (6, 11, "6-10"), (11, 21, "11-20"))
        bucket_expr = SqlCaseExpression.create({
            SqlLogicalExpression.create(SqlLogicalOperator.AND, (
                SqlComparisonExpression.create(position, SqlComparison.GREATER_THAN_OR_EQUALS,
                                               SqlIntegerExpression.create(lower)),
                SqlComparisonExpression.create(position, SqlComparison.LESS_THAN,
                                               SqlIntegerExpression.create(upper)),
            )): SqlStringLiteralExpression.create(label)
            for lower, upper, label in buckets
        })
        sample = SqlSelectStatementNode.create(
            description="Assign a bucket to each 90-day page-query CTR sample",
            select_columns=(
                SqlSelectColumn(col(src, "project__project_id"), "project_id"),
                SqlSelectColumn(bucket_expr, "position_bucket"),
                SqlSelectColumn(col(src, "query_page_ctr"), "sample_ctr"),
            ),
            from_source=source, from_source_alias=src,
            where=FactoryAdvancedPlanCompiler._nonempty_page(src),
        )
        sample_alias = "bucketed_samples"
        bucket = col(sample_alias, "position_bucket")
        project = col(sample_alias, "project_id")
        aggregated = SqlSelectStatementNode.create(
            description="Median of page-query CTR samples and sample count per project/bucket",
            select_columns=(
                SqlSelectColumn(project, "project_id"),
                SqlSelectColumn(bucket, "position_bucket"),
                SqlSelectColumn(SqlPercentileExpression.create(
                    col(sample_alias, "sample_ctr"),
                    SqlPercentileExpressionArgument.create(.5, False, False)), "raw_median"),
                SqlSelectColumn(FactoryAdvancedPlanCompiler._sum(SqlIntegerExpression.create(1)), "n"),
            ),
            from_source=sample, from_source_alias=sample_alias,
            where=SqlComparisonExpression.create(bucket, SqlComparison.NOT_EQUALS,
                                                 SqlStringLiteralExpression.create("")),
            group_bys=(SqlSelectColumn(project, "project_id"),
                       SqlSelectColumn(bucket, "position_bucket")),
        )
        agg_alias = "bucket_totals"
        count = col(agg_alias, "n")
        median = col(agg_alias, "raw_median")
        return SqlSelectStatementNode.create(
            description="Suppress CTR medians below 30 page-query samples",
            select_columns=(
                SqlSelectColumn(col(agg_alias, "project_id"), "project_id"),
                SqlSelectColumn(col(agg_alias, "position_bucket"), "position_bucket"),
                SqlSelectColumn(SqlCaseExpression.create({
                    SqlComparisonExpression.create(count, SqlComparison.GREATER_THAN_OR_EQUALS,
                                                   SqlIntegerExpression.create(30)): median,
                }, else_expr=SqlNullExpression.create()), "ctr_median"),
                SqlSelectColumn(count, "n"),
            ),
            from_source=aggregated, from_source_alias=agg_alias,
        )

    @staticmethod
    def _striking_distance(source: SqlPlanNode) -> SqlSelectStatementNode:
        col = FactoryAdvancedPlanCompiler._col
        src = "metricflow_source"
        group_names = ("project__project_id", "page__page_url", "search_page_query_daily__query")
        group_columns = tuple(SqlSelectColumn(col(src, name), name) for name in group_names)
        aggregate = SqlSelectStatementNode.create(
            description="Aggregate 28 days and population SD of daily page-query positions",
            select_columns=group_columns + (
                SqlSelectColumn(FactoryAdvancedPlanCompiler._sum(col(src, "query_page_impressions")),
                                "impressions_28d"),
                SqlSelectColumn(FactoryAdvancedPlanCompiler._sum(col(src, "query_page_clicks")),
                                "clicks_28d"),
                SqlSelectColumn(FactoryAdvancedPlanCompiler._sum(col(src, "query_page_weighted_position")),
                                "weighted_position_28d"),
                SqlSelectColumn(SqlAggregateFunctionExpression.create(
                    SqlFunction.STDDEV_POP, (col(src, "query_page_avg_position"),)),
                    "position_sd_28d"),
            ),
            from_source=source, from_source_alias=src,
            where=FactoryAdvancedPlanCompiler._nonempty_page(src),
            group_bys=group_columns,
        )
        agg = "page_query_28d"
        project = col(agg, group_names[0])
        page = col(agg, group_names[1])
        query = col(agg, group_names[2])
        impressions = col(agg, "impressions_28d")
        rank = SqlWindowFunctionExpression.create(
            SqlWindowFunction.ROW_NUMBER,
            partition_by_args=(project, query),
            order_by_args=(SqlWindowOrderByArgument(impressions, descending=True),
                           SqlWindowOrderByArgument(page, descending=False)),
        )
        return SqlSelectStatementNode.create(
            description="Select primary URL by impressions, then lexicographic URL",
            select_columns=(
                SqlSelectColumn(project, "project_id"),
                SqlSelectColumn(page, "page_url"),
                SqlSelectColumn(query, "query"),
                SqlSelectColumn(impressions, "impressions_28d"),
                SqlSelectColumn(col(agg, "clicks_28d"), "clicks_28d"),
                SqlSelectColumn(SqlRatioComputationExpression.create(
                    col(agg, "weighted_position_28d"), impressions), "avg_position_28d"),
                SqlSelectColumn(col(agg, "position_sd_28d"), "position_sd_28d"),
                SqlSelectColumn(SqlComparisonExpression.create(
                    rank, SqlComparison.EQUALS, SqlIntegerExpression.create(1)),
                    "is_primary_page"),
            ),
            from_source=aggregate, from_source_alias=agg,
        )

    @staticmethod
    def _cannibalization(source: SqlPlanNode) -> SqlSelectStatementNode:
        col = FactoryAdvancedPlanCompiler._col
        src = "metricflow_source"
        project = col(src, "project__project_id")
        query = col(src, "search_page_query_daily__query")
        page = col(src, "page__page_url")
        impressions = col(src, "query_page_impressions")
        partition = (project, query)
        denominator = SqlWindowFunctionExpression.create(
            SqlWindowFunction.SUM, sql_function_args=(impressions,), partition_by_args=partition)
        count = SqlWindowFunctionExpression.create(
            SqlWindowFunction.SUM,
            sql_function_args=(SqlIntegerExpression.create(1),), partition_by_args=partition)
        partitioned = SqlSelectStatementNode.create(
            description="Query impression share and distinct page count",
            select_columns=(
                SqlSelectColumn(project, "project_id"),
                SqlSelectColumn(query, "query"),
                SqlSelectColumn(page, "page_url"),
                SqlSelectColumn(SqlRatioComputationExpression.create(impressions, denominator),
                                "impressions_share"),
                SqlSelectColumn(impressions, "impressions_28d"),
                SqlSelectColumn(count, "url_count"),
            ),
            from_source=source, from_source_alias=src,
            where=FactoryAdvancedPlanCompiler._nonempty_page(src),
        )
        part = "partitioned_queries"
        output_names = ("project_id", "query", "page_url", "impressions_share",
                        "impressions_28d")
        return SqlSelectStatementNode.create(
            description="Keep only queries with at least two URLs",
            select_columns=tuple(SqlSelectColumn(col(part, name), name) for name in output_names),
            from_source=partitioned, from_source_alias=part,
            where=SqlComparisonExpression.create(
                col(part, "url_count"), SqlComparison.GREATER_THAN_OR_EQUALS,
                SqlIntegerExpression.create(2)),
        )

    @staticmethod
    def validate_definitions(definitions: Sequence[dict]) -> dict[str, AdvancedQueryDefinition]:
        """Accept only named typed recipes; SQL originates from MetricFlow plans."""
        allowed = {"mart_page_weekly", "mart_ctr_curve", "mart_striking_distance",
                   "mart_cannibalization"}
        result = {}
        for item in definitions:
            if not isinstance(item, dict) or item.get("name") not in allowed:
                raise ValueError("Unknown advanced MetricFlow plan definition")
            name = item["name"]
            if name in result:
                raise ValueError(f"Duplicate advanced MetricFlow plan: {name}")
            metrics = item.get("base_metrics")
            groups = item.get("group_by")
            filters = item.get("where", [])
            if not all(isinstance(values, list) and all(isinstance(value, str) for value in values)
                       for values in (metrics, groups, filters)):
                raise ValueError(f"Invalid advanced MetricFlow plan: {name}")
            if not metrics or not groups:
                raise ValueError(f"Incomplete advanced MetricFlow plan: {name}")
            result[name] = AdvancedQueryDefinition(name, tuple(metrics), tuple(groups), tuple(filters))
        return result
