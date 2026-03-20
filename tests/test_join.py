"""Tests for join operations."""

from __future__ import annotations

import polars as pl
import pytest

from transformplan import TransformPlan


@pytest.fixture
def main_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85, 72, 91, 68, 79],
        }
    )


@pytest.fixture
def cohort_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "person_id": [1, 3, 5],
        }
    )


@pytest.fixture
def concepts_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "concept_id": [85, 72, 91, 68, 79],
            "concept_name": ["Excellent", "Good", "Outstanding", "Fair", "Good+"],
            "category": ["A", "B", "A", "C", "B"],
        }
    )


class TestJoinInner:
    """Tests for inner join (cohort filtering)."""

    def test_inner_join_filters_rows(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        result, _ = plan.process(main_df, references={"cohort": cohort_df})
        assert len(result) == 3
        assert set(result["person_id"].to_list()) == {1, 3, 5}

    def test_inner_join_preserves_columns(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        result, _ = plan.process(main_df, references={"cohort": cohort_df})
        assert result.columns == ["person_id", "name", "score"]


class TestJoinLeft:
    """Tests for left join (enrichment)."""

    def test_left_join_keeps_all_rows(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(
            on="score",
            right_name="concepts",
            how="left",
            right_on="concept_id",
            select_columns=["concept_name"],
        )
        result, _ = plan.process(main_df, references={"concepts": concepts_df})
        assert len(result) == 5
        assert "concept_name" in result.columns

    def test_left_join_enriches_data(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(
            on="score",
            right_name="concepts",
            how="left",
            right_on="concept_id",
            select_columns=["concept_name"],
        )
        result, _ = plan.process(main_df, references={"concepts": concepts_df})
        # Alice has score 85 -> "Excellent"
        alice_row = result.filter(pl.col("name") == "Alice")
        assert alice_row["concept_name"][0] == "Excellent"


class TestJoinLeftOnRightOn:
    """Tests for left_on/right_on (different column names)."""

    def test_different_column_names(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(
            on="score",
            right_name="concepts",
            how="left",
            left_on="score",
            right_on="concept_id",
        )
        result, _ = plan.process(main_df, references={"concepts": concepts_df})
        assert len(result) == 5
        assert "concept_name" in result.columns
        assert "category" in result.columns


class TestJoinSuffix:
    """Tests for suffix handling with duplicate columns."""

    def test_suffix_on_duplicate_columns(self) -> None:
        left = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        right = pl.DataFrame({"id": [1, 2], "value": [100, 200]})
        plan = TransformPlan().join(
            on="id", right_name="right", how="left", suffix="_r"
        )
        result, _ = plan.process(left, references={"right": right})
        assert "value" in result.columns
        assert "value_r" in result.columns

    def test_default_suffix(self) -> None:
        left = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        right = pl.DataFrame({"id": [1, 2], "value": [100, 200]})
        plan = TransformPlan().join(on="id", right_name="right", how="left")
        result, _ = plan.process(left, references={"right": right})
        assert "value_right" in result.columns


class TestJoinSelectColumns:
    """Tests for select_columns parameter."""

    def test_select_specific_columns(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(
            on="score",
            right_name="concepts",
            how="left",
            right_on="concept_id",
            select_columns=["concept_name"],
        )
        result, _ = plan.process(main_df, references={"concepts": concepts_df})
        assert "concept_name" in result.columns
        assert "category" not in result.columns

    def test_no_select_columns_gets_all(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(
            on="score",
            right_name="concepts",
            how="left",
            right_on="concept_id",
        )
        result, _ = plan.process(main_df, references={"concepts": concepts_df})
        assert "concept_name" in result.columns
        assert "category" in result.columns


class TestJoinMissingReference:
    """Tests for missing reference error."""

    def test_missing_reference_raises(self, main_df: pl.DataFrame) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        with pytest.raises(ValueError, match="Reference 'cohort' not found"):
            plan.process(main_df)

    def test_wrong_reference_name_raises(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        with pytest.raises(ValueError, match="Reference 'cohort' not found"):
            plan.process(main_df, references={"wrong_name": cohort_df})


class TestJoinSerialization:
    """Tests for serialization round-trip."""

    def test_to_json_from_json_roundtrip(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        json_str = plan.to_json()

        plan2 = TransformPlan.from_json(json_str)
        result1, _ = plan.process(main_df, references={"cohort": cohort_df})
        result2, _ = plan2.process(main_df, references={"cohort": cohort_df})
        assert result1.equals(result2)

    def test_to_dict_contains_join(self) -> None:
        plan = TransformPlan().join(
            on="id",
            right_name="ref",
            how="left",
            select_columns=["col1"],
        )
        d = plan.to_dict()
        assert d["steps"][0]["operation"] == "join"
        assert d["steps"][0]["params"]["right_name"] == "ref"
        assert d["steps"][0]["params"]["select_columns"] == ["col1"]

    def test_to_python_contains_join(self) -> None:
        plan = TransformPlan().join(on="id", right_name="ref", how="inner")
        code = plan.to_python()
        assert ".join(" in code
        assert 'right_name="ref"' in code


class TestJoinValidation:
    """Tests for validation with and without references."""

    def test_validation_without_references(self, main_df: pl.DataFrame) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        result = plan.validate(main_df)
        # Should pass — left-side columns exist
        assert result.is_valid

    def test_validation_with_references(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        result = plan.validate(main_df, references={"cohort": cohort_df})
        assert result.is_valid

    def test_validation_fails_missing_left_column(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="nonexistent", right_name="cohort", how="inner")
        result = plan.validate(main_df, references={"cohort": cohort_df})
        assert not result.is_valid

    def test_validation_fails_missing_right_column(self, main_df: pl.DataFrame) -> None:
        ref = pl.DataFrame({"other_col": [1, 2]})
        plan = TransformPlan().join(on="person_id", right_name="ref", how="inner")
        result = plan.validate(main_df, references={"ref": ref})
        assert not result.is_valid

    def test_validation_with_select_columns_adds_to_schema(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = (
            TransformPlan()
            .join(
                on="score",
                right_name="concepts",
                how="left",
                right_on="concept_id",
                select_columns=["concept_name"],
            )
            .str_upper("concept_name")
        )
        result = plan.validate(main_df, references={"concepts": concepts_df})
        assert result.is_valid


class TestJoinDryRun:
    """Tests for dry run with join."""

    def test_dry_run_shows_added_columns(
        self, main_df: pl.DataFrame, concepts_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(
            on="score",
            right_name="concepts",
            how="left",
            right_on="concept_id",
            select_columns=["concept_name"],
        )
        preview = plan.dry_run(main_df, references={"concepts": concepts_df})
        assert preview.is_valid
        assert "concept_name" in preview.output_columns

    def test_dry_run_without_references(self, main_df: pl.DataFrame) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        preview = plan.dry_run(main_df)
        # Should pass — just checks left-side columns
        assert preview.is_valid


class TestJoinChunking:
    """Tests for chunking compatibility."""

    def test_join_is_global(self) -> None:
        plan = TransformPlan().join(on="id", right_name="ref", how="inner")
        validation = plan.validate_chunked(
            schema={"id": pl.Int64()},
        )
        assert not validation.is_valid
        assert "join" in validation.global_operations


class TestJoinProtocol:
    """Tests for protocol recording."""

    def test_protocol_records_reference_hashes(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        _, protocol = plan.process(main_df, references={"cohort": cohort_df})
        meta = protocol.metadata
        assert "references" in meta
        assert "cohort" in meta["references"]
        assert "hash" in meta["references"]["cohort"]
        assert "shape" in meta["references"]["cohort"]

    def test_protocol_step_has_right_name(
        self, main_df: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> None:
        plan = TransformPlan().join(on="person_id", right_name="cohort", how="inner")
        _, protocol = plan.process(main_df, references={"cohort": cohort_df})
        steps = protocol.to_dict()["steps"]
        assert steps[0]["operation"] == "join"
        assert steps[0]["params"]["right_name"] == "cohort"
        # right_data should NOT be in params (not serializable)
        assert "right_data" not in steps[0]["params"]


class TestJoinMultiColumn:
    """Tests for multi-column joins."""

    def test_join_on_multiple_columns(self) -> None:
        left = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "val": [10, 20, 30]})
        right = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "extra": [100, 200]})
        plan = TransformPlan().join(on=["a", "b"], right_name="right", how="inner")
        result, _ = plan.process(left, references={"right": right})
        assert len(result) == 2
        assert "extra" in result.columns
