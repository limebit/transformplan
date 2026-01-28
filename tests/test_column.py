"""Tests for column operations (ops/column.py)."""

import polars as pl

from transformplan import TransformPlan


class TestColDrop:
    """Tests for col_drop operation."""

    def test_col_drop_single(self, basic_df: pl.DataFrame) -> None:
        """Test dropping a single column."""
        plan = TransformPlan().col_drop("age")
        result, _ = plan.process(basic_df)
        assert "age" not in result.columns
        assert len(result.columns) == len(basic_df.columns) - 1

    def test_col_drop_preserves_other_columns(self, basic_df: pl.DataFrame) -> None:
        """Test that other columns are preserved."""
        plan = TransformPlan().col_drop("age")
        result, _ = plan.process(basic_df)
        assert "id" in result.columns
        assert "name" in result.columns
        assert "salary" in result.columns

    def test_col_drop_nonexistent_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that dropping nonexistent column fails validation."""
        plan = TransformPlan().col_drop("nonexistent")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "does not exist" in str(result.errors[0])


class TestColRename:
    """Tests for col_rename operation."""

    def test_col_rename_basic(self, basic_df: pl.DataFrame) -> None:
        """Test basic column rename."""
        plan = TransformPlan().col_rename("name", "full_name")
        result, _ = plan.process(basic_df)
        assert "full_name" in result.columns
        assert "name" not in result.columns

    def test_col_rename_preserves_data(self, basic_df: pl.DataFrame) -> None:
        """Test that data is preserved after rename."""
        plan = TransformPlan().col_rename("name", "full_name")
        result, _ = plan.process(basic_df)
        assert result["full_name"].to_list() == basic_df["name"].to_list()

    def test_col_rename_nonexistent_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that renaming nonexistent column fails validation."""
        plan = TransformPlan().col_rename("nonexistent", "new_name")
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_col_rename_to_existing_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that renaming to existing column fails validation."""
        plan = TransformPlan().col_rename("name", "age")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])


class TestColCast:
    """Tests for col_cast operation."""

    def test_col_cast_int_to_float(self, basic_df: pl.DataFrame) -> None:
        """Test casting int to float."""
        plan = TransformPlan().col_cast("age", pl.Float64)
        result, _ = plan.process(basic_df)
        assert result["age"].dtype == pl.Float64

    def test_col_cast_int_to_string(self, basic_df: pl.DataFrame) -> None:
        """Test casting int to string."""
        plan = TransformPlan().col_cast("id", pl.Utf8)
        result, _ = plan.process(basic_df)
        assert result["id"].dtype == pl.Utf8
        assert result["id"][0] == "1"

    def test_col_cast_float_to_int(self, basic_df: pl.DataFrame) -> None:
        """Test casting float to int."""
        plan = TransformPlan().col_cast("salary", pl.Int64)
        result, _ = plan.process(basic_df)
        assert result["salary"].dtype == pl.Int64


class TestColReorder:
    """Tests for col_reorder operation."""

    def test_col_reorder_basic(self, basic_df: pl.DataFrame) -> None:
        """Test basic column reordering."""
        plan = TransformPlan().col_reorder(["name", "id", "age"])
        result, _ = plan.process(basic_df)
        assert result.columns == ["name", "id", "age"]

    def test_col_reorder_drops_unlisted(self, basic_df: pl.DataFrame) -> None:
        """Test that unlisted columns are dropped."""
        plan = TransformPlan().col_reorder(["id", "name"])
        result, _ = plan.process(basic_df)
        assert result.columns == ["id", "name"]
        assert "age" not in result.columns

    def test_col_reorder_nonexistent_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that nonexistent column fails validation."""
        plan = TransformPlan().col_reorder(["id", "nonexistent"])
        result = plan.validate(basic_df)
        assert not result.is_valid


class TestColSelect:
    """Tests for col_select operation."""

    def test_col_select_subset(self, basic_df: pl.DataFrame) -> None:
        """Test selecting a subset of columns."""
        plan = TransformPlan().col_select(["id", "name"])
        result, _ = plan.process(basic_df)
        assert result.columns == ["id", "name"]

    def test_col_select_single(self, basic_df: pl.DataFrame) -> None:
        """Test selecting a single column."""
        plan = TransformPlan().col_select(["name"])
        result, _ = plan.process(basic_df)
        assert result.columns == ["name"]

    def test_col_select_preserves_order(self, basic_df: pl.DataFrame) -> None:
        """Test that column order is preserved."""
        plan = TransformPlan().col_select(["salary", "name", "id"])
        result, _ = plan.process(basic_df)
        assert result.columns == ["salary", "name", "id"]


class TestColDuplicate:
    """Tests for col_duplicate operation."""

    def test_col_duplicate_basic(self, basic_df: pl.DataFrame) -> None:
        """Test basic column duplication."""
        plan = TransformPlan().col_duplicate("name", "name_copy")
        result, _ = plan.process(basic_df)
        assert "name_copy" in result.columns
        assert result["name_copy"].to_list() == result["name"].to_list()

    def test_col_duplicate_nonexistent_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that duplicating nonexistent column fails."""
        plan = TransformPlan().col_duplicate("nonexistent", "copy")
        result = plan.validate(basic_df)
        assert not result.is_valid

    def test_col_duplicate_to_existing_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that duplicating to existing column fails."""
        plan = TransformPlan().col_duplicate("name", "age")
        result = plan.validate(basic_df)
        assert not result.is_valid


class TestColFillNull:
    """Tests for col_fill_null operation."""

    def test_col_fill_null_with_value(self, df_with_nulls: pl.DataFrame) -> None:
        """Test filling nulls with a value."""
        plan = TransformPlan().col_fill_null("name", value="Unknown")
        result, _ = plan.process(df_with_nulls)
        assert result["name"].null_count() == 0
        assert "Unknown" in result["name"].to_list()

    def test_col_fill_null_with_forward(self, df_with_nulls: pl.DataFrame) -> None:
        """Test filling nulls with forward strategy."""
        plan = TransformPlan().col_fill_null("name", strategy="forward")
        result, _ = plan.process(df_with_nulls)
        # Forward fill should reduce null count
        assert result["name"].null_count() < df_with_nulls["name"].null_count()

    def test_col_fill_null_with_zero(self, df_with_nulls: pl.DataFrame) -> None:
        """Test filling nulls with zero strategy."""
        plan = TransformPlan().col_fill_null("age", strategy="zero")
        result, _ = plan.process(df_with_nulls)
        assert result["age"].null_count() == 0
        assert 0 in result["age"].to_list()


class TestColDropNull:
    """Tests for col_drop_null operation."""

    def test_col_drop_null_specific_column(self, df_with_nulls: pl.DataFrame) -> None:
        """Test dropping rows with nulls in specific column."""
        plan = TransformPlan().col_drop_null("name")
        result, _ = plan.process(df_with_nulls)
        assert result["name"].null_count() == 0
        assert len(result) < len(df_with_nulls)

    def test_col_drop_null_all_columns(self, df_with_nulls: pl.DataFrame) -> None:
        """Test dropping rows with nulls in any column."""
        plan = TransformPlan().col_drop_null()
        result, _ = plan.process(df_with_nulls)
        # Should have fewer rows since we drop any row with any null
        assert len(result) < len(df_with_nulls)


class TestColDropZero:
    """Tests for col_drop_zero operation."""

    def test_col_drop_zero(self) -> None:
        """Test dropping rows with zero values."""
        df = pl.DataFrame({"a": [1, 0, 3, 0, 5], "b": [10, 20, 30, 40, 50]})
        plan = TransformPlan().col_drop_zero("a")
        result, _ = plan.process(df)
        assert 0 not in result["a"].to_list()
        assert len(result) == 3


class TestColAdd:
    """Tests for col_add operation."""

    def test_col_add_with_value(self, basic_df: pl.DataFrame) -> None:
        """Test adding column with constant value."""
        plan = TransformPlan().col_add("status", value="active")
        result, _ = plan.process(basic_df)
        assert "status" in result.columns
        assert all(s == "active" for s in result["status"].to_list())

    def test_col_add_with_expr(self, basic_df: pl.DataFrame) -> None:
        """Test adding column by copying another column."""
        plan = TransformPlan().col_add("name_copy", expr="name")
        result, _ = plan.process(basic_df)
        assert "name_copy" in result.columns
        assert result["name_copy"].to_list() == result["name"].to_list()

    def test_col_add_numeric_value(self, basic_df: pl.DataFrame) -> None:
        """Test adding column with numeric value."""
        plan = TransformPlan().col_add("discount", value=0.1)
        result, _ = plan.process(basic_df)
        assert "discount" in result.columns
        assert all(d == 0.1 for d in result["discount"].to_list())

    def test_col_add_to_existing_raises(self, basic_df: pl.DataFrame) -> None:
        """Test that adding existing column fails validation."""
        plan = TransformPlan().col_add("name", value="test")
        result = plan.validate(basic_df)
        assert not result.is_valid
        assert "already exists" in str(result.errors[0])


class TestColAddUuid:
    """Tests for col_add_uuid operation."""

    def test_col_add_uuid_basic(self, basic_df: pl.DataFrame) -> None:
        """Test adding UUID column."""
        plan = TransformPlan().col_add_uuid("uuid")
        result, _ = plan.process(basic_df)
        assert "uuid" in result.columns
        assert result["uuid"].dtype == pl.Utf8

    def test_col_add_uuid_unique(self, basic_df: pl.DataFrame) -> None:
        """Test that UUIDs are unique."""
        plan = TransformPlan().col_add_uuid("uuid")
        result, _ = plan.process(basic_df)
        uuids = result["uuid"].to_list()
        assert len(uuids) == len(set(uuids))

    def test_col_add_uuid_length(self, basic_df: pl.DataFrame) -> None:
        """Test UUID length."""
        plan = TransformPlan().col_add_uuid("uuid", length=8)
        result, _ = plan.process(basic_df)
        assert all(len(u) == 8 for u in result["uuid"].to_list())


class TestColHash:
    """Tests for col_hash operation."""

    def test_col_hash_single_column(self, basic_df: pl.DataFrame) -> None:
        """Test hashing a single column."""
        plan = TransformPlan().col_hash("name", "name_hash")
        result, _ = plan.process(basic_df)
        assert "name_hash" in result.columns
        assert result["name_hash"].dtype == pl.Utf8

    def test_col_hash_multiple_columns(self, basic_df: pl.DataFrame) -> None:
        """Test hashing multiple columns."""
        plan = TransformPlan().col_hash(["id", "name"], "combined_hash")
        result, _ = plan.process(basic_df)
        assert "combined_hash" in result.columns

    def test_col_hash_with_salt(self, basic_df: pl.DataFrame) -> None:
        """Test hashing with salt."""
        plan1 = TransformPlan().col_hash("name", "hash1", salt="")
        plan2 = TransformPlan().col_hash("name", "hash2", salt="secret")
        result1, _ = plan1.process(basic_df)
        result2, _ = plan2.process(basic_df)
        # Different salts should produce different hashes
        assert result1["hash1"][0] != result2["hash2"][0]

    def test_col_hash_deterministic(self, basic_df: pl.DataFrame) -> None:
        """Test that hash is deterministic."""
        plan = TransformPlan().col_hash("name", "hash")
        result1, _ = plan.process(basic_df)
        result2, _ = plan.process(basic_df)
        assert result1["hash"].to_list() == result2["hash"].to_list()


class TestColCoalesce:
    """Tests for col_coalesce operation."""

    def test_col_coalesce_basic(self) -> None:
        """Test basic coalesce operation."""
        df = pl.DataFrame(
            {
                "a": [None, 2, None, 4],
                "b": [10, None, 30, None],
                "c": [100, 200, 300, 400],
            }
        )
        plan = TransformPlan().col_coalesce(["a", "b", "c"], "result")
        result, _ = plan.process(df)
        assert "result" in result.columns
        assert result["result"].to_list() == [10, 2, 30, 4]

    def test_col_coalesce_all_null(self) -> None:
        """Test coalesce when all values are null."""
        df = pl.DataFrame(
            {
                "a": [None, None],
                "b": [None, None],
            }
        )
        plan = TransformPlan().col_coalesce(["a", "b"], "result")
        result, _ = plan.process(df)
        assert result["result"].null_count() == 2


class TestMethodChaining:
    """Tests for method chaining of column operations."""

    def test_multiple_operations(self, basic_df: pl.DataFrame) -> None:
        """Test chaining multiple column operations."""
        plan = (
            TransformPlan()
            .col_rename("name", "full_name")
            .col_drop("active")
            .col_add("status", value="active")
        )
        result, _ = plan.process(basic_df)
        assert "full_name" in result.columns
        assert "name" not in result.columns
        assert "active" not in result.columns
        assert "status" in result.columns

    def test_chaining_preserves_order(self, basic_df: pl.DataFrame) -> None:
        """Test that operations execute in chain order."""
        # Rename then drop the renamed column
        plan = TransformPlan().col_rename("name", "full_name").col_drop("full_name")
        result, _ = plan.process(basic_df)
        assert "name" not in result.columns
        assert "full_name" not in result.columns
