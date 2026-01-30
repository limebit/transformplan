"""Tests for chunked processing (chunking.py and core.py process_chunked)."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from transformplan import (
    ChunkedProtocol,
    ChunkingError,
    ChunkValidationResult,
    Col,
    TransformPlan,
)
from transformplan.chunking import (
    OPERATION_CHUNK_REGISTRY,
    ChunkInfo,
    ChunkMode,
    OperationMeta,
    validate_chunked_pipeline,
)


class TestChunkMode:
    """Tests for ChunkMode enum."""

    def test_chunk_mode_values(self) -> None:
        """Test that ChunkMode has expected values."""
        assert ChunkMode.CHUNKABLE.value == 1
        assert ChunkMode.GROUP_DEPENDENT.value == 2
        assert ChunkMode.GLOBAL.value == 3


class TestOperationMeta:
    """Tests for OperationMeta dataclass."""

    def test_chunkable_operation(self) -> None:
        """Test metadata for chunkable operation."""
        meta = OperationMeta(ChunkMode.CHUNKABLE)
        assert meta.chunk_mode == ChunkMode.CHUNKABLE
        assert meta.group_param is None

    def test_group_dependent_operation(self) -> None:
        """Test metadata for group-dependent operation."""
        meta = OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="columns")
        assert meta.chunk_mode == ChunkMode.GROUP_DEPENDENT
        assert meta.group_param == "columns"


class TestOperationChunkRegistry:
    """Tests for OPERATION_CHUNK_REGISTRY."""

    def test_registry_has_all_column_ops(self) -> None:
        """Test that registry includes all column operations."""
        column_ops = [
            "col_drop",
            "col_rename",
            "col_cast",
            "col_reorder",
            "col_select",
            "col_duplicate",
            "col_fill_null",
            "col_drop_null",
            "col_drop_zero",
            "col_add",
            "col_add_uuid",
            "col_hash",
            "col_coalesce",
        ]
        for op in column_ops:
            assert op in OPERATION_CHUNK_REGISTRY
            assert OPERATION_CHUNK_REGISTRY[op].chunk_mode == ChunkMode.CHUNKABLE

    def test_registry_has_all_math_ops(self) -> None:
        """Test that registry includes all math operations."""
        chunkable_math_ops = [
            "math_add",
            "math_subtract",
            "math_multiply",
            "math_divide",
            "math_clamp",
            "math_abs",
            "math_round",
            "math_set_min",
            "math_set_max",
            "math_add_columns",
            "math_subtract_columns",
            "math_multiply_columns",
            "math_divide_columns",
            "math_percent_of",
        ]
        for op in chunkable_math_ops:
            assert op in OPERATION_CHUNK_REGISTRY
            assert OPERATION_CHUNK_REGISTRY[op].chunk_mode == ChunkMode.CHUNKABLE

    def test_registry_has_group_dependent_ops(self) -> None:
        """Test that group-dependent operations are registered correctly."""
        group_ops = {
            "math_cumsum": "group_by",
            "math_rank": "group_by",
            "rows_unique": "columns",
            "rows_deduplicate": "columns",
        }
        for op, param in group_ops.items():
            assert op in OPERATION_CHUNK_REGISTRY
            meta = OPERATION_CHUNK_REGISTRY[op]
            assert meta.chunk_mode == ChunkMode.GROUP_DEPENDENT
            assert meta.group_param == param

    def test_registry_has_global_ops(self) -> None:
        """Test that global operations are registered correctly."""
        global_ops = [
            "rows_sort",
            "rows_pivot",
            "rows_sample",
            "rows_head",
            "rows_tail",
        ]
        for op in global_ops:
            assert op in OPERATION_CHUNK_REGISTRY
            assert OPERATION_CHUNK_REGISTRY[op].chunk_mode == ChunkMode.GLOBAL


class TestValidateChunkedPipeline:
    """Tests for validate_chunked_pipeline function."""

    def test_empty_pipeline_is_valid(self) -> None:
        """Test that empty pipeline is valid for chunking."""
        result = validate_chunked_pipeline([], partition_key=None)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_chunkable_only_pipeline_valid(self) -> None:
        """Test that pipeline with only chunkable ops is valid."""
        plan = TransformPlan().col_drop("x").col_rename("a", "b").math_add("c", 1)
        result = validate_chunked_pipeline(plan._operations, partition_key=None)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_global_operation_blocks_chunking(self) -> None:
        """Test that global operations make pipeline invalid."""
        plan = TransformPlan().col_drop("x").rows_sort("y")
        result = validate_chunked_pipeline(plan._operations, partition_key=None)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "rows_sort" in result.errors[0]
        assert "rows_sort" in result.global_operations

    def test_multiple_global_operations(self) -> None:
        """Test that multiple global operations are all reported."""
        plan = TransformPlan().rows_sort("x").rows_head(10)
        result = validate_chunked_pipeline(plan._operations, partition_key=None)
        assert not result.is_valid
        assert len(result.errors) == 2
        assert "rows_sort" in result.global_operations
        assert "rows_head" in result.global_operations

    def test_group_dependent_without_partition_key_invalid(self) -> None:
        """Test that group-dependent ops without partition key are invalid."""
        plan = TransformPlan().rows_unique(columns=["id"])
        result = validate_chunked_pipeline(plan._operations, partition_key=None)
        assert not result.is_valid
        assert "rows_unique" in result.errors[0]

    def test_group_dependent_with_matching_partition_key_valid(self) -> None:
        """Test that group-dependent ops with matching partition key are valid."""
        plan = TransformPlan().rows_unique(columns=["patient_id"])
        result = validate_chunked_pipeline(plan._operations, partition_key="patient_id")
        assert result.is_valid
        assert len(result.errors) == 0

    def test_group_dependent_with_subset_partition_key_valid(self) -> None:
        """Test that group-dependent ops are valid if partition key is superset."""
        plan = TransformPlan().rows_unique(columns=["patient_id"])
        result = validate_chunked_pipeline(
            plan._operations, partition_key=["patient_id", "visit_date"]
        )
        assert result.is_valid

    def test_group_dependent_with_mismatched_partition_key_invalid(self) -> None:
        """Test that group-dependent ops with wrong partition key are invalid."""
        plan = TransformPlan().rows_unique(columns=["patient_id", "visit_date"])
        result = validate_chunked_pipeline(plan._operations, partition_key="patient_id")
        assert not result.is_valid
        assert "visit_date" in result.errors[0]

    def test_group_dependent_without_group_param_invalid(self) -> None:
        """Test that group-dependent ops without group columns are invalid."""
        plan = TransformPlan().math_cumsum("value", new_column="cumsum")
        result = validate_chunked_pipeline(plan._operations, partition_key="id")
        assert not result.is_valid
        assert "group_by" in result.errors[0]

    def test_group_dependent_with_group_param_valid(self) -> None:
        """Test that group-dependent ops with matching group param are valid."""
        plan = TransformPlan().math_cumsum("value", new_column="cumsum", group_by="id")
        result = validate_chunked_pipeline(plan._operations, partition_key="id")
        assert result.is_valid


class TestChunkValidationResult:
    """Tests for ChunkValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test valid result string representation."""
        result = ChunkValidationResult(is_valid=True)
        assert "compatible" in str(result)

    def test_invalid_result_with_errors(self) -> None:
        """Test invalid result string representation."""
        result = ChunkValidationResult(
            is_valid=False,
            errors=["Operation 'rows_sort' requires full dataset"],
            global_operations=["rows_sort"],
        )
        s = str(result)
        assert "NOT compatible" in s
        assert "rows_sort" in s


class TestChunkInfo:
    """Tests for ChunkInfo dataclass."""

    def test_chunk_info_creation(self) -> None:
        """Test creating ChunkInfo."""
        info = ChunkInfo(
            chunk_index=0,
            input_rows=1000,
            output_rows=950,
            input_hash="abc123",
            output_hash="def456",
            elapsed_seconds=0.5,
        )
        assert info.chunk_index == 0
        assert info.input_rows == 1000
        assert info.output_rows == 950


class TestChunkedProtocol:
    """Tests for ChunkedProtocol class."""

    def test_empty_protocol(self) -> None:
        """Test empty protocol."""
        protocol = ChunkedProtocol()
        assert protocol.num_chunks == 0
        assert protocol.total_input_rows == 0
        assert protocol.total_output_rows == 0
        assert len(protocol) == 0

    def test_add_chunks(self) -> None:
        """Test adding chunks to protocol."""
        protocol = ChunkedProtocol()
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=1000,
                output_rows=950,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.5,
            )
        )
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=1,
                input_rows=1000,
                output_rows=980,
                input_hash="ghi",
                output_hash="jkl",
                elapsed_seconds=0.4,
            )
        )
        assert protocol.num_chunks == 2
        assert protocol.total_input_rows == 2000
        assert protocol.total_output_rows == 1930
        assert protocol.total_elapsed_seconds == pytest.approx(0.9)

    def test_set_metadata(self) -> None:
        """Test setting metadata."""
        protocol = ChunkedProtocol()
        protocol.set_metadata(author="test", version="1.0")
        assert protocol.metadata["author"] == "test"
        assert protocol.metadata["version"] == "1.0"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        protocol = ChunkedProtocol()
        protocol.set_source(path="test.parquet", partition_key=["id"], chunk_size=1000)
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=90,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )
        d = protocol.to_dict()
        assert d["version"] == "1.0"
        assert d["source"]["path"] == "test.parquet"
        assert d["summary"]["num_chunks"] == 1
        assert len(d["chunks"]) == 1

    def test_from_dict_roundtrip(self) -> None:
        """Test serialization round trip."""
        original = ChunkedProtocol()
        original.set_source(path="test.parquet", partition_key=["id"], chunk_size=1000)
        original.set_metadata(test="value")
        original.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=90,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )

        restored = ChunkedProtocol.from_dict(original.to_dict())
        assert restored.num_chunks == original.num_chunks
        assert restored.total_input_rows == original.total_input_rows
        assert restored.metadata == original.metadata

    def test_to_json_string(self) -> None:
        """Test serialization to JSON string."""
        protocol = ChunkedProtocol()
        json_str = protocol.to_json()
        assert isinstance(json_str, str)
        assert '"version"' in json_str

    def test_to_json_file(self) -> None:
        """Test serialization to JSON file."""
        protocol = ChunkedProtocol()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            protocol.to_json(path)
            assert path.exists()
            restored = ChunkedProtocol.from_json(path)
            assert restored.num_chunks == 0
        finally:
            path.unlink()

    def test_summary(self) -> None:
        """Test summary generation."""
        protocol = ChunkedProtocol()
        protocol.set_source(path="test.parquet", partition_key=["id"], chunk_size=1000)
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=90,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )
        summary = protocol.summary()
        assert "CHUNKED PROCESSING PROTOCOL" in summary
        assert "test.parquet" in summary
        assert "100" in summary


class TestProcessChunked:
    """Integration tests for process_chunked method."""

    @pytest.fixture
    def sample_parquet(self, tmp_path: Path) -> Path:
        """Create a sample Parquet file for testing."""
        df = pl.DataFrame(
            {
                "patient_id": ["P001"] * 50 + ["P002"] * 30 + ["P003"] * 20,
                "visit_date": list(range(50)) + list(range(30)) + list(range(20)),
                "value": list(range(100)),
                "name": ["Test"] * 100,
            }
        )
        path = tmp_path / "test_data.parquet"
        df.write_parquet(path)
        return path

    def test_process_chunked_basic(self, sample_parquet: Path) -> None:
        """Test basic chunked processing without partition key."""
        plan = TransformPlan().col_drop("name")
        result, protocol = plan.process_chunked(
            source=sample_parquet,
            chunk_size=30,
        )
        assert "name" not in result.columns
        assert len(result) == 100
        assert protocol.num_chunks >= 1
        assert protocol.total_input_rows == 100
        assert protocol.total_output_rows == 100

    def test_process_chunked_with_filter(self, sample_parquet: Path) -> None:
        """Test chunked processing with filter."""
        plan = TransformPlan().rows_filter(Col("value") >= 50)
        result, _ = plan.process_chunked(
            source=sample_parquet,
            chunk_size=30,
        )
        assert len(result) == 50
        assert all(result["value"].to_list()[i] >= 50 for i in range(len(result)))

    def test_process_chunked_with_partition_key(self, sample_parquet: Path) -> None:
        """Test chunked processing with partition key."""
        plan = TransformPlan().rows_unique(columns=["patient_id"])
        result, _ = plan.process_chunked(
            source=sample_parquet,
            partition_key="patient_id",
            chunk_size=40,
        )
        # Should have 3 unique patients
        assert len(result) == 3
        assert set(result["patient_id"].to_list()) == {"P001", "P002", "P003"}

    def test_process_chunked_validates_global_ops(self, sample_parquet: Path) -> None:
        """Test that global operations raise ChunkingError."""
        plan = TransformPlan().rows_sort("value")
        with pytest.raises(ChunkingError) as exc_info:
            plan.process_chunked(source=sample_parquet, chunk_size=30)
        assert "rows_sort" in str(exc_info.value)
        assert exc_info.value.validation_result is not None

    def test_process_chunked_validates_schema(self, sample_parquet: Path) -> None:
        """Test that schema validation is performed."""
        from transformplan.validation import SchemaValidationError

        plan = TransformPlan().col_drop("nonexistent_column")
        with pytest.raises(SchemaValidationError, match="nonexistent_column"):
            plan.process_chunked(source=sample_parquet, chunk_size=30)

    def test_process_chunked_skip_validation(self, sample_parquet: Path) -> None:
        """Test that validation can be skipped."""
        plan = TransformPlan().col_drop("name")
        result, _ = plan.process_chunked(
            source=sample_parquet,
            chunk_size=30,
            validate=False,
        )
        assert "name" not in result.columns

    def test_process_chunked_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing file."""
        plan = TransformPlan().col_drop("x")
        with pytest.raises(FileNotFoundError):
            plan.process_chunked(source=tmp_path / "nonexistent.parquet")

    def test_process_chunked_protocol_records_operations(
        self, sample_parquet: Path
    ) -> None:
        """Test that protocol records operations."""
        plan = TransformPlan().col_drop("name").math_add("value", 10)
        _, protocol = plan.process_chunked(source=sample_parquet, chunk_size=50)
        d = protocol.to_dict()
        assert len(d["operations"]) == 2
        assert d["operations"][0]["operation"] == "col_drop"
        assert d["operations"][1]["operation"] == "math_add"

    def test_process_chunked_protocol_has_source_info(
        self, sample_parquet: Path
    ) -> None:
        """Test that protocol records source info."""
        plan = TransformPlan().col_drop("name")
        _, protocol = plan.process_chunked(
            source=sample_parquet,
            partition_key=["patient_id"],
            chunk_size=50,
        )
        d = protocol.to_dict()
        assert str(sample_parquet) in d["source"]["path"]
        assert d["source"]["partition_key"] == ["patient_id"]
        assert d["source"]["chunk_size"] == 50


class TestValidateChunked:
    """Tests for validate_chunked method on TransformPlan."""

    def test_validate_chunked_with_schema(self) -> None:
        """Test validation with schema dict."""
        plan = TransformPlan().col_drop("x")
        result = plan.validate_chunked(
            schema={"x": pl.Int64, "y": pl.Utf8},
            partition_key=None,
        )
        assert result.is_valid

    def test_validate_chunked_with_data(self) -> None:
        """Test validation with DataFrame."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        plan = TransformPlan().col_drop("x")
        result = plan.validate_chunked(data=df, partition_key=None)
        assert result.is_valid

    def test_validate_chunked_requires_schema_or_data(self) -> None:
        """Test that validation requires schema or data."""
        plan = TransformPlan().col_drop("x")
        with pytest.raises(ValueError, match="Either schema or data must be provided"):
            plan.validate_chunked()

    def test_validate_chunked_global_op(self) -> None:
        """Test validation catches global operations."""
        plan = TransformPlan().rows_sort("x")
        result = plan.validate_chunked(
            schema={"x": pl.Int64},
            partition_key=None,
        )
        assert not result.is_valid
        assert "rows_sort" in result.global_operations


class TestEdgeCases:
    """Tests for edge cases in chunked processing."""

    @pytest.fixture
    def empty_parquet(self, tmp_path: Path) -> Path:
        """Create an empty Parquet file."""
        df = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )
        path = tmp_path / "empty.parquet"
        df.write_parquet(path)
        return path

    @pytest.fixture
    def single_row_parquet(self, tmp_path: Path) -> Path:
        """Create a Parquet file with single row."""
        df = pl.DataFrame({"id": [1], "value": [100.0]})
        path = tmp_path / "single.parquet"
        df.write_parquet(path)
        return path

    def test_empty_file(self, empty_parquet: Path) -> None:
        """Test processing empty Parquet file."""
        plan = TransformPlan().col_drop("value")
        result, protocol = plan.process_chunked(source=empty_parquet, chunk_size=10)
        assert len(result) == 0
        assert protocol.num_chunks == 0

    def test_single_row_file(self, single_row_parquet: Path) -> None:
        """Test processing single row Parquet file."""
        plan = TransformPlan().math_multiply("value", 2)
        result, protocol = plan.process_chunked(
            source=single_row_parquet, chunk_size=10
        )
        assert len(result) == 1
        assert result["value"][0] == 200.0
        assert protocol.num_chunks == 1

    def test_chunk_size_larger_than_file(self, single_row_parquet: Path) -> None:
        """Test when chunk_size exceeds file size."""
        plan = TransformPlan().math_add("value", 1)
        result, protocol = plan.process_chunked(
            source=single_row_parquet, chunk_size=1000000
        )
        assert len(result) == 1
        assert protocol.num_chunks == 1

    @pytest.fixture
    def large_single_group_parquet(self, tmp_path: Path) -> Path:
        """Create a Parquet file with one large group."""
        df = pl.DataFrame(
            {
                "patient_id": ["P001"] * 100,
                "value": list(range(100)),
            }
        )
        path = tmp_path / "single_group.parquet"
        df.write_parquet(path)
        return path

    def test_partition_key_single_large_group(
        self, large_single_group_parquet: Path
    ) -> None:
        """Test partition chunking when one group exceeds chunk_size."""
        plan = TransformPlan().rows_unique(columns=["patient_id"])
        result, _ = plan.process_chunked(
            source=large_single_group_parquet,
            partition_key="patient_id",
            chunk_size=30,  # Smaller than the group size
        )
        # All 100 rows should be processed together since they're same patient
        # rows_unique should reduce to 1 row
        assert len(result) == 1
        assert result["patient_id"][0] == "P001"


class TestChunkValidationResultStr:
    """Tests for ChunkValidationResult.__str__() edge cases."""

    def test_str_valid_with_warnings(self) -> None:
        """Test __str__ for valid result with warnings."""
        result = ChunkValidationResult(
            is_valid=True,
            warnings=["Unknown operation 'custom_op' - assuming chunkable"],
        )
        s = str(result)
        assert "compatible" in s
        assert "Warnings:" in s
        assert "custom_op" in s

    def test_str_group_dependent_ops(self) -> None:
        """Test __str__ for result with group-dependent operations."""
        result = ChunkValidationResult(
            is_valid=True,
            group_dependent_ops=[("rows_unique", ["patient_id"])],
        )
        s = str(result)
        assert "compatible" in s
        assert "Group-dependent operations" in s
        assert "rows_unique" in s
        assert "patient_id" in s


class TestChunkedProtocolEdgeCases:
    """Tests for ChunkedProtocol edge cases."""

    def test_output_hash_empty(self) -> None:
        """Test output_hash returns empty string when no chunks."""
        protocol = ChunkedProtocol()
        assert protocol.output_hash() == ""

    def test_repr(self) -> None:
        """Test __repr__ output."""
        protocol = ChunkedProtocol()
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=90,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )
        r = repr(protocol)
        assert "ChunkedProtocol" in r
        assert "1 chunks" in r
        assert "100 rows" in r

    def test_len(self) -> None:
        """Test __len__ returns number of chunks."""
        protocol = ChunkedProtocol()
        assert len(protocol) == 0
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=90,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )
        assert len(protocol) == 1

    def test_summary_with_metadata(self) -> None:
        """Test summary includes metadata."""
        protocol = ChunkedProtocol()
        protocol.set_source(path="test.parquet", partition_key=["id"], chunk_size=1000)
        protocol.set_metadata(author="test_user", project="test_project")
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=90,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )
        summary = protocol.summary()
        assert "author: test_user" in summary
        assert "project: test_project" in summary

    def test_summary_with_row_change(self) -> None:
        """Test summary shows row change when rows differ."""
        protocol = ChunkedProtocol()
        protocol.set_source(path="test.parquet", partition_key=None, chunk_size=1000)
        protocol.add_chunk(
            ChunkInfo(
                chunk_index=0,
                input_rows=100,
                output_rows=80,
                input_hash="abc",
                output_hash="def",
                elapsed_seconds=0.1,
            )
        )
        summary = protocol.summary()
        assert "Row change:" in summary
        assert "-20" in summary

    def test_print(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test print method."""
        protocol = ChunkedProtocol()
        protocol.set_source(path="test.parquet", partition_key=None, chunk_size=1000)
        protocol.print()
        captured = capsys.readouterr()
        assert "CHUNKED PROCESSING PROTOCOL" in captured.out

    def test_from_json_string(self) -> None:
        """Test from_json with JSON string."""
        original = ChunkedProtocol()
        original.set_source(path="test.parquet", partition_key=["id"], chunk_size=1000)
        json_str = original.to_json()
        restored = ChunkedProtocol.from_json(json_str)
        assert restored._source_path == "test.parquet"

    def test_chunks_property(self) -> None:
        """Test chunks property returns list of ChunkInfo."""
        protocol = ChunkedProtocol()
        chunk = ChunkInfo(
            chunk_index=0,
            input_rows=100,
            output_rows=90,
            input_hash="abc",
            output_hash="def",
            elapsed_seconds=0.1,
        )
        protocol.add_chunk(chunk)
        chunks = protocol.chunks
        assert len(chunks) == 1
        assert chunks[0].input_rows == 100


class TestValidateChunkedPipelineUnknownOp:
    """Tests for unknown operations in chunked validation."""

    def test_unknown_operation_warning(self) -> None:
        """Test that unknown operations produce a warning."""

        # Create a fake operation that's not in the registry
        class FakeOp:
            @staticmethod
            def _custom_unknown_op(df: pl.DataFrame) -> pl.DataFrame:
                return df

        fake_ops = [(FakeOp._custom_unknown_op, {})]
        result = validate_chunked_pipeline(fake_ops, partition_key=None)
        assert result.is_valid  # Unknown ops don't block
        assert len(result.warnings) == 1
        assert "custom_unknown_op" in result.warnings[0]


class TestPartitionChunkIteratorEdgeCases:
    """Tests for partition chunk iterator edge cases."""

    @pytest.fixture
    def empty_parquet_with_partition(self, tmp_path: Path) -> Path:
        """Create an empty Parquet file with partition column."""
        df = pl.DataFrame(
            {
                "patient_id": pl.Series([], dtype=pl.Utf8),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )
        path = tmp_path / "empty_partitioned.parquet"
        df.write_parquet(path)
        return path

    def test_partition_iterator_empty_file(
        self, empty_parquet_with_partition: Path
    ) -> None:
        """Test partition chunk iterator with empty file returns early."""
        plan = TransformPlan().rows_unique(columns=["patient_id"])
        result, protocol = plan.process_chunked(
            source=empty_parquet_with_partition,
            partition_key="patient_id",
            chunk_size=10,
        )
        assert len(result) == 0
        assert protocol.num_chunks == 0


class TestValidateChunkedGroupColsNormalization:
    """Tests for group_cols string normalization in chunked validation."""

    def test_group_cols_as_string_normalized_to_list(self) -> None:
        """Test that group_cols as string is normalized to list."""
        # When group_by is specified as a string (not a list), it should be normalized
        plan = TransformPlan().math_cumsum("value", new_column="cumsum", group_by="id")
        result = validate_chunked_pipeline(plan._operations, partition_key="id")
        assert result.is_valid
        # Check that group_dependent_ops has the operation with group_cols as list
        assert len(result.group_dependent_ops) == 1
        op_name, group_cols = result.group_dependent_ops[0]
        assert op_name == "math_cumsum"
        assert group_cols == ["id"]

    def test_group_cols_string_directly_in_params(self) -> None:
        """Test group_cols normalization when param is a string directly."""

        # Directly create operation tuple with string group_cols to test normalization
        def _mock_group_op(df: pl.DataFrame, columns: str) -> pl.DataFrame:
            _ = columns  # Unused but required in signature
            return df

        # Register as GROUP_DEPENDENT with 'columns' param
        from transformplan.chunking import (
            OPERATION_CHUNK_REGISTRY,
            ChunkMode,
            OperationMeta,
        )

        # Temporarily add a mock operation with string columns
        OPERATION_CHUNK_REGISTRY["mock_group_op"] = OperationMeta(
            ChunkMode.GROUP_DEPENDENT, group_param="columns"
        )
        try:
            # Create fake operation with string columns (not list)
            fake_ops = [(_mock_group_op, {"columns": "patient_id"})]
            result = validate_chunked_pipeline(fake_ops, partition_key="patient_id")
            assert result.is_valid
            # The group_cols should be normalized to a list
            op_name, group_cols = result.group_dependent_ops[0]
            assert op_name == "mock_group_op"
            assert group_cols == ["patient_id"]
        finally:
            # Cleanup
            del OPERATION_CHUNK_REGISTRY["mock_group_op"]


class TestProcessChunkedPendingRows:
    """Tests for pending rows handling in partition chunk iterator."""

    @pytest.fixture
    def multi_group_parquet(self, tmp_path: Path) -> Path:
        """Create a Parquet file with multiple groups for testing pending rows."""
        # Create data where groups don't evenly divide into chunks
        # Group P001 has 5 rows, P002 has 3 rows, P003 has 2 rows
        df = pl.DataFrame(
            {
                "patient_id": ["P001"] * 5 + ["P002"] * 3 + ["P003"] * 2,
                "value": list(range(10)),
            }
        )
        path = tmp_path / "multi_group.parquet"
        df.write_parquet(path)
        return path

    def test_process_chunked_with_pending_rows(
        self, multi_group_parquet: Path
    ) -> None:
        """Test chunked processing handles pending rows correctly."""
        # Use a chunk size that will cause groups to be split
        plan = TransformPlan().rows_unique(columns=["patient_id"])
        result, protocol = plan.process_chunked(
            source=multi_group_parquet,
            partition_key="patient_id",
            chunk_size=4,  # Will force partial groups
        )
        # Should have 3 unique patients
        assert len(result) == 3
        assert set(result["patient_id"].to_list()) == {"P001", "P002", "P003"}
        # Should have processed at least 1 chunk
        assert protocol.num_chunks >= 1
