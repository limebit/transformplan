# Roadmap

Planned additions that are deliberately **not** in the current release, with the
reasoning and the groundwork already in place.

Both items below add new `@abstractmethod` entries to the `Backend` ABC
(`transformplan/backends/base.py`). That breaks any backend implemented outside
this repository, so they belong in a minor release (0.3.0), not a patch.

---

## Aggregation — `group_by` with aggregates

**Status:** planned, largest known gap.

There is `rows_pivot` and `rows_melt`, but no way to group with aggregation. Any
pipeline whose job is to condense onto an entity — one row per case, customer, or
transaction — has to leave the plan at exactly that point. The consequence is that
the most complex part of such a pipeline is the one part with **no protocol**,
while the simpler row and column operations are documented in full.

### Proposed API

```python
.group_by("case_id", {
    "n_positions": ("*", "count"),
    "total_amount": ("amount", "sum"),
    "n_categories": ("field", "n_unique"),
})
```

### Groundwork already present

Two pieces exist and should be reused rather than rebuilt:

- **`AggFunction`** (`backends/base.py`) already defines the aggregate vocabulary
  `Literal["first", "sum", "mean", "median", "min", "max", "count"]`, used today by
  `rows_pivot` and `math_diff_from_agg`. `group_by` should extend this literal
  (`n_unique`, `last`, `std`) rather than introduce a second vocabulary.
- **`ChunkMode.GROUP_DEPENDENT` with `group_param`** (`chunking.py`) already models
  "needs all rows of a group together", used by `math_cumsum`, `math_rank` and
  `rows_unique`. Registering `group_by` as
  `OperationMeta(ChunkMode.GROUP_DEPENDENT, group_param="by")` makes it correct
  under `process_chunked()` whenever `partition_key == by` — so aggregation fits
  *inside* the chunking model instead of standing beside it.

### Where the real cost is

Not in the backends. The expensive part is `SchemaTracker` in `validation.py`:
every existing validator *advances* the schema, whereas `group_by` must **replace**
it — after the step, only the grouping columns and the aggregate outputs exist.
This is the one place where the current forward-propagation logic does not carry
over. Beyond that: DuckDB SQL generation and `dry_run` output.

### Constraint to hold

Keep the aggregate vocabulary **closed** — names only, never arbitrary callables.
A lambda cannot be serialized, and serializability is the library's core promise.

---

## Time components — `dt_hour`, `dt_minute`, `dt_second`

**Status:** planned, small.

The `dt_` family covers date components thoroughly but has no time components.
Turning `16:30` into `16.5` currently takes six steps and a detour through text
formatting and back:

```python
.dt_format("admitted_at", "%H", "admit_h")
.dt_format("admitted_at", "%M", "admit_m")
.col_cast("admit_h", "Float64")
.col_cast("admit_m", "Float64")
.math_divide("admit_m", 60)
.math_add_columns("admit_h", "admit_m", "admit_hour")
```

With `dt_hour` and `dt_minute` this becomes three steps and no string round trip.

### Notes for implementation

- DuckDB is trivial here: `EXTRACT(hour FROM …)`.
- Each new operation touches **nine files**: `ops/datetime.py`, `backends/base.py`,
  `backends/polars.py`, `backends/duckdb.py`, `validation.py` (validator plus
  registry), `chunking.py` (registry), `tests/test_datetime.py`,
  `tests/test_duckdb.py`, and `docs/api/ops/datetime.md`. Budget for that, not
  just for the operation itself.
- **`dt_decimal_hour` is deliberately excluded.** With `dt_hour`/`dt_minute` the
  block is already down to three steps, and the library has no other convenience
  combinations. Adding one opens a door to an unbounded set of special cases.

---

## Delivered in this release

For reference, the related suggestions that **are** implemented:

| Item | What shipped |
|---|---|
| `.pipe()` | Apply a function that takes and returns a plan |
| `.extend()` / `+` | Compose plans from reusable blocks |
| `.section()` | Named sections, grouped with a balance in the protocol |
| `.because()` | Reasons recorded per step in the protocol |
| Column sequences | 24 in-place operations accept `str \| Sequence[str]` |
| `col_cast` names | Canonical dtype names keep plans serializable |
