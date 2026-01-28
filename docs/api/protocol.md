# Protocol

The Protocol class captures transformation history for auditability and reproducibility.

## Overview

When you process data with a TransformPlan, you receive both the transformed DataFrame and a Protocol object. The protocol contains:

- Input/output DataFrame hashes for verification
- Step-by-step operation details
- Shape changes and timing information
- Optional metadata

```python
from transformplan import TransformPlan

plan = TransformPlan().col_drop("temp").math_multiply("price", 1.1)
df_result, protocol = plan.process(df)

# View the protocol
protocol.print()

# Save for audit
protocol.to_json("audit_trail.json")
```

## Protocol Class

::: transformplan.protocol.Protocol
    options:
      show_root_heading: true
      members:
        - set_input
        - set_metadata
        - add_step
        - input_hash
        - output_hash
        - metadata
        - to_dataframe
        - to_csv
        - to_dict
        - from_dict
        - to_json
        - from_json
        - summary
        - print

## frame_hash Function

::: transformplan.protocol.frame_hash
    options:
      show_root_heading: true

## Example Output

The `print()` method generates a formatted summary:

```
======================================================================
TRANSFORM PROTOCOL
======================================================================
Input:  1000 rows x 5 cols  [a1b2c3d4e5f6g7h8]
Output: 850 rows x 4 cols   [h8g7f6e5d4c3b2a1]
Total time: 0.0234s
----------------------------------------------------------------------

#    Operation            Rows         Cols         Time       Hash
----------------------------------------------------------------------
0    input                1000         5            -          a1b2c3d4e5f6g7h8
1    col_drop             1000         4 (-1)       0.0012s    b2c3d4e5f6g7h8a1
     -> column='temp'
2    math_multiply        1000         4            0.0008s    c3d4e5f6g7h8a1b2
     -> column='price', value=1.1
3    rows_filter          850 (-150)   4            0.0214s    h8g7f6e5d4c3b2a1
     -> filter=(age >= 18)
======================================================================
```

## Reproducibility

The frame_hash function computes a deterministic hash that is:

- **Row-order invariant**: Same rows in different order produce the same hash
- **Column-order invariant**: Same columns in different order produce the same hash
- **Content-sensitive**: Any value change produces a different hash

This enables verification that the same pipeline on the same input produces identical results.
