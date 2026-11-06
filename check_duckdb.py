try:
    import duckdb
    print("DuckDB version:", duckdb.__version__)
except ImportError:
    print("DuckDB is not installed.")
