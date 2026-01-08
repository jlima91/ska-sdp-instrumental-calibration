def write_csv(filepath: str, rows: list[list[str]]):
    """
    Write rows to a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to write.
    rows : list of list of str
        List of rows, where each row is a list of string values.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
