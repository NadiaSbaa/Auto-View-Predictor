from pathlib import Path


def validate_input_data(data_path):
    # Check if 'data_path' key is present
    if not data_path:
        return False, 'Missing / Empty parameter: data_path'

    # Check if 'data_path' exist
    data_file = Path(data_path)
    if not data_file.exists():
        return False, 'Invalid data_path value. Directory or File does not exist'

    return True, None