import json

def json_to_dict(json_file: str) -> dict:
    """
    Reads a JSON file and converts it to a dictionary.

    Parameters
    ----------
    json_file : str
        The path to the JSON file.

    Returns
    -------
    dict
        A dictionary containing the data from the JSON file.
    """
    with open(json_file, 'r') as f:
        data_dict = json.load(f)
    return data_dict