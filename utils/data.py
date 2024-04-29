import os


def get_image_filepaths(data_path):
    """
    Retrieves file paths of all images within a directory.

    Args:
        data_path (str): The path to the directory containing images.

    Returns:
        List[str]: A list of file paths to the images.
    """
    filepaths = []

    # If the given path is a directory
    if os.path.isdir(data_path):
        # Recursively traverse the directory
        for root, dirs, files in os.walk(data_path):
            for file in files:
                # Check if the file is an image file
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    filepaths.append(os.path.join(root, file))
    # If the given path is a file
    elif os.path.isfile(data_path):
        # Check if it's an image file
        if data_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filepaths.append(data_path)

    return filepaths
