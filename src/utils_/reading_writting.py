import os


def get_image_paths(folder_path):
    image_paths = []
    for file_name in os.listdir(folder_path):
        # Check if the file is an image (you can adjust the conditions as needed)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(os.path.join(folder_path, file_name))
    return image_paths