from deepface import DeepFace


def represent_image_model(path_img):
    return DeepFace.represent(img_path=path_img)[0]["embedding"]
