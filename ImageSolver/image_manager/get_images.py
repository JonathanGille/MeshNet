import os
from .custom_image import CustomImage

def get_images_of_folder(folder):
    dirlist = os.listdir(folder)
    # image_paths = [os.path.join(folder,img) for img in dirlist]
    # image_names = [img[:-4] for img in dirlist]

    images = [CustomImage(path=os.path.join(folder,img), name=img[:-4]) for img in dirlist if img.endswith('.png') or img.endswith('.jpg')]

    # images = [c_image(path) for path in image_paths]
    # return (images, image_names)

    return images

def get_image(img_path):
    image = CustomImage(path=img_path, name=os.path.basename(img_path[:-4]))
    return image


