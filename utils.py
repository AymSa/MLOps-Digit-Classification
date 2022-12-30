from torchvision import transforms
import torch
from PIL import Image
import cv2

#Probleme le modele a été entrainé sur des torch.float32 mais ne possede pas suffisament d'infos 
def process_data(data):

    return transforms.Compose(
        [transforms.PILToTensor(), transforms.Grayscale(), transforms.Resize((28, 28))]
    )(data).to(torch.float32)


if __name__ == "__main__":

    path_img = "/home/reinstate/Desktop/7.png"

    pil = Image.open(path_img)

    data = process_data(pil)
    print(data.shape)
    cv2.imshow('img', data.permute(1,2,0).numpy())
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
