import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchsummary import summary

from .model import predict_image, show_train_images


class Question5:
    def showDataAugmentation(self, imgPath):
        if imgPath == None:
            print('Please load the image.')
        else:
            imgRotation = self.showRandomRotation(imgPath)
            imgResized = self.showRandomResizedCrop(imgPath)
            imgFlipped = self.showRandomHorizontalFlip(imgPath)
            result = self.getConcatH(imgRotation, imgResized, imgFlipped)
            result.show()
            # result.save('Q5/augmentation.png')


    def showRandomRotation(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomRotation(degrees=(0, 180))    # rotated degree from 0 to 180
        img = transfrom(img)
        # img.show()
        return img

    def showRandomResizedCrop(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomResizedCrop(size=img.size, scale=(0.05, 0.99)) # random cropped size is 0.05x to 0.99x
        img = transfrom(img)
        # img.show()
        return img

    def showRandomHorizontalFlip(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomHorizontalFlip(p=0.5)   # filp rate is 1/2
        img = transfrom(img)
        # img.show()
        return img

    # mix the images horizontally
    def getConcatH(self, img1, img2, img3):
        concatenated = Image.new('RGB', (img1.width + img2.width + img3.width, img1.height))
        concatenated.paste(img1, (0, 0))
        concatenated.paste(img2, (img1.width, 0))
        concatenated.paste(img3, (img1.width + img2.width, 0))
        return concatenated

    def showTrainImages(self):
        show_train_images()

    def showModelStructure(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cuda:0':
            model = torch.load('Q5/model_vgg19.pth')
        else:
            model = torch.load('Q5/model_vgg19.pth', map_location ='cpu')

        summary(model, (3, 224, 224))   # show model structure

    def makeAccuracyAndLoss(self):
        imgAcc = cv2.imread('Q5/accuracy.png')
        imgLoss = cv2.imread('Q5/loss.png')
        result = np.concatenate((imgAcc, imgLoss), axis=0)  # concat two pictures together
        cv2.imwrite('Q5/result.png', result)

    def showInference(self, imgPath):
        if imgPath == None:
            print('Please load the image.')
        else:
            conf, label = predict_image(imgPath=imgPath)
            return conf, label     



if __name__ == '__main__':
    print('This is Q5')
    print('Do run run this file')