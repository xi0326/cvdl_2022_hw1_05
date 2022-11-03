import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from random import randint
import tqdm
from PIL import Image

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # use GPU 1

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
    return dict

def show_train_images():
    fig, axes = plt.subplots(3, 3)
    trainset = unpickle('Q5/data/cifar-10-batches-py/data_batch_' + str(randint(1, 5)))    # show the images randomly
    datainfo = unpickle('Q5/data/cifar-10-batches-py/batches.meta')

    for i in range(0, 3):
        for j in range(0, 3):
            img = np.reshape(trainset[b'data'][i * 3 + j], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            axes[i, j].imshow(img)
            axes[i, j].set_title(datainfo[b'label_names'][trainset[b'labels'][i * 3 + j]].decode())
            axes[i, j].set_axis_off()

    plt.show()

def validate(device, model, criterion, test_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for data in test_dataloader:
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss / len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct / len(test_dataloader.dataset)
    
    return val_loss, val_accuracy

def fit(device, model, criterion, optimizer, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for data in train_dataloader:
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy

def train():
    # detect device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyper parameter
    # batch_size=32
    # learning_rate = 1e-3

    # Data
    # for SGD
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    """ # for Adam
    transform = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.ToTensor()])
    """
    
    trainset = torchvision.datasets.CIFAR10(root='Q5/data', train=True, # train set
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='Q5/data', train=False, # test set
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False)
    

    model = torchvision.models.vgg19(weights = True) # using vgg19
    model.to(device)
  
    model.classifier[6].out_features = 10   # change the number of classes to 10

    # freeze convolution weights
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()   # cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=0.01) # choose Adam as optimiser
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate) # choose Adam as optimiser
    # optimizer = optim.Adam(model.parameters())    # choose Adam as optimiser, default learning rate is 1e-3


    # start training
    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    for epoch in tqdm.trange(50):   # show process bar, train 50 epochs
        train_epoch_loss, train_epoch_accuracy = fit(device, model, criterion, optimizer, trainloader)
        val_epoch_loss, val_epoch_accuracy = validate(device, model, criterion, testloader)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    # create accuracy and loss images
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy')
    plt.ylabel('%')
    plt.plot(train_accuracy, color='blue', label='Training')
    plt.plot(val_accuracy, color='orange', label='Testing')
    plt.legend()
    plt.savefig('accuracy.png')
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss, color='blue', label='Loss')
    plt.legend()
    plt.savefig('loss.png')
    # plt.show()

    summary(model, (3, 224, 224))   # show model structure

    torch.save(model, 'model_vgg19.pth')    # save the model


def predict_image(imgPath):
    if imgPath == None:
        print('Please load the image.')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device == 'cuda:0':
            model = torch.load('Q5/model_vgg19.pth')
        else:
            model = torch.load('Q5/model_vgg19.pth', map_location ='cpu')
        
        model.eval()    # turn on evaluation mode

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        
        img = Image.open(imgPath)
        img_tensor = transform(img)    # change the img matrix to a tensor
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.view(1, 3, 224, 224)  # batch_size = 1, channel = 3(RGB), img_size = 224 * 224
        eval = model(img_tensor)
        eval = nn.functional.softmax(eval, dim=1)

        conf, classes = torch.max(eval, 1)  # predict
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']    # CIFAR10 dataset class name
        
        # print(float(conf), class_names[classes.item()])
        return float(conf), class_names[classes.item()]


if __name__ == '__main__':
    train()
