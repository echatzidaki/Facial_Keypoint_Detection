## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.findFeatures = nn.Sequential(
            # This network takes in a square (same width and height), grayscale image as input = (224,224)
            # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
            # (W-F+2P)/S +1 

            # output size =(224-5+0)/1 +1 = 220
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32, 220, 220) --> (32, 110, 110)
            nn.MaxPool2d(2, 2),

            # output size = (110-3)/1 +1 = 108
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (64, 108, 108) --> (64, 54, 54)
            nn.MaxPool2d(2, 2),

            # output size = (54-5)/1 +1 = 50
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            # (128, 50, 50) --> (128, 25, 25)
            nn.MaxPool2d(2, 2),

            # output size = (25-1+2)/2 +1 = 14
            nn.Conv2d(128, 256, 1, 2, 1),
            nn.ReLU(),
            # (256, 14, 14) --> (256, 7, 7)
            nn.MaxPool2d(2, 2),

            # output size = (7-2+2)/1 +1 = 8
            nn.Conv2d(256, 512, 2, 1, 1),
            nn.ReLU(),
            # (512, 8, 8) --> (512, 4, 4)
            nn.MaxPool2d(2, 2)


#             # output size =(224-5+0)/1 +1 = 220
#             nn.Conv2d(1, 32, 5),
# #             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             # (32, 220, 220) --> (32, 110, 110)
#             nn.MaxPool2d(2, 2),

#             # output size = (110-3)/1 +1 = 108
#             nn.Conv2d(32, 64, 3),
# #             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             # (64, 108, 108) --> (64, 54, 54)
#             nn.MaxPool2d(2, 2),

#             # output size = (54-3)/1 +1 = 52
#             nn.Conv2d(64, 128, 3),
#             nn.ReLU(),
#             # (128, 52, 52) --> (128, 26, 26)
#             nn.MaxPool2d(2, 2),

#             # output size = (26-3)/1 +1= 24
#             nn.Conv2d(128, 256, 3),
#             nn.ReLU(),
#             # (256, 24, 24) --> (256, 12, 12)
#             nn.MaxPool2d(2, 2)

        )

        self.classify = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # It ends with a linear layer that represents the keypoints
            ## Last layer output: 136 values, 2 for each of the 68 keypoint (x, y) pairs
            nn.Linear(512, 136),
            nn.Dropout(p=0.2)
#             nn.Linear(256*12*12, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 136),
#             nn.Dropout(p=0.2)
        )
        # self.fc1 = nn.Linear(512*4*4, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # # It ends with a linear layer that represents the keypoints
        # ## Last layer output: 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # self.fc3 = nn.Linear(512, 136)
        # self.drop = nn.Dropout(p=0.4)
        
        
    def forward(self, x):
        
        #Define the feedforward behavior of this model
        # x is the input image 
        x = self.findFeatures(x)

        x = x.view(x.size(0), -1)

        x = self.classify(x)
        
        # a modified x, having gone through all the layers of your model
        return x
