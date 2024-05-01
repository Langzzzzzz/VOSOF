import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionNet(nn.Module):
    def __init__(self, num_classes, height, width):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv2d(num_classes * 2, num_classes * 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_classes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=0.3)
        
        self.conv2 = nn.Conv2d(num_classes * 4, num_classes * 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_classes * 8)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=0.5)
        
        self.conv3 = nn.Conv2d(num_classes * 8, num_classes * 16, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_classes * 16)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(num_classes * 16, num_classes * 8, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(num_classes * 8)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout2d(p=0.8)


        self.conv5 = nn.Conv2d(num_classes * 8, num_classes * 4, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(num_classes * 4)
        self.relu5 = nn.ReLU(inplace=True)


        self.weight_conv = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


        
    def forward(self, seg_softmax, of_confidence):

        # Stack both inputs along the channel dimension
        x = torch.cat([seg_softmax, of_confidence], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        weights = self.weight_conv(x)
        weights = self.sigmoid(weights)
        
        # x = x.permute(1, 0, 2, 3)

        # # Normalize weights to range [0, 1] for a weighted sum
        # normalized_weights = torch.sigmoid(self.weights)
        
        # # Apply the learned weights to combine the processed features with the original E_softmax
        combined_output = weights * seg_softmax + (1 - weights) * of_confidence
        
        return combined_output

# # Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes, height, width = 11, 480, 854
model = FusionNet(num_classes, height, width).to(device)

print(model)
# # generate random input with shape (1, 11, 480, 854)
# seg_softmax = torch.randn(1, num_classes, height, width).to(device)
# of_confidence = torch.randn(1, num_classes, height, width).to(device)
# out = model(seg_softmax, of_confidence)
# print(out.shape)
