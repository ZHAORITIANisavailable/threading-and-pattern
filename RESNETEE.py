import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class EarlyExit(nn.Module):
    def __init__(self, in_planes,num_blocks, num_classes=10, block=BasicBlock, downsample=False):
        super(EarlyExit, self).__init__()
        self.block = block
        self.in_planes = in_planes
        stride = 2 if downsample else 1
        # Adding a block to the early exit using make_layer
        self.block_layer = self.make_layer(block, in_planes, num_blocks, stride=stride)
        
        # Average pooling and FC layer for classification in the early exit
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_planes, num_classes)        
        
        
        
    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    
    
    def forward(self, x,return_features=False):
        x = self.block_layer(x)  # Apply the block layer to the input
        features = self.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        else:
            return logits
    

    
class ResNet18AutoEarlyExits(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18AutoEarlyExits, self).__init__()
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        # === 自行构建 ResNet18_32x32 ===
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # === 添加 Early Exits ===
        self.early_exit1 = EarlyExit(128, num_blocks=2, num_classes=num_classes, downsample=True)
        self.early_exit2 = EarlyExit(256, num_blocks=2, num_classes=num_classes, downsample=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_exit1=False, return_exit2=False,
                auto_select=False, threshold1=0.8872, threshold2=0.5428, disabled_exits=None):

        batch_size = x.size(0)
        device = x.device

        # general part
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        

        if auto_select:
            logits_early1, features_early1 = self.early_exit1(feature3, return_features=True)
            if disabled_exits is None:
                disabled_exits = []

            conf1 = torch.softmax(logits_early1, dim=1).max(dim=1)[0]
            exit1_mask = (conf1 > threshold1) & (torch.tensor(1 not in disabled_exits).to(device))

            # middle part
            feature4 = self.layer3(feature3)
            logits_early2, features_early2 = self.early_exit2(feature4, return_features=True)

            conf2 = torch.softmax(logits_early2, dim=1).max(dim=1)[0]
            exit2_mask = (conf2 > threshold2) & ~exit1_mask & (torch.tensor(2 not in disabled_exits).to(device))

            # main branch
            feature5 = self.layer4(feature4)
            feature5 = self.avgpool(feature5)
            feature = feature5.view(batch_size, -1)
            logits_main = self.fc(feature)
            exit3_mask = ~exit1_mask & ~exit2_mask

            logits = torch.zeros_like(logits_main)
            exit_ids = torch.zeros(batch_size, dtype=torch.long).to(device)

            logits[exit1_mask] = logits_early1[exit1_mask]
            logits[exit2_mask] = logits_early2[exit2_mask]
            logits[exit3_mask] = logits_main[exit3_mask]

            exit_ids[exit1_mask] = 1
            exit_ids[exit2_mask] = 2
            exit_ids[exit3_mask] = 3

            return logits, exit_ids

        if return_exit1:
            logits_early1, features_early1 = self.early_exit1(feature3, return_features=True)
            return logits_early1, features_early1

        feature4 = self.layer3(feature3)
        

        if return_exit2:
            logits_early2, features_early2 = self.early_exit2(feature4, return_features=True)
            return logits_early2, features_early2

        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(batch_size, -1)
        main_out = self.fc(feature)

        return main_out, feature


    ### model(x, return_exit1=True)	只用出口1（原始模式）
#model(x, auto_select=True)	自动判断置信度退出
#model(x, auto_select=True, threshold1=0.85)	改变退出阈值
#model(x)



class ResNet18WithEarlyExits(nn.Module):
    def __init__(self, resnet, num_classes=10):
        super(ResNet18WithEarlyExits, self).__init__()
        self.resnet = resnet
        self.early_exit1 = EarlyExit(128, num_blocks=2,num_classes=num_classes)
        self.early_exit2 = EarlyExit(256, num_blocks=2,num_classes=num_classes)


    def forward(self, x, return_exit1=False, return_exit2=False):
        # Pass input through the initial layers of ResNet
        feature1 = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        feature2 = self.resnet.layer1(feature1)
        feature3 = self.resnet.layer2(feature2)

        # Early exit 1
        logits_early1,features_early1 = self.early_exit1(feature3,return_features=True)
        if return_exit1:
            return logits_early1,features_early1

        # Further layers for main branch and second exit
        feature4 = self.resnet.layer3(feature3)

        # Early exit 2
        logits_early2,features_early2 = self.early_exit2(feature4,return_features=True)
        if return_exit2:
            return logits_early2,features_early2

        # Final ResNet layers for the main branch
        feature5 = self.resnet.layer4(feature4)
        feature5 = self.resnet.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        main_out = self.resnet.fc(feature)

        return main_out, feature
    

class ResNet18AutoEarlyExitstemp(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18AutoEarlyExitstemp, self).__init__()
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        # === 自行构建 ResNet18_32x32 ===
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # === 添加 Early Exits ===
        self.early_exit1 = EarlyExit(128, num_blocks=2, num_classes=num_classes, downsample=True)
        self.early_exit2 = EarlyExit(256, num_blocks=2, num_classes=num_classes, downsample=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_exit1=False, return_exit2=False,
                auto_select=False, threshold1=0.8872, threshold2=0.5428, disabled_exits=None):

        batch_size = x.size(0)
        device = x.device

        # general part
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        
        logits_early1, features_early1 = self.early_exit1(feature3, return_features=True)


        if auto_select:
            
            if disabled_exits is None:
                disabled_exits = []

            conf1 = torch.softmax(logits_early1, dim=1).max(dim=1)[0]
            exit1_mask = (conf1 > threshold1) & (torch.tensor(1 not in disabled_exits).to(device))

            # middle part
            feature4 = self.layer3(feature3)
            logits_early2, features_early2 = self.early_exit2(feature4, return_features=True)

            conf2 = torch.softmax(logits_early2, dim=1).max(dim=1)[0]
            exit2_mask = (conf2 > threshold2) & ~exit1_mask & (torch.tensor(2 not in disabled_exits).to(device))

            # main branch
            feature5 = self.layer4(feature4)
            feature5 = self.avgpool(feature5)
            feature = feature5.view(batch_size, -1)
            logits_main = self.fc(feature)
            exit3_mask = ~exit1_mask & ~exit2_mask

            logits = torch.zeros_like(logits_main)
            exit_ids = torch.zeros(batch_size, dtype=torch.long).to(device)

            logits[exit1_mask] = logits_early1[exit1_mask]
            logits[exit2_mask] = logits_early2[exit2_mask]
            logits[exit3_mask] = logits_main[exit3_mask]

            exit_ids[exit1_mask] = 1
            exit_ids[exit2_mask] = 2
            exit_ids[exit3_mask] = 3

            return logits, exit_ids

        if return_exit1:           
            return logits_early1, features_early1

        feature4 = self.layer3(feature3)
        
        logits_early2, features_early2 = self.early_exit2(feature4, return_features=True)
        
        if return_exit2:
            return logits_early2, features_early2

        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(batch_size, -1)
        main_out = self.fc(feature)

        return main_out, feature