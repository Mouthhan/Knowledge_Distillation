import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
from tqdm.auto import tqdm
from model import DistillModel, DistillModel_DepthSep
import random
import numpy as np
from utils import dkd_loss, hcl_loss, SAM, CenterLoss


## Fix Random Seed
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    # Cuda
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_resnet50_on_fashion_mnist(weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose(
        [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    all_set = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=transform)
    train_idx = [idx for idx in range(len(all_set)) if idx % 10!=0]
    valid_idx = [idx for idx in range(len(all_set)) if idx % 10==0]
    train_set = Subset(all_set, train_idx)
    valid_set = Subset(all_set, valid_idx)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                             shuffle=True, num_workers=4)
    
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=128,
                                             shuffle=False, num_workers=4)
    
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                             shuffle=False, num_workers=4)

    # ResNet-50 Model class
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.resnet50 = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, 10)

        def forward(self, x):
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)

            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet50.fc(x)
            return x

    teacher = ResNet().to(device)

    checkpoint = torch.load(weights_path)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()

    student = DistillModel_DepthSep().to(device)
    student.train()
    student_path = 'models/KD_depthsep_ASAM_epoch200_DKD.pth'
    EPOCHS = 200
    lr = 3e-4
    # optimizer = torch.optim.AdamW(student.parameters(), lr = lr)
    base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(student.parameters(), base_optimizer, lr=lr,adaptive=True, rho=2.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_hard = nn.CrossEntropyLoss(label_smoothing=0.2)
    # center_loss = CenterLoss(num_classes=10, feat_dim=256, use_gpu=True)
    dkd_alpha = 0.5
    dkd_beta = 0.5
    dkd_T = 1
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss = []
        train_CE_loss = []
        train_KL_loss = []
        train_center_loss = []
        correct = 0
        total = 0
        student.train()
        for data in tqdm(train_loader):
            # with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast():
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                for i in range(2):
                    teacher_outputs = teacher(images)
                    outputs, feature = student(images)
                    
                    loss_CE = criterion(outputs, teacher_outputs.softmax(dim=-1))
                    loss_CE_hard = criterion_hard(outputs, labels)
                    loss_KL = dkd_loss(outputs, teacher_outputs, labels, dkd_alpha, dkd_beta, dkd_T)
                    # loss_center = center_loss(feature, labels)
                    if epoch < 50:
                        loss = loss_CE + loss_CE_hard + loss_KL
                    elif epoch < 100:
                        loss = loss_CE + loss_CE_hard
                    else:
                        loss = loss_CE_hard
                    loss.backward()
                    if i == 0:
                        optimizer.first_step(zero_grad=True)
                        
                        train_loss.append(loss.item())
                        train_CE_loss.append(loss_CE.item())
                        train_KL_loss.append(loss_KL.item())
                        # train_center_loss.append(loss_center.item())
                    else:
                        optimizer.second_step(zero_grad=True)
                
                # scheduler.step()
                # optimizer.zero_grad()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_loss = sum(train_loss) / len(train_loss)
        train_CE_loss = sum(train_CE_loss) / len(train_CE_loss)
        train_KL_loss = sum(train_KL_loss) / len(train_KL_loss)
        # train_center_loss = sum(train_center_loss) / len(train_center_loss)
        accuracy = correct / total
        print(f"[ Train {epoch + 1:03d}/{EPOCHS:03d} ] Accuracy = {accuracy:.5f}, Loss = {train_loss:.5f}, CE Loss = {train_CE_loss:.5f}, KL Loss = {train_KL_loss:.5f}")
        
        valid_loss = []
        valid_CE_loss = []
        valid_KL_loss = []
        # valid_center_loss = []
        correct = 0
        total = 0
        student.eval()
        with torch.no_grad():
            for data in tqdm(valid_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                teacher_outputs = teacher(images)
                outputs, feature = student(images)
                
                loss_CE = criterion(outputs, teacher_outputs.softmax(dim=-1))
                loss_CE_hard = criterion_hard(outputs, labels)
                loss_KL = dkd_loss(outputs, teacher_outputs, labels, dkd_alpha, dkd_beta, dkd_T)
                # loss_center = center_loss(feature, labels)
                if epoch < 50:
                    loss = loss_CE + loss_CE_hard + loss_KL
                elif epoch < 100:
                    loss = loss_CE + loss_CE_hard
                else:
                    loss = loss_CE_hard
                
                valid_loss.append(loss.item())
                valid_CE_loss.append(loss_CE.item())
                valid_KL_loss.append(loss_KL.item())
                # valid_center_loss.append(loss_center.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_CE_loss = sum(valid_CE_loss) / len(valid_CE_loss)
        valid_KL_loss = sum(valid_KL_loss) / len(valid_KL_loss)
        # valid_center_loss = sum(valid_center_loss) / len(valid_center_loss)
        accuracy = correct / total
        print(f"[ Valid {epoch + 1:03d}/{EPOCHS:03d} ] Accuracy = {accuracy:.5f}, Loss = {valid_loss:.5f}, CE Loss = {valid_CE_loss:.5f}, KL Loss = {valid_KL_loss:.5f}")
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(student.state_dict(), student_path)
            print(f'Save model with accuracy = {best_acc:.5f}')
            
        test_loss = []
        test_CE_loss = []
        test_KL_loss = []
        # test_center_loss = []
        correct = 0
        total = 0
        student.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                teacher_outputs = teacher(images)
                outputs, feature = student(images)
                
                loss_CE = criterion(outputs, teacher_outputs.softmax(dim=-1))
                loss_CE_hard = criterion_hard(outputs, labels)
                loss_KL = dkd_loss(outputs, teacher_outputs, labels, dkd_alpha, dkd_beta, dkd_T)
                # loss_center = center_loss(feature, labels)
                if epoch < 50:
                    loss = loss_CE + loss_CE_hard + loss_KL
                elif epoch < 100:
                    loss = loss_CE + loss_CE_hard
                else:
                    loss = loss_CE_hard
                
                test_loss.append(loss.item())
                test_CE_loss.append(loss_CE.item())
                test_KL_loss.append(loss_KL.item())
                # test_center_loss.append(loss_center.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = sum(test_loss) / len(test_loss)
        test_CE_loss = sum(test_CE_loss) / len(test_CE_loss)
        test_KL_loss = sum(test_KL_loss) / len(test_KL_loss)
        # test_center_loss = sum(test_center_loss) / len(test_center_loss)
        accuracy = correct / total
        print(f"[ Test  {epoch + 1:03d}/{EPOCHS:03d} ] Accuracy = {accuracy:.5f}, Loss = {test_loss:.5f}, CE Loss = {test_CE_loss:.5f}, KL Loss = {test_KL_loss:.5f}")
    return accuracy


        
def main():
    same_seeds(63)
    teachert_path ="./resnet-50.pth"
    train_resnet50_on_fashion_mnist(teachert_path)
    print("----------------------------------------------")

if __name__ == "__main__":
    main()        