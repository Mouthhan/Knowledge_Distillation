import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import DistillModel
from torchsummary import summary
import pandas as pd
from tqdm.auto import tqdm


def test_resnet18_on_fashion_mnist(weights_path):
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # transform
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),  # gray to 3 channel
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # load Fashion MNIST
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)

    # My student model
    

    net = DistillModel().to(device)

    # load checkpoint
    checkpoint = torch.load(weights_path)
    net.load_state_dict(checkpoint)
    net.eval()

    
    summary(net, (3, 28, 28))

    # test 
    correct = 0
    total = 0
    pred_arr = []
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_arr.append(predicted.item())

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")

    pred_data = {"pred":pred_arr}
    df_pred = pd.DataFrame(pred_data)
    df_pred.to_csv('example_pred.csv', index_label='id')

    return accuracy


        
def main():
    
    ckpt_path ="models/KD_baseline_autocast_onecycle_noaugment.pth"
    # print("Epoch {}:".format(6))
    test_resnet18_on_fashion_mnist(ckpt_path)
    

if __name__ == "__main__":
    main()        
