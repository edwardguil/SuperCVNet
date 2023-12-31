import argparse, torch, os
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
from models import SuperGlobal, CVNetGlobal
from datasets import GoogleLandMarks, PairedDataset
from losses import ClassificationLoss, MomentumContrastiveLoss
import torchvision.transforms as transforms

def train_backbone(model, dataset, num_classes, num_epochs, batch_size, learning_rate, device, **kwargs):
    tau = kwargs.get('tau', 1/30)
    lambda_cls = kwargs.get('lambda_cls', 0.5)
    lambda_con = kwargs.get('lambda_con', 0.5)
    save = not kwargs.get('no_save', False) 
    progress = not kwargs.get('no_progress', False)   
    
    paired_dataset = PairedDataset(dataset)
    paired_trainloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    classification_loss_fn = ClassificationLoss(num_classes, tau=tau)
    momentum_contrastive_loss_fn = MomentumContrastiveLoss(num_classes, tau=tau)

    classification_loss_fn.to(device)
    momentum_contrastive_loss_fn.to(device)

    optimizer = optim.SGD(model.global_network.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        avg_cls_loss = 0
        avg_con_loss = 0
        for i, data in enumerate(paired_trainloader):
            inputs, positive_inputs, labels = data
            inputs, positive_inputs, labels = inputs.to(device), positive_inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            global_features, momentum_features = model(inputs, positive_inputs)

            loss_cls = classification_loss_fn(global_features, labels)
            loss_con = momentum_contrastive_loss_fn(global_features, momentum_features, labels)

            loss = lambda_cls * loss_cls + lambda_con * loss_con
            loss.backward()

            # To prevent vanishing gradients
            torch.nn.utils.clip_grad_norm_(model.global_network.parameters(), max_norm=15)

            optimizer.step()

            momentum_dict = {k: v.data for k, v in model.global_network.state_dict().items()}
            model.momentum_network.load_state_dict(momentum_dict)

            # Save info
            avg_cls_loss += loss_cls.item()
            avg_con_loss += loss_con.item()
            if progress:
                print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {loss_cls:.4f}, Contrast Loss: {loss_con:.4f}', end='\r')
        
        # Print info
        if progress:
            print(f'Epoch [{epoch+1}/{num_epochs}], Class Loss: {avg_cls_loss/(i + 1):.4f}, Contrast Loss: {avg_con_loss/(i + 1):.4f}', end='\r')
            print('')

        if save:
            if not os.path.exists('./weights'):
                os.makedirs('./weights')
            torch.save(model.state_dict(), os.path.join("./weights", f'{model.__class__.__name__}{dataset.__class__.__name__}-{epoch}.pkl'))

    print('Finished Training')


def get_args():
    parser = argparse.ArgumentParser(description='Training script for model backbones')

    parser.add_argument('--model', type=str, default='CVNet', choices=['CVNet'], help='Name of the model to train')
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['GoogleLandmarks', 'Cifar10'], help='Name of the dataset to train on')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--tau', type=float, default=1/30, help='Tau for losses')
    parser.add_argument('--lambda_cls', type=float, default=0.5, help='Lambda for classification loss')
    parser.add_argument('--lambda_con', type=float, default=0.5, help='Lambda for contrastive loss')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momemtum for contrastive loss')
    parser.add_argument('--reduction_dim', type=int, default=2048, help='Reduction dim size (imbedding size)')
    parser.add_argument('--resnet_depth', type=int, default=50, help='The depth of the underlying ResNet model')
    parser.add_argument('--no_save', action='store_true', default=False, help="Don't save the weights during training")
    parser.add_argument('--no_progress', action='store_true', default=False, help="Don't to print training progress")

    return parser.parse_args()

def launch_script(args):
    if args.model.lower() == 'CVNet'.lower():
        model = CVNetGlobal(args.reduction_dim, args.resnet_depth, args.momentum)

    transform = transforms.Compose([
        transforms.Resize(512),  # Resize the images to 512x512 (multiples of 64)
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    if args.dataset == 'Cifar10':
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'GoogleLandmarks':
        dataset = GoogleLandMarks(root='./data', split='train', remove_tars=False, verify=False)
        num_classes = 1000
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_backbone(model=model, 
                dataset=dataset, 
                num_epochs=args.num_epochs, 
                num_classes=num_classes,
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate, 
                device=device,
                tau=args.tau, 
                lambda_cls=args.lambda_cls, 
                lambda_con=args.lambda_con,
                no_save=args.no_save,
                no_progress=args.no_progress)

if __name__ == '__main__':
    args = get_args()
    launch_script(args)