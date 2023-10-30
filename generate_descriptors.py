import torch, argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from models import CVNetGlobal, SuperCVNetGlobal
from models.helpers import load_weights
from datasets import GoogleLandMarks
from helpers import PineconeIndex



def generate_descriptors(model, dataset, batch_size, device, no_csv, to_pinecone):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_descriptors = []
    all_labels = []
    print("Generating Descriptors...", '\n')
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to(device)

            global_features = model(inputs)

            global_features = torch.nn.functional.normalize(global_features, p=2, dim=1)
 
            all_descriptors.extend(global_features.cpu().tolist())
            all_labels.extend(labels.tolist())
            print(f"    Set: {i*batch_size}/{batch_size*len(dataset)}", end="\r")
    

        vec_ids = [f"vec{i}" for i in range(len(all_labels))]

        df = pd.DataFrame({
            'vecid': vec_ids,
            'vectors': all_descriptors,
            'labels': all_labels
        })

        if not no_csv:
            df.to_csv('descriptors.csv', index=False)

        if to_pinecone:
            print("Sending to pinecone...")
            index = PineconeIndex()
            output = [(row['vecid'], row['vectors'], {"label": str(int(row['labels']))}) for _, row in df.iterrows()]
            index.upsert_vectors(output)
    

def get_args():
    parser = argparse.ArgumentParser(description='Script for generating global descriptors backbones')

    parser.add_argument('--model', type=str, default='CVNet', help='Name of the model to train')
    parser.add_argument('--dataset', type=str, default='Cifar10', help='Name of the dataset to train on')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--reduction_dim', type=int, default=2048, help='Reduction dim for resnet')
    parser.add_argument('--weights', type=str, default='weights.pth', help='Weight file')
    parser.add_argument('--no_csv', action='store_true', default=False, help="Don't save the label/descriptor pairs to a CSV.")
    parser.add_argument('--to_pinecone', action='store_true', default=False, help="Send the descriptors to pinecone index.")

    return parser.parse_args()


def launch_script(args):
    if args.model.lower() == 'CVNet'.lower():
        model = CVNetGlobal(reduction_dim=args.reduction_dim)  # you might need to customize the initialization
    elif args.model.lower() == 'SuperCVNet'.lower():
        model = SuperCVNetGlobal(reduction_dim=args.reduction_dim)
    else:
        raise ValueError(f"Model {args.model} dosen't exist. Only: 'SuperCVNet' or 'CVNet'")

    transform = transforms.Compose([
        transforms.Resize(512),  # Resize the images to 512x512 (multiples of 64)
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    if args.dataset.lower() == 'Cifar10'.lower():
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset.lower() == 'GoogleLandmarks'.lower():
        dataset = GoogleLandMarks(root='./data', split='train', remove_tars=False, verify=False)
        num_classes = 1000
    else:
        raise ValueError(f"Dataset {args.dataset} dosen't exist. Only: 'Cifar10' or 'GoogleLandmarks'")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load_weights(model, args.weights)
    
    model.to(device)

    generate_descriptors(model=model.global_network, 
                dataset=dataset, 
                batch_size=args.batch_size,
                device=device,
                no_csv=args.no_csv,
                to_pinecone=True # to_pinecone = args.to_pinecone
    )

if __name__ == '__main__':
    argv = get_args()
    launch_script(argv)