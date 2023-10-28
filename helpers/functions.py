import os, torch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from losses import ClassificationLoss, MomentumContrastiveLoss
from datasets import PairedDataset

def load_weights(model, weights_file):
    if os.path.isfile(weights_file):
        print ('loading weight file')
        weight_dict = torch.load(weights_file, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    print('    loaded: ' + name)
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')
    return model


def train_backbone(model, dataset, num_classes, num_epochs, batch_size, learning_rate, device, **kwargs):
    tau = kwargs.get('tau', 1/30)
    lambda_cls = kwargs.get('lambda_cls', 0.5)
    lambda_con = kwargs.get('lambda_con', 0.5)
    save = kwargs.get('save', True) 
    progress = kwargs.get('progress', True)   
    
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

