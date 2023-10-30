import os, torch

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

