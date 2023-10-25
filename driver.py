import torch.optim as optim
from losses import ClassificationLoss, MomentumContrastiveLoss
from models import CVNetGlobal

# Hyperparameters from the paper
tau = 1/30
momentum = 0.999
lambda_cls = 0.5
lambda_con = 0.5
learning_rate = 0.005625
batch_size = 144 
num_epochs = 25

dataloader = None
num_classes = 1000
cvnet_global = CVNetGlobal(num_classes=num_classes)

# Instantiate the loss functions
classification_loss_fn = ClassificationLoss(num_classes=num_classes, tau=tau)
momentum_contrastive_loss_fn = MomentumContrastiveLoss(cvnet_global.queue, tau=tau)

# Define an optimizer. Only update the global backbone network.
optimizer = optim.SGD(cvnet_global.global_network.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch in dataloader:  # Assuming dataloader yields batches of input images and labels
        inputs, labels = batch

        # Forward pass through the network
        dq_g, dp_g = cvnet_global(inputs)
        
        # Calculate losses
        classification_loss = classification_loss_fn(dq_g, labels)
        momentum_contrastive_loss = momentum_contrastive_loss_fn(dq_g, dp_g, labels)
        
        # Total loss
        total_loss = lambda_cls * classification_loss + lambda_con * momentum_contrastive_loss
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        
        momentum_contrastive_loss_fn.clear_queue()
        
        # Perform momentum update to the momentum network manually
        # momentum_dict = {k: v.data for k, v in cvnet_global.global_network.state_dict().items()}
        # cvnet_global.momentum_network.load_state_dict(momentum_dict, strict=False)
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")
