import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import copy

import torch
from torch.utils.tensorboard import SummaryWriter


class NetP(nn.Module):
    def __init__(
        self,
        n_chars ,
        hidden,
        seq_len,
    ):
        super().__init__()
        assert seq_len == 20
        self.convs = nn.Sequential(
            nn.Conv2d(n_chars, 96, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(96, 128, kernel_size=(1, 4)),
            nn.ReLU(),
        )  # [50, 128, 1, 6]
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x, latent=False):
        # x (bs,20,48)
        x = x.float()
        x = x.permute(0, 2, 1)[:, :, None]
        # x (bs, 48, 1, 20)
        x = self.convs(x)
        # #[50, 128, 1, 6]
        x = x.reshape([x.shape[0], 128 * 6])
        latent_tensor = self.fc1(x)
        x = self.fc2(latent_tensor)
        if latent:
            return x, latent_tensor
        else:
            return x
        
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1  :
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)
    

    
class NetP_CNN(nn.Module):
    def __init__(self, n_chars, seq_len, hidden):
        super().__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input,latent=False):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        latent_tensor = output
        output = self.linear(latent_tensor)
        if latent:
            return output, latent_tensor
        else:
            return output
        

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1  :
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


def train_regression_model(
    train_loader,
    valid_loader,
    net,
    loss_fn,
    optimizer,
    num_epochs=10,
    use_tensorboard=True,
    tensorboard_path="logs",
    early_stopping_patience=None,
):
    # Initialize TensorBoard SummaryWriter if requested
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Initialize variables for early stopping
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # Training loop for regression
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        total_samples_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)

        average_train_loss = total_train_loss / total_samples_train

        # Validation
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            for batch_x, batch_y in valid_loader:
                outputs = net(batch_x)
                loss = loss_fn(outputs, batch_y)
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

            average_valid_loss = total_valid_loss / total_samples_valid

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Validation Loss: {average_valid_loss:.4f}"
            )

            # Write to TensorBoard if requested
            if use_tensorboard:
                writer.add_scalar("Train Loss", average_train_loss, epoch)
                writer.add_scalar("Validation Loss", average_valid_loss, epoch)

            # Early Stopping
            if (
                early_stopping_patience is not None
                and average_valid_loss < best_valid_loss
            ):
                best_valid_loss = average_valid_loss
                patience_counter = 0
                best_weights = copy.deepcopy(net.state_dict())  # Record the best weights
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best weights back to the net
    if best_weights is not None:
        net.load_state_dict(best_weights)

    # Close the TensorBoard SummaryWriter if used
    if use_tensorboard:
        writer.close()



def train_regression_model_with_weight(
    train_loader,
    valid_loader,
    net,
    loss_fn,
    optimizer,
    device = 'cpu',
    num_epochs=10,
    use_tensorboard=True,
    tensorboard_path="logs",
    early_stopping_patience=None,
):
    # Initialize TensorBoard SummaryWriter if requested
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Initialize variables for early stopping
    best_valid_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # Training loop for regression
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        total_samples_train = 0
        for batch_x, batch_y, batch_w in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = loss_fn(outputs, batch_y, batch_w)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_y.size(0)
            total_samples_train += batch_y.size(0)

        average_train_loss = total_train_loss / total_samples_train

        # Validation
        net.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_samples_valid = 0
            for batch_x, batch_y, batch_w in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                outputs = net(batch_x)
                loss = loss_fn(outputs, batch_y, batch_w)
                total_valid_loss += loss.item() * batch_y.size(0)
                total_samples_valid += batch_y.size(0)

            average_valid_loss = total_valid_loss / total_samples_valid

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.5f}, Validation Loss: {average_valid_loss:.5f}"
            )

            # Write to TensorBoard if requested
            if use_tensorboard:
                writer.add_scalar("Train Loss", average_train_loss, epoch)
                writer.add_scalar("Validation Loss", average_valid_loss, epoch)

            # Early Stopping
            if (
                early_stopping_patience is not None
                and average_valid_loss < best_valid_loss - 1e-5
            ):
                best_valid_loss = average_valid_loss
                patience_counter = 0
                best_weights = copy.deepcopy(net.state_dict())  # Record the best weights
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best weights back to the net
    if best_weights is not None:
        net.load_state_dict(best_weights)

    # Close the TensorBoard SummaryWriter if used
    if use_tensorboard:
        writer.close()


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 重写上面的函数 要求针对不同样本 有权重
def train_net_p_with_weight(cfg,net,x,y,weights,lr=0.001):
    # Example usage
    x_train, x_valid, y_train, y_valid,weights_train,weights_valid = train_test_split(x, y,weights, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train,weights_train),
                                batch_size=cfg.batch_size_p, shuffle=True,
                            )
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid,weights_valid), 
                              batch_size=cfg.batch_size_p, shuffle=False)

    # 带权重的loss
    def weighted_mse_loss(input, target, weights):
        out = (input - target)**2
        out = out * weights.expand_as(out)
        loss = out.mean()
        return loss

    # Create your loss function, and optimizer
    # loss_fn = torch.nn.MSELoss()
    loss_fn = weighted_mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    train_regression_model_with_weight(train_loader, valid_loader, net, 
                           loss_fn, optimizer, device=cfg.device,
                           num_epochs=cfg.num_epochs_p, use_tensorboard=False, 
                           tensorboard_path='logs', early_stopping_patience=cfg.es_p)

