import torch
from torch import nn
from torch.nn import BatchNorm1d, Conv1d
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class GAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes):
        super(GAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout_disac)
        self.conv2 = GATConv(hidden_dim * num_heads, 24, heads=1, concat=False, dropout=dropout_disac)
        self.conv3 = GATConv(num_node_features, 24, heads=num_heads, concat=False, dropout=dropout_disac)
        self.bn1 = BatchNorm1d(num_node_features)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)
        self.conv_normal = Conv1d(in_channels=24, out_channels=8, kernel_size=62)
        nn.init.kaiming_uniform_(self.conv3.lin.weight)
        nn.init.xavier_uniform_(self.conv3.att_src)
        nn.init.xavier_uniform_(self.conv3.att_dst)
        # self.fc = nn.Linear(out_features * num_features, out_features)

    def forward(self, band_data):
        x, edge_index, batch = band_data.x, band_data.edge_index, band_data.batch
        batch_size = batch.max().item() + 1
        # 第一层GAT卷积
        # x = F.dropout(x, p=0.6)
        x = self.bn1(x)
        # print('before gconv1:', x.shape)
        x = self.conv3(x, edge_index)
        # x = self.conv2(x, edge_index)
        x = F.elu(x)
        # print('after gconv1:', x.shape)  # (2976, hidden_dim)
        
        x = x.reshape(batch_size, 62*3, -1)
        
        x = x.permute(0, 2, 1)  # (batch_size, num_features, num_channels)
        # print(x.shape)
        x = self.conv_normal(x)
        # print('shape after conv1:', x.shape)
        # 第二层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.bn2(x)
        # x = self.conv2(x, edge_index)
        # x = F.elu(x)
        # print(x.shape)
        # x = self.conv3(x, edge_index)
        # 全局平均池化
        # x = global_mean_pool(x, batch)
        # print('after pool:', x.shape)

        return F.leaky_relu(x)


class SelfAttention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # 应用注意力得分
        return torch.matmul(attention_scores, V)
    

class MultiH_Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiH_Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm = nn.LayerNorm(embed_size)  # 添加 Layer Normalization 层

    def forward(self, x):
        # 注意力机制要求输入形式为 (seq_len, bs, embed_size)
        x = self.norm(x)
        x = x.permute(1, 0, 2)  # 转换输入为适合 PyTorch 注意力模块的维度
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # 将输出转换回原始的维度
        return attn_output
   

class FusionModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes, dataset):
        super(FusionModel, self).__init__()
        self.GAT_delta = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_alpha = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_beta = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                            dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_theta = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_gamma = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.dataset = dataset
        if self.dataset == "DEAP":
            self.fusion_gat = nn.Linear(4 * num_classes, num_classes, bias=True)
        elif self.dataset == "SEED":
            self.fusion_gat = MultiH_Attention(embed_size=125, heads=5)
        else:
            raise ValueError("Please give a dataset")
        # nn.init.kaiming_uniform_(self.fusion_gat.weight, nonlinearity='relu')
        # self.de_encoder = SelfAttention(310*3, 64)  # 62*5*3
        self.fusion_all = nn.Linear(40*125, num_classes)
        self.ln = nn.LayerNorm(310*3)

    def forward(self, data):
        x_alpha = self.GAT_alpha(data['alpha'])
        x_beta = self.GAT_beta(data['beta'])
        x_gamma = self.GAT_theta(data['theta'])
        x_theta = self.GAT_gamma(data['gamma'])
        if self.dataset == "DEAP":
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        elif self.dataset == "SEED":
            x_delta = self.GAT_delta(data['delta'])
            x_concat = torch.cat((x_delta, x_alpha, x_beta, x_gamma, x_theta), dim=1)
            # print(data['alpha'], x_delta.shape, x_concat.shape)
            x_concat = self.fusion_gat(x_concat)
            x_concat = x_concat.reshape(x_concat.size(0), -1)
        else:
            print('[Attention]!!!')
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        
        # x_out = self.fusion_gat(x_concat)  # bs, 64
        # x_out = F.elu(x_out)
        # print(data['de'].shape)
        # new_de_shape = (-1, data['de'].size(-3) * data['de'].size(-2) * data['de'].size(-1))
        # de_feats = data['de'].view(*new_de_shape)
        # de_feats = self.ln(de_feats)
        # x_de = self.de_encoder(de_feats)  # bs, 64
        # x_de = F.leaky_relu(x_de, negative_slope=1)
        # print(x_de.shape)
        # x_out = self.fusion_all(torch.cat((x_out, x_de), dim=1))
        x_out = self.fusion_all(x_concat)
        return F.leaky_relu(x_out)


class MultiBandDataset(Dataset):
    def __init__(self, constructed):
        super(MultiBandDataset, self).__init__()
        self.constructed = constructed

    def __len__(self):
        return len(self.constructed['label'])

    def __getitem__(self, idx):
        band_list = list(self.constructed.keys())
        band_list.remove("label")
        sample = {band: self.constructed[band][idx] for band in band_list}
        label = self.constructed['label'][idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        sample['label'] = label
        return sample


def train(model, tr_loader, optimizer, criterion, device, max_grad):
    model.train()
    for training_data in tr_loader:
        # print(data)
        # print('train labels:', training_data['label'])
        labels = training_data['label'].to(device)
        training_data = {key: value.to(device) for key, value in training_data.items() if key != 'label'}
        optimizer.zero_grad()  # 清空梯度
        out = model(training_data)  # 前向传播
        # print(out.shape, labels.shape)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optimizer.step()


def evaluate(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_loss = []
    with torch.no_grad():
        for testing_data in data_loader:
            # print('test labels:', testing_data['label'])
            labels = testing_data['label'].to(device)
            testing_data = {key: value.to(device) for key, value in testing_data.items() if key != 'label'}
            outputs = model(testing_data)
            loss = criterion(outputs, labels).item()
            # print(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_loss.append(loss)
            # print(loss)
    all_loss = sum(all_loss) / len(all_loss)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1, all_loss


if __name__ == '__main__':
    def print_model_details(model, indent=0):
        for name, module in model.named_children():
            print('    ' * indent + f'{name}: {module}')
            if len(list(module.children())) > 0:
                print_model_details(module, indent + 1)


    model = FusionModel(num_node_features=200, hidden_dim=128, num_heads=4, dropout_disac=0.6, num_classes=3,
                        dataset='SEED')
    print_model_details(model)
