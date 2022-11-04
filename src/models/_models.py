import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))
    
def acc(real: list, predict: list) -> float:
    return accuracy_score(real, predict)

def confusion_mat(real: list, predict: list):
    return confusion_matrix(real, predict, normalize='true')


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta):
        self.beta = beta
        super(SmoothL1Loss,self).__init__()

    def forward(self, x, y):
        criterion = nn.SmoothL1Loss(beta=self.beta)
        loss = criterion(x, y)
        return loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight):
        self.weight = weight
        super(CrossEntropyLoss,self).__init__()

    def forward(self, x, y):
        criterion = nn.CrossEntropyLoss(weight=self.weight)
        loss = criterion(x, y)
        return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ExpectationLoss(nn.Module):
    """
    Expectation Loss definition
    """

    def __init__(self):
        super(ExpectationLoss, self).__init__()

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() and cfg['use_gpu'] else 'cpu')
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        cls = torch.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float).T).cuda()
        # cls = torch.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float).T).to(self.device)
        return self.mae(torch.mm(pred, cls.float()).view(-1), target)

class CategoryLoss(nn.Module):
    def __init__(self):
        super(CategoryLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, pred, target):
        # print(pred.shape, target.shape)
        pred1 = torch.sum(pred[:, :5], dim  = -1, keepdim = True)
        pred2 = torch.sum(pred[:, 6:9], dim  = -1, keepdim = True)
        pred3 = torch.sum(pred[:, 9:], dim  = -1, keepdim = True)
        pred = torch.cat([pred1, pred2, pred3], dim = 1)
        target_onehot = F.one_hot(target)
        target1 = torch.sum(target_onehot[:, :5], dim = -1, keepdim = True)
        target2 = torch.sum(target_onehot[:, 6:9], dim = -1, keepdim = True)
        target3 = torch.sum(target_onehot[:, 9:], dim = -1, keepdim = True)
        target = torch.cat([target1, target2, target3], dim = 1)
        # print(target.shape)
        # print(target)
        target = target.argmax(dim = 1)
        return self.ce_loss(pred, target)
        
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.clf_loss = LabelSmoothingLoss(classes = 10, smoothing = 0.1)
        self.expectation_loss = ExpectationLoss()
        self.category_loss = CategoryLoss()

    def forward(self, pred, target):
        return 2 * self.clf_loss(pred, target) + self.expectation_loss(pred, target) + self.category_loss(pred, target)

class PIECELoss(torch.nn.Module):
    def __init__(self):
        super(PIECELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        errorGreater = torch.greater((x-y), 0.3)
        errorLess = torch.less_equal((x-y),-0.7)
        tmp = (errorGreater|errorLess)
        tmp = tmp.type(torch.FloatTensor).to(device='cuda')
        loss = torch.mean(torch.square(tmp*2*(x-y)))
        return loss

class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum:bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachine_v(nn.Module):

    def __init__(self, input_dim, latent_dim, classifier):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        if classifier:
            self.linear = nn.Linear(input_dim, 10, bias = False)
        else:
            self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FeaturesLinear(nn.Module):

    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias

class _FactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int, last_dim=1):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

        # 클래시파이어 수정 부분
        self.last_dim = last_dim
        if last_dim != 1:
            self.last_classifier = nn.Linear(1, last_dim)

        
    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        # 클래시파이어 수정 부분
        if self.last_dim != 1:
            x = self.last_classifier(x)
        return x.squeeze(1)

class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets, dtype= np.long).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix

class _FieldAwareFactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class _NeuralCollaborativeFiltering(nn.Module):

    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout, last_dim=1):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + 2 * embed_dim, 1)

        # 클래시파이어 수정 부분
        self.last_dim = last_dim
        if last_dim != 1:
            self.last_classifier = nn.Linear(1, last_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        context_x = x[:, 2:, :]
        context_x = torch.sum(context_x, dim = 1)
        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))
        x = torch.cat([gmf, x, context_x], dim=1)

        # 클래시파이어 수정 부분        
        x = self.fc(x)

        if self.last_dim != 1:
            x = self.last_classifier(x)

        return x.squeeze(1)

class _WideAndDeepModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.linear = FeaturesLinear(field_dims[:2])
        self.embedding = FeaturesEmbedding(field_dims[:2], embed_dim)
        self.context_embedding = FeaturesEmbedding(field_dims[2:], embed_dim)
        self.embed_output_dim = len(field_dims[:2]) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.context_embed_output_dim = len(field_dims[2:]) * embed_dim
        self.context_mlp = MultiLayerPerceptron(self.context_embed_output_dim, mlp_dims, dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # print(f"[WDN MODEL DATA INPUT SHAPE]\n {x.shape}")
        # embed_x = self.embedding(x)
        embed_x = self.embedding(x[:,:2])
        context_embed_x = self.context_embedding(x[:,2:])
        x = x[:, :2]
        linear_out = self.linear(x)
        mlp_out = self.mlp(embed_x.view(-1, self.embed_output_dim))
        context_mlp_out = self.context_mlp(context_embed_x.view(-1, self.context_embed_output_dim))
        x = linear_out + mlp_out + context_mlp_out
        return x.squeeze(1)

class CrossNetwork(nn.Module):

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class _DeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims: np.ndarray, embed_dim: int, num_layers: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims[:2], embed_dim)
        self.embed_output_dim = len(field_dims[:2]) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.cd_linear = nn.Linear(embed_dim * 2, 1, bias=False)
        
        self.context_embedding = FeaturesEmbedding(field_dims[2:], embed_dim)
        

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x[:,:2]).view(-1, self.embed_output_dim)
        context_embed_x = self.context_embedding(x[:,2:])
        context_embed_x = torch.sum(context_embed_x, dim = 1)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        x_concat = torch.cat([x_out, context_embed_x], dim = 1)
        p = self.cd_linear(x_concat)
        return p.squeeze(1)
