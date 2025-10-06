
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def spmm(graph, x, actnn=False, fast_spmm=None, fast_spmm_cpu=None):
    if fast_spmm is None:
        initialize_spmm()
        fast_spmm = CONFIGS["fast_spmm"]
    if fast_spmm_cpu is None:
        initialize_spmm_cpu()
        fast_spmm_cpu = CONFIGS["fast_spmm_cpu"]
    if fast_spmm is not None and str(x.device) != "cpu":
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        if x.dtype == torch.half:
            csr_data = csr_data.half()
        x = fast_spmm(row_ptr.int(), col_indices.int(), x, csr_data, graph.is_symmetric(), actnn=actnn)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    elif fast_spmm_cpu is not None and str(x.device) == "cpu" and x.requires_grad is False:
        if graph.out_norm is not None:
            x = graph.out_norm * x

        row_ptr, col_indices = graph.row_indptr, graph.col_indices
        csr_data = graph.raw_edge_weight
        x = fast_spmm_cpu(row_ptr.int(), col_indices.int(), csr_data, x)

        if graph.in_norm is not None:
            x = graph.in_norm * x
    else:
        row, col = graph.edge_index
        x = spmm_scatter(row, col, graph.edge_weight, x)
    return x


def spmm_scatter(row, col, values, b):
    r"""
    Args:
        (row, col): Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        b : Tensor, shape=(N, d)
    """
    output = b.index_select(0, col) * values.unsqueeze(-1).to(b.dtype)
    output = torch.zeros_like(b).scatter_add_(0, row.unsqueeze(-1).expand_as(output), output)
    return output


def initialize_spmm_cpu():
    if CONFIGS["spmm_cpu_flag"]:
        return
    CONFIGS["spmm_cpu_flag"] = True

    from cogdl.operators.spmm import spmm_cpu

    CONFIGS["fast_spmm_cpu"] = spmm_cpu


def initialize_spmm():
    if CONFIGS["spmm_flag"]:
        return
    CONFIGS["spmm_flag"] = True
    if torch.cuda.is_available():
        from cogdl.operators.spmm import csrspmm

        CONFIGS["fast_spmm"] = csrspmm

def orthogonal_loss(output_task_induced, output_view_specific):
    _shared = output_task_induced.detach()
    _shared = _shared - _shared.mean(dim=0)
    correlation_matrix = _shared.t().matmul(output_view_specific)
    norm = torch.norm(correlation_matrix, p=1)
    return norm

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, n, activation=None):
        super(GraphConvSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation
        self.num_views = num_views
        self.bias = nn.Parameter(torch.randn(output_dim,), requires_grad=True)


    def forward(self, inputs, adj, fea_sp=False):
        x = inputs
        if fea_sp:
            x = torch.spmm(x, self.weight)
        else:
            x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x) #+ self.bias
        if self.activation is None:
            return x
        else:
            return self.activation(x)

class GCN(nn.Module):
    def __init__(self, device, hidden_dims, num_views, dropout, n):
        super(GCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.num_views = num_views
        self.gc1 = GraphConvSparse(self.hidden_dims[0], self.hidden_dims[1], self.num_views, n)
        self.gc2 = GraphConvSparse(self.hidden_dims[1], self.hidden_dims[2], self.num_views, n)

    def forward(self, input, adj):
        hidden = F.relu(self.gc1(input, adj.to(self.device)))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        output = self.gc2(hidden, adj.to(self.device))
        return output


class Extractor(nn.Module):
    def __init__(self, input_dim, output_dim, num_views, activation=F.tanh):
        super(Extractor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.num_views = num_views
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(output_dim, output_dim)


    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        return x

class Disentangler(nn.Module):
    def __init__(self, temp_dims, num_views):
        super(Disentangler, self).__init__()
        self.hidden_dims = temp_dims
        self.num_views = num_views
        self.invariant_extractor = Extractor(self.hidden_dims[0], self.hidden_dims[1], self.num_views)
        self.specific_extractor = Extractor(self.hidden_dims[0], self.hidden_dims[1], self.num_views)

    def forward(self, feature):
        invariant_fea = self.invariant_extractor(feature)
        specific_fea = self.specific_extractor(feature)
        return invariant_fea, specific_fea

class Our(nn.Module):
    def __init__(self, n, input_dims, num_classes, dropout, hdim, device):
        super(Our, self).__init__()
        self.device = device
        self.input_dims = input_dims
        self.num_views = len(input_dims)
        self.mv_disentangler_module = nn.ModuleList()
        self.mv_reconstruct_module = nn.ModuleList()
        self.Classifier = nn.ModuleList()
        self.Obfuscator = nn.ModuleList()

        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(input_dims[i])
            temp_dims.append(hdim[0])
            temp_dims.append(num_classes)
            self.mv_disentangler_module.append(Disentangler(temp_dims, self.num_views))
            self.mv_reconstruct_module.append(Extractor(temp_dims[1], temp_dims[0], self.num_views))
            self.Classifier.append(GCN(device, hidden_dims=[hdim[0], hdim[0] // 2, num_classes], num_views=self.num_views, dropout=dropout, n=n))
            self.Obfuscator.append(GCN(device,hidden_dims=[hdim[0], hdim[0] // 2, num_classes], num_views=self.num_views, dropout=dropout, n=n))

    def forward(self, fea_list, adj_list):
        invariant_fea_list = []
        specific_fea_list = []
        MI_loss_list = []
        rec_loss_list = []
        for i in range(self.num_views):
            invariant_fea, specific_fea = self.mv_disentangler_module[i](fea_list[i])

            invariant_fea_list.append(invariant_fea)
            specific_fea_list.append(specific_fea)

            MI_loss = orthogonal_loss(invariant_fea, specific_fea)
            MI_loss_list.append(MI_loss)

            rec_fea = self.mv_reconstruct_module[i](invariant_fea + specific_fea)
            loss_rec = torch.norm(fea_list[i] - rec_fea, p='fro') ** 2
            rec_loss_list.append(loss_rec)

        output_view_invariant_list = []
        output_view_specific_list = []
        for i in range(self.num_views):
            output_task_induced = self.Classifier[i](invariant_fea_list[i], adj_list[i])
            output_view_specific = self.Obfuscator[i](specific_fea_list[i], adj_list[i])
            output_view_invariant_list.append(output_task_induced)
            output_view_specific_list.append(output_view_specific)


        return MI_loss_list, rec_loss_list, output_view_invariant_list, output_view_specific_list


