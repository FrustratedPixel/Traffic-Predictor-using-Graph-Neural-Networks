import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean, scatter_sum, scatter_max

class GraphConv(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info,
        aggregation_type="mean",
        combination_type="concat",
        activation=None,
    ):
        super(GraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        # Initializing weights using Xavier uniform (similar to Glorot uniform in TensorFlow)
        self.weight = nn.Parameter(torch.empty(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight)

        # Activation function if provided
        if activation:
            self.activation = getattr(F, activation)
        else:
            self.activation = None

    def compute_nodes_representation(self, features: torch.Tensor):
        """
        Computes each node's representation.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        print(f"features shape: {features.shape}, weight shape: {self.weight.shape}")
        print(f"features device: {features.device}")
        return torch.matmul(features, self.weight)
    
    def aggregate(self, neighbour_representations: torch.Tensor, edge_index: torch.Tensor):
        """
        Aggregates neighbor representations.

        Args:
            neighbour_representations: Tensor of neighbor node representations.
            edge_index: Tensor containing the source and target edges.

        Returns:
            Aggregated messages.
        """

        edge_index = edge_index.to(neighbour_representations.device)

        print(f"edge_index device: {edge_index.device}")

        aggregation_func = {
            "sum": scatter_sum,
            "mean": scatter_mean,
            "max": scatter_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                edge_index[0],  # Aggregating based on source nodes
                dim=0,  # Aggregating along node dimension
                dim_size=self.graph_info.num_nodes  # Total number of nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_aggregated_messages(self, features: torch.Tensor, edge_index: torch.Tensor):
        """
        Compute aggregated messages for each node from its neighbors.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`
            edge_index: Tensor containing the source and target edges.

        Returns:
            Aggregated messages after applying graph convolution.
        """
        # Gather neighbor representations from the features using the target nodes (edges[1])
        neighbor_representations = features[edge_index[1]]  # Gathering neighbors' features
        aggregated_messages = self.aggregate(neighbor_representations, edge_index)
        return torch.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: torch.Tensor, aggregated_messages: torch.Tensor):
        """
        Update node representations by combining them with aggregated messages.

        Args:
            nodes_representation: Tensor of node representations.
            aggregated_messages: Tensor of aggregated messages from neighbors.

        Returns:
            Updated node representations.
        """
        if self.combination_type == "concat":
            h = torch.cat([nodes_representation, aggregated_messages], dim=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply activation if provided
        if self.activation:
            h = self.activation(h)

        return h

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass through the GraphConv layer.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`
            edge_index: Tensor of shape `(2, num_edges)` containing graph edges.

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """

        edge_index = edge_index.to(features.device)
        # Compute node representations
        nodes_representation = self.compute_nodes_representation(features)
        
        # Compute aggregated messages from neighbors
        aggregated_messages = self.compute_aggregated_messages(features, edge_index)
        
        # Update node representations
        return self.update(nodes_representation, aggregated_messages)
    
