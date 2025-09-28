from torch_geometric.data import Data, Dataset
import torch


class KHopSubgraphDataset(Dataset):
    def __init__(self, subgraphs, original_data):
        super().__init__()
        self.subgraphs = subgraphs
        self.original_data = original_data

    def len(self):
        return len(self.subgraphs)

    def get(self, idx):
        subgraph = self.subgraphs[idx]
        nodes, edge_index = subgraph["nodes"], subgraph["edge_index"]

        # Create a new Data object for the subgraph
        data = Data(
            x=self.original_data.x[nodes],
            edge_index=edge_index,
            nodes=nodes,
            user=subgraph["user"],
            item=subgraph["item"],
            community_labels=subgraph["community_labels"],
        )
        return data


class DataMapper:
    def __init__(self, pt_file_path, debug_mode=False):
        self.debug_mode = debug_mode
        self.original_data = torch.load(pt_file_path)
        print(f"Loaded original data with {len(self.original_data.raw_texts)} texts.")
        self.num_users = len(self.original_data.user_id_to_node)

    def get_user_raw_text(self, dgl_user_id):
        # print(f'Getting raw text for user id: {dgl_user_id}')
        data = None 
        try:
            print(f'dgl user id: {dgl_user_id}')
            map_user_id = self.original_data.user_id_to_node[dgl_user_id]
            data = self.original_data.raw_texts[map_user_id]
        except IndexError as e:
            print(f"IndexError: {e}. dgl_user_id: {dgl_user_id}, num_users: {self.num_users}")
        return data 

    def get_item_title(self, dgl_item_id):
        return self.original_data.item_titles[dgl_item_id]

    def get_item_raw_text(self, dgl_item_id):
        if self.debug_mode:
            print(
                f"id sum: {dgl_item_id + self.num_users}, num_users: {self.num_users}, dgl_item_id: {dgl_item_id}"
            )
        return self.original_data.raw_texts[dgl_item_id + self.num_users]
