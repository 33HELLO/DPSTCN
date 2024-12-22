#encoding=utf-8
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance") -> np.array:
    """
    construct adjacent matrix by csv file
    :param distance_file: path of csv file to save the distances between nodes
    :param num_nodes: number of nodes in the graph
    :param id_file: path of txt file to save the order of the nodes
    :param graph_type: ["connect", "distance"] if use weight, please set distance
    :return:
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")

        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    return A


def get_flow_data(flow_file: str) -> np.array:
    """
    parse npz to get flow data
    :param flow_file: (N, T, D)
    :return:
    """
    data = np.load(flow_file)
    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0] # [N, T, D]  D = 1
    
    #print("flow",np.isnan(flow_data).any())
    #flow_data[np.where(flow_data == 0)] = 1
    return flow_data


class PEMSDataset(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, predit_length, train_mode):
        """
        load processed data
        :param data_path: ["graph file name" , "flow data file name"]
        :param num_nodes: number of nodes in graph
        :param divide_days: [ days of train data, days of test data], list to divide the original data
        :param time_interval: time interval between two traffic data records (mins)
        :param history_length: length of history data to be used
        :param train_mode: ["train", "test"]
        """
        self.predit_length = predit_length
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.history_length = history_length  # 12
        self.total_length = self.history_length + self.predit_length
        self.time_interval = time_interval  # 5 min
        self.one_day_length = int(24 * 60 / self.time_interval)
        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes) 
        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1)
        



    def __len__(self):
        if self.train_mode == "train":
            
            return self.train_days * self.one_day_length - self.total_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length - self.total_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        if self.train_mode == "train":
            index = index
            tmp_wl, tmp_dl = PEMSDataset.get_day_week_cyc(self.history_length, 0, index)
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
            tmp_wl, tmp_dl = PEMSDataset.get_day_week_cyc(self.history_length, self.train_days, index)
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))
        
        data_x, data_y = PEMSDataset.slice_data(self.flow_data, self.history_length, self.predit_length, index, self.train_mode)
        data_x = PEMSDataset.to_tensor(data_x)  # [N, H, D]
        data_y = PEMSDataset.to_tensor(data_y) # [N, 1, D]


        return {"graph": PEMSDataset.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y, "week_cyc": PEMSDataset.to_tensor(tmp_wl), "day_cyc": PEMSDataset.to_tensor(tmp_dl)}

    @staticmethod
    def slice_data(data, history_length, predict_length, index, train_mode):
        """
        :param predict_length: int, length of predict
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index
            end_index = index + history_length
        else:
            raise ValueError("train model {} is not defined".format(train_mode))
        
        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index:end_index + predict_length]

        return data_x, data_y


    @staticmethod
    def get_day_week_cyc(history_length, start_day,index):
        tmp_daylist = []
        tmp_weeklist = []
        for x in range(index, index + history_length):
            if x == 0:
                tmp_daylist.append(x + 1)
                tmp_weeklist.append(x + 1)
            else:
                tmp_day = x % 288 + 1
                tmp_week = x // 288 - start_day
                tmp_daylist.append(tmp_day)
                tmp_weeklist.append(tmp_week % 7 + 1)
        return tmp_weeklist, tmp_daylist


    @staticmethod
    def pre_process_data(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = PEMSDataset.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = PEMSDataset.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        
    
        mean_val = np.mean(data, axis=(0, 1)).reshape(1, 1)
        std_val = np.std(data, axis=(0, 1)).reshape(1, 1)

        
        #return max_data, min_data
        return mean_val, std_val

    @staticmethod
    def normalize_data(mean_val, std_val, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        #mid = min_data
        #base = max_data - min_data
        #normalized_data = (data - mid) / base
        #normalized_data = data
        normalized_data = (data - mean_val) / std_val

        return normalized_data

    

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


def get_loader(ds_name="PEMS04", num_nodes=307):

    if ds_name == 'PEMS04':
        div_day = [35,12,12]
    elif ds_name == 'PEMS03':
        div_day = [55,6,6]
    elif ds_name == 'PEMS07':
        div_day = [58,20,20]
    else:
        div_day = [38,12,12]

    train_data = PEMSDataset(data_path=["./dataset/{}/{}.csv".format(ds_name, ds_name), "./dataset/{}/{}.npz".format(ds_name, ds_name)], num_nodes=num_nodes,
                             divide_days=div_day,
                             time_interval=5, history_length=12, predit_length=12,
                             train_mode="train")

    train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

    test_data = PEMSDataset(data_path=["./dataset/{}/{}.csv".format(ds_name, ds_name), "./dataset/{}/{}.npz".format(ds_name, ds_name)], num_nodes=num_nodes,
                            divide_days=div_day,
                            time_interval=5, history_length=12,predit_length=12,
                            train_mode="test")

    test_loader = DataLoader(test_data, batch_size=16,  shuffle=False)
    return train_loader, train_data.flow_norm, test_loader, test_data.flow_norm, torch.tensor(train_data.graph)
