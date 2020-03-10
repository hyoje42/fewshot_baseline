#-------------------------------------
# The code adopted from https://github.com/csyanbin/TPN-pytorch
#-------------------------------------
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np

from torch.distributions.dirichlet import Dirichlet

class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self, x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out


class CNNEncoder_average(nn.Module):
    """Encoder for feature embedding"""

    def __init__(self):
        super(CNNEncoder_average, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        return out

def preprocess_input(inputs):
    """
    inputs are preprocessed. Set number of classes, support, queries.
    Concat images of support and queries.
    :param
        inputs:
            support:    (N_way*N_shot)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot [25, 5]
            query:      (N_way*N_query)x3x84x84
            q_labels:   (N_way*N_query)xN_way, one-hot
    :return : num_classes, num_support, num_queries, concat_s_q
    """
    ## input part
    [support, s_labels, query, q_labels] = inputs
    num_classes = s_labels.shape[1]
    num_support = int(s_labels.shape[0] / num_classes)
    num_queries = int(query.shape[0] / num_classes)
    # support : (Ns, c, w, h), query : (Nq, c, w, h)
    concat_s_q = torch.cat((support, query), 0)  # inp : (Ns + Nq, c, w, h)

    return num_classes, num_support, num_queries, concat_s_q

class Prototypical(nn.Module):
    """Main Module for prototypical networlks"""
    def __init__(self, args, optimizer, encoder=CNNEncoder):
        super(Prototypical, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args.x_dim.split(',')))
        self.args = args
        self.encoder = encoder()
        self.optimizer = optimizer(self.parameters(), lr=args.lr)

    def forward(self, inputs):
        """
        get embedding of support and query
        :param
            inputs:
                support:    (N_way*N_shot)x3x84x84
                s_labels:   (N_way*N_shot)xN_way, one-hot [25, 5]
                query:      (N_way*N_query)x3x84x84
                q_labels:   (N_way*N_query)xN_way, one-hot
        :return:
            emb_support : (1, Nc, 64*5*5)
            emb_query : (Nq, 1, 64*5*5)
        """
        ## process inputs
        self.num_classes, self.num_support, self.num_queries, self.concat_s_q = preprocess_input(inputs)
        self.q_labels = inputs[-1]

        ## encoding part
        emb   = self.encoder(self.concat_s_q) # emb shape 100, 64, 5, 5
        # emb_s : (Ns, 64, 5, 5), emb_q : (Nq, 64, 5, 5)
        emb_s, emb_q = torch.split(emb, [self.num_classes*self.num_support, self.num_classes*self.num_queries], 0)

        ## prototype part
        emb_s = emb_s.view(self.num_classes, self.num_support, np.prod(emb_s.shape[1:])).mean(1)  # (5, 5, 64*5*5), (Nc*Ns*s_dim)
        emb_q = emb_q.view(-1, np.prod(emb_q.shape[1:]))    # (Nq, 64*5*5)
        assert emb_s.shape[-1] == emb_q.shape[-1], 'the dimension of embeddings must be equal'
        emb_s = torch.unsqueeze(emb_s, 0)     # 1xNxD, (1, Nc, 64*5*5)
        emb_q = torch.unsqueeze(emb_q, 1)     # Nx1xD, (Nq, 1, 64*5*5)

        return emb_s, emb_q

    def cal_loss(self, emb_s, emb_q):
        """
        calculate loss of emb_s and emb_q
        :param emb_s: (1, Nc, 64*5*5)
        :param emb_q: (Nq, 1, 64*5*5)
        :return:
        """
        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)

        ce = nn.CrossEntropyLoss().cuda(0)
        loss = ce(-dist, torch.argmax(self.q_labels, 1))

        return loss

    def update(self, loss):
        """
        Update parameters. optimizer is already defiend.
        """
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_acc(self, inputs):
        """
        Get accuracy for an episode.
        If inputs are [support, s_label, query, q_label], then get embedding of support and query.
        If inputs are [emb_s, emb_q], tehn do not inference embeddings.
        :return: accuracy of an episode.
        """
        if len(inputs) == 2:
            # embedding is already inferenced.
            emb_s, emb_q = inputs
        else:
            emb_s, emb_q = self.forward(inputs)
        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)
        ## acc
        pred = torch.argmax(-dist, 1)
        gt = torch.argmax(self.q_labels, 1)
        correct = (pred == gt).sum()
        total = self.num_queries * self.num_classes
        acc = 1.0 * correct.float() / float(total)

        return acc

class PrototypicalCycle(Prototypical):
    """Main Module for prototypical networlks"""
    def __init__(self, args, optimizer, encoder=CNNEncoder):
        super(PrototypicalCycle, self).__init__(args, optimizer, encoder)

    def forward(self, inputs):
        """
        get embedding of support and query
        :param
            inputs:
                support:    (N_way*N_shot)x3x84x84
                s_labels:   (N_way*N_shot)xN_way, one-hot [25, 5]
                query:      (N_way*N_query)x3x84x84
                q_labels:   (N_way*N_query)xN_way, one-hot
        :return:
            emb_support : (1, Nc, 64*5*5)
            emb_query : (Nq, 1, 64*5*5)
            emb_qq : (1, Nc, 64*5*5)
            emb_ss : (Ns, 1, 64*5*5)
        """
        ## process inputs
        self.num_classes, self.num_support, self.num_queries, self.concat_s_q = preprocess_input(inputs)
        _, self.s_labels, _, self.q_labels = inputs

        ## encoding part
        emb   = self.encoder(self.concat_s_q) # emb shape 100, 64, 5, 5
        # emb_s : (Ns, 64, 5, 5), emb_q : (Nq, 64, 5, 5)
        emb_s, emb_q = torch.split(emb, [self.num_classes*self.num_support, self.num_classes*self.num_queries], 0)

        ## prototype part
        emb_s = emb_s.view(self.num_classes, self.num_support, np.prod(emb_s.shape[1:])).mean(1)  # (5, 64*5*5)
        emb_q = emb_q.view(-1, np.prod(emb_q.shape[1:]))    # (Nq, 64*5*5)
        assert emb_s.shape[-1] == emb_q.shape[-1], 'the dimension of embeddings must be equal'
        emb_s = torch.unsqueeze(emb_s, 0)     # 1xNxD, (1, Nc, 64*5*5)
        emb_q = torch.unsqueeze(emb_q, 1)     # Nx1xD, (Nq, 1, 64*5*5)

        # cyclic part
        emb_ss, emb_qq = torch.split(emb, [self.num_classes*self.num_support, self.num_classes*self.num_queries], 0)

        emb_qq = emb_qq.view(self.num_classes, self.num_queries, np.prod(emb_qq.shape[1:])).mean(1)  # (Nc, 64*5*5)
        emb_ss = emb_ss.view(-1, np.prod(emb_ss.shape[1:]))  # (Ns, 64*5*5)
        assert emb_qq.shape[-1] == emb_ss.shape[-1], 'the dimension of embeddings must be equal'
        emb_qq = torch.unsqueeze(emb_qq, 0)  # 1xNxD, (1, Nc, 64*5*5)
        emb_ss = torch.unsqueeze(emb_ss, 1)  # Nx1xD, (Ns, 1, 64*5*5)

        return emb_s, emb_q, emb_qq, emb_ss

    def cal_loss(self, emb_s, emb_q, emb_qq, emb_ss):
        """
        calculate loss of emb_s and emb_q
        :param emb_s: (1, Nc, 64*5*5)
        :param emb_q: (Nq, 1, 64*5*5)
        :return:
        """
        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)

        ce = nn.CrossEntropyLoss().cuda(0)
        loss = ce(-dist, torch.argmax(self.q_labels, 1))

        dist_ = ((emb_ss - emb_qq) ** 2).mean(2)  # NxNxD -> NxN, (Ns, Nc)

        ce_ = nn.CrossEntropyLoss().cuda(0)
        loss_ = ce_(-dist_, torch.argmax(self.s_labels, 1))

        return loss, loss_

    def get_acc(self, inputs):
        """
        Get accuracy for an episode.
        If inputs are [support, s_label, query, q_label], then get embedding of support and query.
        If inputs are [emb_s, emb_q], tehn do not inference embeddings.
        :return: accuracy of an episode.
        """
        if len(inputs) == 2:
            # embedding is already inferenced.
            emb_s, emb_q = inputs
        else:
            emb_s, emb_q, _, _ = self.forward(inputs)
        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)
        ## acc
        pred = torch.argmax(-dist, 1)
        gt = torch.argmax(self.q_labels, 1)
        correct = (pred == gt).sum()
        total = self.num_queries * self.num_classes
        acc = 1.0 * correct.float() / float(total)

        return acc

class PrototypicalCycleQueryProto(PrototypicalCycle):
    """Main Module for prototypical networlks"""
    def __init__(self, args, optimizer, encoder=CNNEncoder):
        super(PrototypicalCycleQueryProto, self).__init__(args, optimizer, encoder)

    def forward(self, inputs):
        """
        Get embedding of support and query.
        :param
            inputs:
                support:    (N_way*N_shot)x3x84x84
                s_labels:   (N_way*N_shot)xN_way, one-hot [25, 5]
                query:      (N_way*N_query)x3x84x84
                q_labels:   (N_way*N_query)xN_way, one-hot
        :return:
            emb_support : (1, Nc, 64*5*5)
            emb_query : (Nq, 1, 64*5*5)
            emb_qq : (1, Nc, 64*5*5)
            emb_ss : (Ns, 1, 64*5*5)
        """
        ## process inputs
        self.num_classes, self.num_support, self.num_queries, self.concat_s_q = preprocess_input(inputs)
        _, self.s_labels, _, self.q_labels = inputs

        ## encoding part
        emb   = self.encoder(self.concat_s_q) # emb shape 100, 64, 5, 5
        # emb_s : (Ns, 64, 5, 5), emb_q : (Nq, 64, 5, 5)
        emb_s, emb_q = torch.split(emb, [self.num_classes*self.num_support, self.num_classes*self.num_queries], 0)

        ## prototype part
        emb_s = emb_s.view(self.num_classes, self.num_support, np.prod(emb_s.shape[1:])).mean(1)  # (5, 64*5*5)
        emb_q = emb_q.view(-1, np.prod(emb_q.shape[1:]))    # (Nq, 64*5*5)
        assert emb_s.shape[-1] == emb_q.shape[-1], 'the dimension of embeddings must be equal'
        emb_s = torch.unsqueeze(emb_s, 0)     # 1xNxD, (1, Nc, 64*5*5)
        emb_q = torch.unsqueeze(emb_q, 1)     # Nx1xD, (Nq, 1, 64*5*5)
        dist = ((emb_q - emb_s) ** 2).mean(2)  # NxNxD -> NxN, (Nq, Nc)

        ## cyclic part (prototype of inferenced query)
        emb_ss, emb_qq = torch.split(emb, [self.num_classes*self.num_support, self.num_classes*self.num_queries], 0)

        # make the prototype of embedded query
        emb_merge = torch.tensor([]).view(0, np.prod(emb_qq.shape[1:])).cuda(0)
        for cl in range(self.num_classes):
            idxs = np.where(torch.argmax(-dist, 1).cpu() == cl)[0]
            if not len(idxs) > 0:
                # use prototype of support set
                emb_temp_proto = emb_s[:, cl, :]
            else:
                emb_temp_proto = emb_qq[idxs]
                emb_temp_proto = emb_temp_proto.view(-1, np.prod(emb_qq.shape[1:])).unsqueeze(0).mean(1)
            emb_merge = torch.cat((emb_merge, emb_temp_proto), 0).cuda(0)

        emb_qq = emb_merge  # (Nc, 64*5*5)
        emb_ss = emb_ss.view(-1, np.prod(emb_ss.shape[1:]))  # (Ns, 64*5*5)
        assert emb_qq.shape[-1] == emb_ss.shape[-1], 'the dimension of embeddings must be equal'
        emb_qq = torch.unsqueeze(emb_qq, 0)  # 1xNxD, (1, Nc, 64*5*5)
        emb_ss = torch.unsqueeze(emb_ss, 1)  # Nx1xD, (Ns, 1, 64*5*5)

        return emb_s, emb_q, emb_qq, emb_ss

class ConvexProto(Prototypical):
    def __init__(self, args, optimizer, encoder=CNNEncoder):
        super(ConvexProto, self).__init__(args, optimizer, encoder)
    
    def forward(self, inputs):
        """
        get embedding of support and query
        :param
            inputs:
                support:    (N_way*N_shot)x3x84x84
                s_labels:   (N_way*N_shot)xN_way, one-hot [25, 5]
                query:      (N_way*N_query)x3x84x84
                q_labels:   (N_way*N_query)xN_way, one-hot
        :return:
            emb_support : (1, Nc, 64*5*5)
            emb_query : (Nq, 1, 64*5*5)
        """
        ## process inputs
        self.num_classes, self.num_support, self.num_queries, self.concat_s_q = preprocess_input(inputs)
        self.q_labels = inputs[-1]

        ## encoding part
        emb   = self.encoder(self.concat_s_q) # emb shape:(100*64*5*5)
        # emb_s:(Ns*64*5*5), emb_q:(Nq*64*5*5)
        emb_s, emb_q = torch.split(emb, [self.num_classes*self.num_support, self.num_classes*self.num_queries], 0)

        ## prototype part
        alpha = 0.2
        m = Dirichlet(torch.tensor([alpha]*5))
        convexhull_weights = m.sample((emb_s.size(0),)).cuda(0)

        if self.training:
            # convex combination
            emb_s = emb_s.view(self.num_classes, self.num_support, np.prod(emb_s.shape[1:]))  # (5, 5, 64*5*5), (Nc*Ns*s_dim)
            emb_s = torch.cat([torch.matmul(emb_s[i].transpose(0,1), convexhull_weights[i].view(5,1)) for i in range(5)], dim=1).transpose(0,1)    # (5, 64*5*5), (Nc*s_dim)
        else:    
            # prototype
            emb_s = emb_s.view(self.num_classes, self.num_support, np.prod(emb_s.shape[1:])).mean(1)  # (5, 64*5*5), (Nc*s_dim)
        
        emb_q = emb_q.view(-1, np.prod(emb_q.shape[1:]))    # (Nq, 64*5*5)
        assert emb_s.shape[-1] == emb_q.shape[-1], 'the dimension of embeddings must be equal'
        emb_s = torch.unsqueeze(emb_s, 0)     # 1xNxD, (1, Nc, 64*5*5)
        emb_q = torch.unsqueeze(emb_q, 1)     # Nx1xD, (Nq, 1, 64*5*5)

        return emb_s, emb_q
