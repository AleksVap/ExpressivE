import torch
from pykeen.nn.modules import Interaction
from pykeen.utils import broadcast_cat

from Utils import preprocess_entities


class EqSlopesInteraction(Interaction):
    relation_shape = (
        'e'
    )
    entity_shape = (
        'd',
    )

    def __init__(self, p: int, embedding_dim: int, tanh_map: bool = True):
        super().__init__()
        self.p = p  # Norm that shall be used (either 1 or 2)
        self.tanh_map = tanh_map
        self.s = torch.empty(2 * embedding_dim, requires_grad=True).cuda()

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.normal_(self.s)


    def get_score(self, d_h, d_t, c_h, c_t, s_h, s_t, h, t):
        # Calculate the score of the triple

        d = torch.concat([d_h, d_t], dim=-1)  # distance
        c = torch.concat([c_h, c_t], dim=-1)  # centers
        s = torch.concat([s_t, s_h], dim=-1)  # slopes

        ht = broadcast_cat([h, t], dim=-1)
        th = broadcast_cat([t, h], dim=-1)

        contextualized_pos = torch.abs(ht - c - torch.mul(s, th))

        is_entity_pair_within_para = torch.le(contextualized_pos, d).all(dim=-1)

        w = 2 * d + 1

        k = torch.mul(0.5 * (w - 1), (w - 1 / w))
        dist = torch.mul(contextualized_pos, w) - k

        dist[is_entity_pair_within_para] = torch.div(contextualized_pos, w)[is_entity_pair_within_para]

        return -dist.norm(p=self.p, dim=-1)

    def preprocess_relations(self, r, tanh_map=True):
        d_h, d_t, c_h, c_t = r.tensor_split(4, dim=-1)

        ReLU = torch.nn.ReLU()

        d_h = torch.abs(d_h)
        d_t = torch.abs(d_t)

        if tanh_map:
            d_h = torch.tanh(d_h)
            d_t = torch.tanh(d_t)
            c_h = torch.tanh(c_h)
            c_t = torch.tanh(c_t)
        else:
            d_h = ReLU(d_h)
            d_t = ReLU(d_t)

        s = self.s[(None,) * (len(d_h.shape) - 1)]  # unsqueeze s to have the same number of dims as d
        s_h, s_t = s.tensor_split(2, dim=-1)

        return d_h, d_t, c_h, c_t, s_h, s_t

    def forward(self, h, r, t):
        d_h, d_t, c_h, c_t, s_h, s_t = self.preprocess_relations(r, tanh_map=self.tanh_map)

        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.get_score(d_h, d_t, c_h, c_t, s_h, s_t, h, t)
