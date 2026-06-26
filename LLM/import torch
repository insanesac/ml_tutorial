    import torch


class ROPE(nn.Module):
    def __init__(self, head_dim, max_seq_length):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length

        cos, sin = self.build_rope()

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def build_rope(self):

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

        positional_ids = torch.arange(self.max_seq_length).float()

        angles = torch.outer(positional_ids, inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


    def rope(self, x):
        T = x.shape[2]

        cos = self.cos[...,:T,:]
        sin = self.sin[...,:T,:]
        even = x[...,::2]
        odd = x[...,1::2]

        pair1 = even*cos - odd*sin
        pair2 = even*sin + odd*cos


        output = torch.empty_like(x)

        output[...,::2] = pair1
        output[...,1::2] = pair2

        return output

    def forward(self, q, k):

       

        q_rot = self.rope(q)

        k_rot = self.rope(k)
        return q_rot, k_rot