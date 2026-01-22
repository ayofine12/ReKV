from eventful_transformer.base import ExtendedModule
from eventful_transformer.counting import CountedMatmul
from eventful_transformer.utils import expand_col_index, expand_row_index

class TokenGate(ExtendedModule):
    def __init__(self, structure="row"):
        super().__init__()
        assert structure in ["row", "col"]
        self.structure = structure
        self.first = True
        self.policy = None
        self.p = None


    def forward(self, c, forced_index=None):
        if self.first:
            return self.forward_first(c)
        else:
            return self.forward_incremental(c, forced_index=forced_index)

    def forward_first(self, c):
        self.first = False
        self.p = c
        return c, None

    def forward_incremental(self, c, forced_index=None):
        if self.count_mode:
            self.counts["token_gate_flops"] += c.numel()
        dim, expanded, index = self._apply_policy(c - self.p, forced_index)
        c_tilde = c.gather(dim=dim, index=expanded)
        self.p.scatter_(dim=dim, index=expanded, src=c_tilde)
        return c_tilde, index

    def _apply_policy(self, x, forced_index):
        dim = -2 if (self.structure == "row") else -1
        if forced_index is None:
            index = self.policy(x, dim=(-1 if (self.structure == "row") else -2))
        else:
            index = forced_index
        if self.structure == "row":
            expanded = expand_row_index(index, x.shape)
        else:
            expanded = expand_col_index(index, x.shape)
        return dim, expanded, index

    def reset_self(self):
        self.first = True
        self.p = None

class MatmulBuffer(ExtendedModule):
    def __init__(self):
        super().__init__()
        self.first = True
        self.product = None
        self.matmul = CountedMatmul()

    def forward(self, q, k, index_q, index_k):
        if self.first:
            return self.forward_first(q, k)
        else:
            return self.forward_incremental(q, k, index_q, index_k)

    def forward_first(self, q, k):
        self.first = False
        self.product = self.matmul(q, k)
        return self.product

    ## q는 계속 같은게 들어올 예정정
    def forward_incremental(self, q, k, index_k):
        k_tilde = k.gather(dim=-1, index=expand_col_index(index_k, k.shape))
        self.product.scatter_(
            dim=-1,
            index=expand_col_index(index_k, self.product.shape),
            src=self.matmul(q, k_tilde)
        )
        return self.product
    def reset_self(self):
        self.first = True
        self.product = None



