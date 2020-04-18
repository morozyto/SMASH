import partition
import taylor_expansion
import tools
import compression

from sklearn.utils.extmath import randomized_svd

class HSS:

    def __init__(self, X, A):
        self.X = X
        self.A = A
        self.Partition = partition.Partition(X)
        self.Partition.build_levels()
        self.build()

    def build(self):
        for l in range(self.Partition.max_level, 0, -1):

            for obj in self.level_to_nodes[l]:


            for obj in self.level_to_nodes[l]:
                A_ = tools.get_block(self.A, obj.i_row, sum([t.i_col for t in obj.N]))
                S_i_, _, _ = randomized_svd(A_,
                              n_components=15,
                              n_iter=5,
                              random_state=None)
                U_i_ = taylor_expansion.form_U([self.X[i] for i in obj.i_row])
                P, G, obj.i_row_cup = compression.compr(tools.concat_column_wise(U_i_, S_i_))
                n = P.shape()
                tmp = np.multiply(P, tools.concat_row_wise(np.identity(n - G.shape()), G))

                if obj.is_leaf:
                    obj.U = tmp
                else:
                    R_1 = tools.get_block(tmp, [i for i in range(len(obj.Children[0].i_row))], [i for i in range('ALL')])
                    R_2 = tools.get_block(tmp, [i for i in range(len(obj.Children[0].i_row), 'A::')], [i for i in range('ALL')])
                    obj.R = [R_1, R_2]


'''
                A_t = tools.get_block(A, sum([t.i_row for t in obj.N]), obj.i_col).transpose()
                T_i_, _, _ = randomized_svd(A_t,
                              n_components=15,
                              n_iter=5,
                              random_state=None)

                V_i_ = taylor_expansion.form_V(obj.i_col)
                F, H, i = compression.compr(tools.concat_column_wise(T_i_, V_i_))
                if obj.is_leaf:
                    obj.V = [F, H]
                else:
                    obj.W = [W_1, W_2]
'''

    def multiply(self, vec):
        pass