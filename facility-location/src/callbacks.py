from data import Data
from gurobipy import GRB, quicksum
from sub_problem import solve_subproblem


class Callback:
    def __init__(self, dat: Data, y, eta):
        self.dat = dat
        self.y = y
        self.eta = eta

        self.num_cuts_mip_rel = 0  
        self.num_cuts_mip_int_L_shape = 0  
        self.num_cuts_rel = 0  

    def __call__(self, mod, where):
        L = 0  # Define an appropriate lower bound value

        if where == GRB.Callback.MIPSOL:
            y_values = mod.cbGetSolution(self.y)
            eta_value = mod.cbGetSolution(self.eta)

            #obj = solve_subproblem(self.dat, y_values, linear_relaxation=False)
            #if obj > eta_value:
                #self.add_int_L_shape_cut(mod, obj, L, y_values) 
                #self.num_cuts_mip_int_L_shape += 1

            
            obj, mu, nu = solve_subproblem(self.dat, y_values, linear_relaxation=True)
            
            if obj > eta_value:
                self.add_benders_cut(mod, mu, nu)
                self.num_cuts_mip_rel += 1
            else:
                obj = solve_subproblem(self.dat, y_values, linear_relaxation=False)
                if obj > eta_value:
                   self.add_int_L_shape_cut(mod, obj, L, y_values)  
                   self.num_cuts_mip_int_L_shape += 1
            
        
        elif where == GRB.Callback.MIPNODE:
            node_count = mod.cbGet(GRB.Callback.MIPNODE_NODCNT)

            if node_count == 1:
                print("Completed solving the root node. Proceeding with branch-and-bound...")

            status = mod.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status != GRB.OPTIMAL:
                return

            y_values = mod.cbGetNodeRel(self.y)
            eta_value = mod.cbGetNodeRel(self.eta)

            obj, mu, nu = solve_subproblem(self.dat, y_values, linear_relaxation=True)

            if obj > eta_value:
                self.add_benders_cut(mod, mu, nu)
                self.num_cuts_rel += 1
        

    def add_benders_cut(self, mod, mu, nu):
        rhs = quicksum( mu[i] for i in self.dat.I)
        rhs += quicksum(self.dat.capacities[j] * nu[j] * self.y[j] for j in self.dat.J)
        mod.cbLazy(self.eta >= rhs)

    def add_int_L_shape_cut(self, mod, obj, L, y_values):
        S_y_star = [j for j in self.dat.J if y_values.get(j, 0) > 0.5]

        sum_in_S = quicksum(self.y[j] for j in S_y_star)
        sum_not_in_S = quicksum(self.y[j] for j in self.dat.J if j not in S_y_star)

        rhs = (obj - L) * (sum_in_S - sum_not_in_S - len(S_y_star)) + obj
        mod.cbLazy(self.eta >= rhs)

        print(f"Added integer L-shaped cut: eta >= {rhs}")
