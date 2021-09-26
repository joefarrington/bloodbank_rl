import pyomo.environ as pyo


class PyomoModelConstructor:
    def __init__(
        self,
        demand,
        t_max=30,
        a_max=3,
        initial_inventory={1: 0, 2: 50},
        fixed_order_cost=225,
        variable_order_cost=650,
        holding_cost=130,
        emergency_procurement_cost=3250,
        wastage_cost=650,
        M=250,
    ):

        self.model = pyo.ConcreteModel()

        self.model.T = pyo.RangeSet(1, t_max)
        self.model.A = pyo.RangeSet(1, a_max)

        self.model.M = M

        self.model.initial_inventory = initial_inventory

        self.model.demand = demand

        self.model.CssF = fixed_order_cost
        self.model.CssP = variable_order_cost
        self.model.CssH = holding_cost
        self.model.CssE = emergency_procurement_cost
        self.model.CssW = wastage_cost

        self.model.cons = pyo.ConstraintList()

    def build_model(self):

        self._add_common_variables()
        self._add_specific_variables()
        self._add_cost_function()
        self._add_common_constraints()
        self._add_specific_constraints()

        return self.model

    def _add_common_variables(self):

        self.model.OQ = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # Units ordered at end of day t
        self.model.X = pyo.Var(
            self.model.T, self.model.A, domain=pyo.NonNegativeReals
        )  # Units received at beginning of day t with shelf life a
        self.model.DssR = pyo.Var(
            self.model.T, self.model.A, domain=pyo.NonNegativeReals
        )  # Remaining demand on day t after using product with shelf life a days
        self.model.binDssR = pyo.Var(
            self.model.T, self.model.A, domain=pyo.Binary
        )  # Binary flag if there is remaining demand on day t after using product with shelf life a days
        self.model.IssB = pyo.Var(
            self.model.T, self.model.A, domain=pyo.NonNegativeReals
        )  # On-hand inventory at the beginning of day t with shelf life a days
        self.model.IssE = pyo.Var(
            self.model.T, self.model.A, domain=pyo.NonNegativeReals
        )  # Inventory at the end of day t with shelf life a days
        self.model.IP = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # Inventory position at the end of day t
        self.model.E = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # Number of units obtained through emergency procurement on day t
        self.model.W = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # Number of units wasted at the end of day t
        self.model.Delta = pyo.Var(
            self.model.T, domain=pyo.Binary
        )  # 1 if IP_t is less than s, 0 otherwise
        self.model.F = pyo.Var(
            self.model.T, domain=pyo.Binary
        )  # 1 is order placed on day t, 0 otherwise
        self.model.s = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # re-order point

    def _add_cost_function(self):

        # This does not include the extra cost we considered looking at the difference between s and S
        self.model.fixed_cost = sum(
            self.model.CssF * self.model.F[t] for t in self.model.T
        )
        self.model.variable_cost = sum(
            self.model.CssP * self.model.OQ[t] for t in self.model.T
        )
        self.model.holding_cost = sum(
            self.model.CssH * sum(self.model.IssE[t, a] for a in self.model.A if a > 1)
            for t in self.model.T
        )
        self.model.wastage_cost = sum(
            self.model.CssW * self.model.W[t] for t in self.model.T
        )
        self.model.shortage_cost = sum(
            self.model.CssE * self.model.E[t] for t in self.model.T
        )
        self.model.objective = pyo.Objective(
            expr=self.model.fixed_cost
            + self.model.variable_cost
            + self.model.holding_cost
            + self.model.wastage_cost
            + self.model.shortage_cost,
            sense=pyo.minimize,
        )

    def _add_common_constraints(self):

        # Equation 3
        for t in self.model.T:
            self.model.cons.add(self.model.OQ[t] <= self.model.M * self.model.F[t])

        # Equation 4
        # For now, not included, because review time never actually gets changed

        # Equations 5 and 6
        for t in self.model.T:
            if t == 1:
                self.model.cons.add(sum(self.model.X[t, a] for a in self.model.A) == 0)
            else:
                self.model.cons.add(
                    sum(self.model.X[t, a] for a in self.model.A)
                    == self.model.OQ[t - 1]
                )

        # For the baseline setting, all inventory should have three useful days of live when received
        for t in self.model.T:
            for a in self.model.A:
                if a != self.model.A[-1]:
                    self.model.cons.add(self.model.X[t, a] == 0)

        # Equations 7 and 8:
        for t in self.model.T:
            for a in self.model.A:
                if a == 1:
                    self.model.cons.add(
                        self.model.demand[t]
                        - self.model.IssB[t, a]
                        - self.model.X[t, a]
                        == self.model.DssR[t, a] - self.model.IssE[t, a]
                    )
                else:
                    self.model.cons.add(
                        self.model.DssR[t, a - 1]
                        - self.model.IssB[t, a]
                        - self.model.X[t, a]
                        == self.model.DssR[t, a] - self.model.IssE[t, a]
                    )

        # We need to enforce that only one variable on the RHS on equations 7 and 8 can be non-zero
        # For that we need an extra binary variable e.g. Pauls-Worm (inventory control for a perishable product with non-stationary
        # demand and service level constraints)
        for t in self.model.T:
            for a in self.model.A:
                self.model.cons.add(
                    self.model.M * self.model.binDssR[t, a] >= self.model.DssR[t, a]
                )
                self.model.cons.add(
                    self.model.M * (1 - self.model.binDssR[t, a])
                    >= self.model.IssE[t, a]
                )

        # Equation 9
        # Amended to just suself.model.m over X for t < current t
        # using u as t'
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t]
                == sum(self.model.IssE[t, a] for a in self.model.A if a > 1)
                + sum(self.model.OQ[u] for u in self.model.T if u < t)
                - sum(
                    self.model.X[u, a]
                    for a in self.model.A
                    for u in self.model.T
                    if u <= t
                )
            )

        # Equation 10
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t]
                <= (self.model.s[t] - 1) + self.model.M * (1 - self.model.Delta[t])
            )

        # Equation 11
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t] >= self.model.s[t] - self.model.M * self.model.Delta[t]
            )

        # Equation 16
        # Paper says this should be in all, but no S for s,Q model, so specify where required

        # Equation 17
        for t in self.model.T:
            if t == self.model.T[-1]:
                pass
            else:
                for a in self.model.A:
                    if a == self.model.A[-1]:
                        pass
                    else:
                        self.model.cons.add(
                            self.model.IssB[t + 1, a] == self.model.IssE[t, a + 1]
                        )

        # Equation 18
        for t in self.model.T:
            self.model.cons.add(self.model.E[t] == self.model.DssR[t, self.model.A[-1]])

        # Equation 19
        for t in self.model.T:
            self.model.cons.add(self.model.W[t] == self.model.IssE[t, 1])

        # Equation 20
        for t in self.model.T:
            self.model.cons.add(self.model.IssB[t, self.model.A[-1]] == 0)

        # Equation 21
        for a in self.model.A:
            if a == self.model.A[-1]:
                pass
            else:
                self.model.cons.add(
                    self.model.IssB[1, a] == self.model.initial_inventory[a]
                )

    def _add_specific_variables(self):
        # Implement for each model
        pass

    def _add_specific_constraints(self):
        # Impletement for each model
        pass

    @staticmethod
    def policy_parameters():
        # Implement for each model
        pass


class sS_PyomoModelConstructor(PyomoModelConstructor):
    def _add_specific_variables(self):

        self.model.S = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)

    def _add_specific_constraints(self):

        # Equation 16, but taking into account
        # that each weekday should have its own parameter
        for t in self.model.T:
            self.model.cons.add(self.model.S[t] >= self.model.s[t] + 1)

        # Equation B-2
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                <= (self.model.S[t] - self.model.IP[t])
                + self.model.M * (1 - self.model.Delta[t])
            )
            # Equation B-3
            self.model.cons.add(
                self.model.OQ[t]
                >= (self.model.S[t] - self.model.IP[t])
                - self.model.M * (1 - self.model.Delta[t])
            )
            # Equation B-4
            self.model.cons.add(self.model.OQ[t] <= self.model.M * self.model.Delta[t])

    @staticmethod
    def policy_parameters():
        return ["s", "S"]


class sQ_PyomoModelConstructor(PyomoModelConstructor):
    def _add_specific_variables(self):

        self.model.Q = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # order-up to level

    def _add_specific_constraints(self):

        # Constraints for s, Q model

        # Constraint C-2
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                <= self.model.Q[t] + self.model.M * (1 - self.model.Delta[t])
            )

        # Constraint C-3
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                >= self.model.Q[t] - self.model.M * (1 - self.model.Delta[t])
            )

        # Constaint C-4
        for t in self.model.T:
            self.model.cons.add(self.model.OQ[t] <= self.model.M * self.model.Delta[t])

    @staticmethod
    def policy_parameters():
        return ["s", "Q"]


class sSaQ_PyomoModelConstructor(PyomoModelConstructor):
    def _add_specific_variables(self):

        self.model.S = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
        self.model.Q = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # order-up to level
        self.model.a = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
        self.model.delta = pyo.Var(
            self.model.T, domain=pyo.Binary
        )  # 1 if IP_t is less than a, 0 otherwise

    def _add_specific_constraints(self):

        # Equation 12
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t]
                <= (self.model.a[t] - 1) + self.model.M * (1 - self.model.delta[t])
            )

        # Equation 13
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t] >= self.model.a[t] - self.model.M * self.model.delta[t]
            )

        # Equation 14 - linearised into A-1 to A-5

        ## Equation A-1
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                <= self.model.Q[t]
                + self.model.M * self.model.delta[t]
                + self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-2
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                >= self.model.Q[t]
                - self.model.M * self.model.delta[t]
                - self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-3
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                <= (self.model.S[t] - self.model.IP[t])
                + self.model.M * (1 - self.model.delta[t])
                + self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-4
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                >= (self.model.S[t] - self.model.IP[t])
                - self.model.M * (1 - self.model.delta[t])
                - self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-5
        for t in self.model.T:
            self.model.cons.add(self.model.OQ[t] <= self.model.M * self.model.Delta[t])

        # Equation 15
        for t in self.model.T:
            self.model.cons.add(self.model.s[t] >= self.model.a[t] + 1)

        # Equation 16
        for t in self.model.T:
            self.model.cons.add(self.model.S[t] >= self.model.s[t] + 1)

    @staticmethod
    def policy_parameters():
        return ["s", "S", "a", "Q"]


class sSbQ_PyomoModelConstructor(PyomoModelConstructor):
    def _add_specific_variables(self):
        self.model.S = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
        self.model.Q = pyo.Var(
            self.model.T, domain=pyo.NonNegativeReals
        )  # order-up to level
        self.model.b = pyo.Var(self.model.T, domain=pyo.NonNegativeReals)
        self.model.nu = pyo.Var(
            self.model.T, domain=pyo.Binary
        )  # 1 if IP_t is less than b, 0 otherwise

    def _add_specific_constraints(self):

        # Equation 16
        for t in self.model.T:
            self.model.cons.add(self.model.S[t] >= self.model.s[t] + 1)

        # Equation 26
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t]
                <= (self.model.b[t] - 1) + self.model.M * (1 - self.model.nu[t])
            )

        # Equation 27
        for t in self.model.T:
            self.model.cons.add(
                self.model.IP[t] >= self.model.b[t] - self.model.M * self.model.nu[t]
            )

        # Equation 28 - lineared into A-6 to A-10

        ## Equation A-6
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                <= self.model.Q[t]
                + self.model.M * (1 - self.model.nu[t])
                + self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-7
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                >= self.model.Q[t]
                - self.model.M * (1 - self.model.nu[t])
                - self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-8
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                <= (self.model.S[t] - self.model.IP[t])
                + (self.model.M * self.model.nu[t])
                + self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-9
        for t in self.model.T:
            self.model.cons.add(
                self.model.OQ[t]
                >= (self.model.S[t] - self.model.IP[t])
                - (self.model.M * self.model.nu[t])
                - self.model.M * (1 - self.model.Delta[t])
            )

        ## Equation A-10
        for t in self.model.T:
            self.model.cons.add(self.model.OQ[t] <= self.model.M * self.model.Delta[t])

        # Equation 29
        for t in self.model.T:
            self.model.cons.add(self.model.s[t] >= self.model.b[t] + 1)

    @staticmethod
    def policy_parameters():
        return ["s", "S", "b", "Q"]
