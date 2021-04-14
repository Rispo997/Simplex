import numpy as np


class simplex:
    def __init__(self, A, b, c, signs, maximize = True, debug=False,M = 1e4):
        self.A = A
        self.b = b
        self.c = c
        self.signs = signs
        self.n_of_constraints,self.n_of_variables = A.shape
        self.base_variables = []
        self.artif_variables_col = []
        self.artif_variables_row = []
        self.M = M
        self.maximize = maximize
        self.debug = debug

    # Add a column to the tableau with the given value at the given index.
    def create_column(self,value,index,artificial = False):
        col = np.zeros(self.tableau.shape[0],)
        col[index] = value
        self.tableau = np.column_stack((self.tableau,col))
        if artificial:
            self.artif_variables_col.append(self.n_of_variables + index + 1)
            self.artif_variables_row.append(index)
            
    # Generate the starting tableau
    def generate_tableau(self):
        # All the rhs must be positive before creating the tableau
        for index,rhs in enumerate(self.b):
            if rhs <= 0:
                self.b[index] = -self.b[index]
                self.signs[index] = -self.signs[index]
                self.A[index] = -self.A[index]
        # Min Z = - Max Z
        if not self.maximize:
            self.c = -self.c
        self.tableau = self.A
        # Add slack and artificial variables for each constraint
        for index,constraint in enumerate(self.tableau):
            if self.signs[index] == 1:       # >= Case: Negative slack variable and artificial variable
                self.create_column(-1,index,False)
                self.create_column(1,index,True)
                self.base_variables.append(index+1)
            elif self.signs[index] == -1:    # <= Case: Positive slack variable
                self.create_column(1,index,False)
            elif self.signs[index] == 0: # = Case: Only artificial variable is needed
                self.create_column(1,index,True)
                self.base_variables.append(index)
        # Add the objective function row
        obj_row = np.zeros(self.tableau.shape[1],)
        obj_row[self.artif_variables_col] = -self.M
        obj_row[:self.c.shape[0]] = self.c.reshape(self.c.shape[0],)
        self.tableau = np.vstack((self.tableau,-obj_row))
        # Add Z Column
        self.create_column(1,-1,False)
        # Add RHS column
        b = np.zeros((self.n_of_constraints + 1,))
        b[:self.b.shape[0]] = self.b.reshape(self.b.shape[0],)
        self.tableau = np.column_stack((self.tableau,b.reshape(b.shape[0],1)))
        # If artificial variables are present, the cost has to be updated
        if self.artif_variables_row:
            self.tableau[-1] = self.tableau[-1] + ((-self.M)*self.tableau[self.artif_variables_row]).sum(axis=0)
        return self.tableau
    
    def get_base_variables(self):
        self.base_variables = [index+1 for index,col in enumerate(self.tableau.T[:-2,]) if col.sum() == 1]
        return self.base_variables
    
    def update_tableau(self):
        # Select entering variable x
        j = np.argmin(self.tableau[-1,:-1])
        # Select leaving variable s
        i = np.argmin([rhs/pivot if rhs>0 and pivot>0 else np.inf for pivot,rhs in zip(self.tableau[:,j],self.tableau[:,-1])])
        # Pivoting step
        pivot = self.tableau[i, j]
        if pivot < 0:
           raise self.unbounded_solution("The solution is unbounded")

        pivot_row = self.tableau[i,:] / pivot
        self.tableau = self.tableau - self.tableau[:,[j]] * pivot_row
        self.tableau[i,:] = pivot_row
        
    def compute_simplex(self):
        np.set_printoptions(suppress=True,precision=3)
        self.tableau = self.generate_tableau()
        # Stop condition: All the values in the last row are positive
        while np.min(self.tableau[-1,:-1]) < 0:
            self.update_tableau()
            if self.debug:
                self.print_tableau()
        if set(self.get_base_variables()).intersection(self.artif_variables_col):
            raise self.infeasible_solution()
        # Computation has ended and it yielded a feasible solution, print the result.
        self.print_results()
        
    def print_tableau(self):
        table = '{:<8}'.format("Base ") \
                  + "".join(['{:>8}'.format("x"+str(i+1)) for i in range(self.tableau.shape[1]-2)])   \
                  + '{:>8}'.format("Z") \
                  + '{:>8}'.format("RHS")
        for i, row in enumerate(self.tableau[:-1]):
            table += "\n" 
            table += '{:<8}'.format("x" + str(self.get_base_variables()[i])) \
                   + "".join(["{:>8.2f}".format(item) for item in row])
        table += "\n" 
        table += '{:<8}'.format("Z") + "".join(["{:>8.1f}".format(item) for item in self.tableau[-1]])
        print(table)
    
        
    def print_results(self):
        print("The optimal value reached is {:.2f}".format(self.tableau[-1,-1] if self.maximize else -self.tableau[-1,-1]))
        print("The following variables are in base:")
        for index,col in enumerate(self.tableau.T[:-2,]):
            if col.sum() == 1:
                value = self.tableau[np.where(col==1),-1]
                print("Variable X" + str(index+1) + " is in base with value: {:.2f}".format(value[0][0]))
        print("Final Tableau:")
        self.print_tableau()
    
    
    class unbounded_solution(Exception):
        pass
    
    class infeasible_solution(Exception):
        pass
    
