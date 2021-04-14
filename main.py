from simplex import simplex
import numpy as np
# Rispoli Luca's assignment
# Implementation of the Simplex algorithm with the big M extension, made as project for the
# combinatorial and decision making course's second module taught by Vittorio Maniezzo.
# The class simplex takes as input:
# A: Matrix in which each row contains the coefficients of the constraint equations.
# B: Column vector containing the right hand side of each constraint.
# c: Column vector containing the coefficients of the objective function.
# Maximize: If true, the problem will be maximized. Otherwise it will be minimized.
# Debug: If true, intermediary steps are printed. Otherwise print only the result.
"""
A company produces chips in two different plants.
The first plant produces at most 800 chips per month.
The second plant produces at most 600 chips per month.
The chips are shipped to three depots (A,B,C) whose minimum monthly requirements are 
resp. 500,400 and 400 units.
Shipping costs are:
Plant 1 --> Depot A = 16€     Plant 2 --> Depot A = 18€
Plant 1 --> Depot B = 20€     Plant 2 --> Depot B = 16€
Plant 1 --> Depot C = 22€     Plant 2 --> Depot C = 14€

What shipping schedule will enable the company to meet the depots' requirements and keep the shipping costs at 
minimum? What is the minimum cost?
Be the variables (X1,X2,....,X6) resp. the numbers of chips shipped from plant 1 to A, plant 1 to B and so on.
"""
A = np.array([[1,1,1,0,0,0],[0,0,0,1,1,1],[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]])
B = np.array([[800],[600],[500],[400],[400]])
c = np.array([[16],[20],[22],[18],[16],[14]])     
signs = np.array([[-1],[-1],[1],[1],[1]])
minimization = simplex(A,B,c,signs,False,False)
minimization.compute_simplex()

print("\n \n")
"""
A company produces four types of engines that are produced from 4 types of machines.
The first machine produces the first engine in 1 minute, the second in 2 minutes, the third in 3 minutes
and the fourth in 2 minutes.
The second machine produces the first engine in 3 minutes, the second in 2 minutes, the third in 4 minutes and the
fourth in 6 minutes.
The third machine produces the first engine in 2 minutes, the second in 1 minute, the third in 3 minutes, the fourth
in 3 minutes.
The fourth machine produces the first engine in 2 minutes, the second engine in 4 minutes, the third engine in 3 
minutes, the fourth engine in 5 minutes.
The first engine is sold for 20€, the second for 35€, the third for 40€ and the fourth for 50€.
Which is the number of engines of each type ( namely X1,X2,X3,X4) that should be produced per hour in order to 
maximize the profit?

"""
A = np.array([[1,2,3,2],[3,2,4,6],[2,1,3,3],[2,4,3,5]])
B = np.array([[60],[60],[60],[60]])
c = np.array([[20],[35],[40],[50]])     
signs = np.array([[-1],[-1],[-1],[-1]])
maximization = simplex(A,B,c,signs,True,False)
maximization.compute_simplex()

