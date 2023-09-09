import numpy as np
from probabilistic_automata import ProbabilisticAutomata
from collections import defaultdict
import random

#################### numpy test ####################

x = np.array([[0,0,0],[0,5,6],[0,8,9]])

v = np.array([[1],[2],[3],[4]])

z = np.array([[1,2,3]])

i = np.array([[0] * 4])

a = np.delete(x,0,0)

b = np.delete(a,0,1)

c = np.delete(z,0,1)

vector = np.array([[1],[0.5]])

#################### method tests ####################

#random automaton

def random_matrix_automata(size) :

    ligne = []
    matrix = np.zeros(size)
    borne = 1
    for i in range(size - 1):
        x = random.uniform(0,borne)
        borne -= x
        ligne.append(x)
        print(matrix.flat)
        matrix.flat[i] = ligne
        
    return matrix


#Automate 1

Sigma = {'a','b'}
Q = {0, 1}
lambdA = np.array([[1/2, 1/2]])
mu = {'a' : np.array([[0,1],[1/2,1/2]]), 'b' : np.array([[3/4,1/4],[1,0]])}
gamma = np.array([[3/4],[1/4]])


PA = ProbabilisticAutomata(Sigma, Q, lambdA, mu, gamma)

#Automate 2


Sigmabis = {'a','b'}
Qbis = {0, 1}
lambdAbis = np.array([[1/4, 3/4]])
mubis = {'a' : np.array([[0,0],[0,1]]), 'b' : np.array([[0,0],[0,1]])}
gammabis = np.array([[3/4],[1/4]])
vectorbis = np.array([[1],[1]])

PAbis =ProbabilisticAutomata(Sigmabis,Qbis,lambdAbis,mubis,gammabis)

#Automate 3


SigmaN = {'a','b'}
QN = {0, 1, 2, 3, 4, 5}
lambdAN = np.array([[1, 0, 0, 0, 0, 0]])
muN = {'a' : np.array([[0, 1/2, 1/2, 0, 0, 0],[1, 0, 0, 0, 0, 0],[0, 0, 1/2, 0, 1/2, 0],[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]]), 'b' : np.array([[0, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 1/2, 0, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1]])}
gammaN = np.array([[0],[0],[0],[1],[0],[1]])

PAanbn = ProbabilisticAutomata(SigmaN,QN,lambdAN,muN,gammaN)

#Automate 4

Sigmater = {'a','b'}
Qter = {0, 1, 2}
lambdAter = np.array([[1, 0, 0]])
muter = {'a' : np.array([[0,0,0],[0,1,0],[0,0,0]]), 'b' : np.array([[1,0,0],[0,0,0],[0,0,0]])}
gammater = np.array([[3/4],[1/4],[1]])

PAter = ProbabilisticAutomata(Sigmater,Qter,lambdAter,muter,gammater)

#Automate 5

Sigma4 = {'a','b'}
Q4 = {0, 1, 2}
lambdA4 = np.array([[1, 0, 0]])
mu4 = {'a' : np.array([[1/2,1/2,0],[0,1,0],[0,1/2,1/2]]), 'b' : np.array([[0,0,1],[1/3,2/3,0],[1,0,0]])}
gamma4 = np.array([[0],[0],[1]])

PA4 = ProbabilisticAutomata(Sigma4,Q4,lambdA4,mu4,gamma4)

#Automate 6

Sigma5 = {'a','b'}
Q5 = {0, 1, 2}
lambdA5 = np.array([[0, 0.5, 0.25]])
mu5 = {'a' : np.array([[0,0,0],[0,0,0],[0,0,0]]), 'b' : np.array([[1,0,0],[1,0,0],[1,0,0]])}
gamma5 = np.array([[0],[0],[1]])

PA5 = ProbabilisticAutomata(Sigma5,Q5,lambdA5,mu5,gamma5)

#Automate 7

Sigma6 = {'a','b'}
Q6 = {0, 1}
lambdA6 = np.array([[ 1, 0]])
mu6 = {'a' : np.array([[1/2,1/2],[0,1]]), 'b' : np.array([[1,0],[1/3,2/3]])}
gamma6 = np.array([[7/8],[5/6]])

PA6 = ProbabilisticAutomata(Sigma6,Q6,lambdA6,mu6,gamma6)

#Automate 8
Sigma7 = {'a','b'}
Q7 = {0, 1, 2}
lambdA7 = np.array([[1/2, 0, 1/2]])
mu7 = {'a' : np.array([[0,1/2,1/2],[0,1/2,1/2],[0,0,1]]), 'b' : np.array([[1,0,0],[0,1,0],[1/3,0,2/3]])}
gamma7 = np.array([[0],[1/4],[1/4]])

PA7 = ProbabilisticAutomata(Sigma7,Q7,lambdA7,mu7,gamma7)
######### test_tools ##########

# print(ProbabilisticAutomata.min_greater_than_zero(PA, vector))

# print(ProbabilisticAutomata.min_greater_than_zero(PA5, gamma5))

#print("vector_decomposition",ProbabilisticAutomata.vector_decomposition(PA, gamma))


#print(ProbabilisticAutomata.convertisor_column_vector_to_an_integer(PA,vector))

######### test_miror_element ##########

#print(ProbabilisticAutomata.miror_states(PA))

#print(ProbabilisticAutomata.miror_vectors(PA))

#print(ProbabilisticAutomata.miror_final_states(PA))

#print(ProbabilisticAutomata.miror_final_states(PAbis))

#print("miror_initial_states",ProbabilisticAutomata.miror_initial_states(PA))

#print(ProbabilisticAutomata.new_transition_matrix(PA, 'a'))

#print(ProbabilisticAutomata.new_set_of_transition_matrix(PA))

############### test_clearing_function ###############

#print(ProbabilisticAutomata.check_if_line_and_column_null(PA,x))

#print(ProbabilisticAutomata.cleared_useless_state(PAbis))

#print(ProbabilisticAutomata.cleared_useless_state(PA))

#print(ProbabilisticAutomata.cleared_useless_state(PAter))

######### test_miror_automaton_function ##########

#print(ProbabilisticAutomata.miror_automaton(PAbis))

#print(ProbabilisticAutomata.miror_automaton(PA))

#print(ProbabilisticAutomata.miror_automaton(PAter))

#print(ProbabilisticAutomata.miror_cleared_automaton(PAanbn))

#print(ProbabilisticAutomata.miror_cleared_automaton(PA))

#print(ProbabilisticAutomata.miror_cleared_automaton(ProbabilisticAutomata.miror_cleared_automaton(PAanbn)))

#print(ProbabilisticAutomata.miror_automaton(ProbabilisticAutomata.miror_automaton(PAanbn)))

#print(ProbabilisticAutomata.miror_cleared_automaton(ProbabilisticAutomata.miror_cleared_automaton(PAanbn)))

#print(ProbabilisticAutomata.miror_cleared_automaton(ProbabilisticAutomata.miror_cleared_automaton(PAbis)))

#print(ProbabilisticAutomata.miror_cleared_automaton(ProbabilisticAutomata.miror_cleared_automaton(PA4)))

######### test_probability_function ##########

#print(ProbabilisticAutomata.word_probability(PA,'ab'))

#print(ProbabilisticAutomata.word_probability(PA,'cab'))


######### test_reachable##########

#print(ProbabilisticAutomata.reachable_states(PAter))

print("original automata", PA7)
miror=ProbabilisticAutomata.miror_automaton(PA7)
print("miror",miror)
cleared_miror = ProbabilisticAutomata.cleared_automaton(miror)
print("cleared_miror", cleared_miror)
miror_miror= ProbabilisticAutomata.miror_automaton(cleared_miror)
print("miror_miror",miror_miror)
cleared_automaton1=ProbabilisticAutomata.cleared_automaton(miror_miror)
print("cleared_automaton",cleared_automaton1)




