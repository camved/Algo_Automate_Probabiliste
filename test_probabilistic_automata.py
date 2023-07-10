import numpy as np
from probabilistic_automata import ProbabilisticAutomata

#################### numpy test ####################

x = np.array([[0,0,0],[0,5,6],[0,8,9]])

v = np.array([[1],[2],[3],[4]])

z = np.array([[1,2,3]])

i = np.array([[0] * 4])

a = np.delete(x,0,0)

b = np.delete(a,0,1)

c = np.delete(z,0,1)

#################### method tests ####################

Sigma = {'a','b'}
Q = {0, 1}
lambdA = np.array([[1/2, 1/2]])
mu = {'a' : np.array([[0,1],[1/2,1/2]]), 'b' : np.array([[3/4,1/4],[1,0]])}
gamma = np.array([[3/4],[1/4]])
vector = np.array([[1],[1]])

PA = ProbabilisticAutomata(Sigma, Q, lambdA, mu, gamma)

Sigmabis = {'a','b'}
Qbis = {0, 1}
lambdAbis = np.array([[1/2, 1/2]])
mubis = {'a' : np.array([[0,0],[0,1]]), 'b' : np.array([[0,0],[0,1]])}
gammabis = np.array([[3/4],[1/4]])
vectorbis = np.array([[1],[1]])

PAbis =ProbabilisticAutomata(Sigmabis,Qbis,lambdAbis,mubis,gammabis)

#print(ProbabilisticAutomata.min_greater_than_zero(PA, gamma))

#print(ProbabilisticAutomata.vector_decomposition(PA, gamma))

#print(ProbabilisticAutomata.miror_final_states(PA))

#print(ProbabilisticAutomata.convertisor_column_vector_to_an_integer(PA,vector ))

#print(ProbabilisticAutomata.miror_vectors(PA))

#print(ProbabilisticAutomata.miror_final_states(PA))

#print(ProbabilisticAutomata.miror_states(PA))

#print(ProbabilisticAutomata.miror_initial_states(PA))

#print(ProbabilisticAutomata.new_transition_matrix(PA, 'a'))

#print(ProbabilisticAutomata.new_set_of_transition_matrix(PA))
 
#print(ProbabilisticAutomata.check_if_line_and_column_null(PA,x))

#print(ProbabilisticAutomata.cleared_useless_state(PAbis))

#print(ProbabilisticAutomata.cleared_useless_state(PA))

#print(ProbabilisticAutomata.miror_cleared_automaton(ProbabilisticAutomata.miror_cleared_automaton(PAbis)))

#print(ProbabilisticAutomata.miror_automaton(PAbis)

#print(ProbabilisticAutomata.word_probability(PA,'ab'))

#print(ProbabilisticAutomata.word_probability(PA,'cab'))