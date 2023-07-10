
import numpy as np
import math


class ProbabilisticAutomata :

#Classical functions of an oriented object structure

    def __init__(self, alphabet, states, initial_states, set_of_transition_matrix, final_states):
        """
        :param alphabet: set of symbols
        :param states: set of states
        :param trans: dictionary giving transitions
        :param ini: set of initial states
        :param final: set of final states
        """
        self.alphabet = alphabet
        self.states = states
        self.initial_states = initial_states
        self.set_of_transition_matrix = set_of_transition_matrix
        self.final_states = final_states

    def __str__(self):
        """ Overrides print function """
        res = "Display Automaton\n"
        res += "Alphabet: " + str(self.alphabet) +"\n"
        res += "Set of states: "+str(self.states) +"\n"
        res += "Initial states "+str(self.initial_states) + "\n"
        res += "Final states "+str(self.final_states) + "\n"

        for source in self.set_of_transition_matrix:
            res += str(source) + "\n"
            for label in self.set_of_transition_matrix[source]:
                    res += str(label) +"\n"
        return res


#Firstly, we have to built tools to implement our algorithm

    def min_greater_than_zero(self,vector) :

        """
        returns the strictly positive minimum of the vector coordinates

        """
        min = 1.1
        for index in vector :
            if index[0] < min and index[0] > 0 :
                min = index[0]
        return min 

    def vector_decomposition(self,vector) :

        """
        returns the decomposition of a vector with factor in [0,1] 

        """
         
        min_vector = self.min_greater_than_zero(vector)
        dictionary_of_new_vectors = {}

        while min_vector != 1.1 :
            new_vector = np.array([[0]] * (np.size(vector)))
            for index in range (0,np.size(vector)) :

                if vector[index, 0] != 0 :
                    new_vector[index, 0] = 1
                    
            dictionary_of_new_vectors[min_vector] = new_vector
            rest_vector = (vector - (new_vector * min_vector))
            vector = rest_vector
            min_vector = self.min_greater_than_zero(vector)
            

        return dictionary_of_new_vectors

    def convertisor_column_vector_to_an_integer(self, vector):

        """
        returns the integer correspoonding to the binary vector

        """

        c = 0
        for index in range (0, np.size(vector)) :
            if vector [index, 0] == 1:
                c += 2**(index)
        return c
    
#Then we have to deal with the new caracteristics of the miror automata

    def miror_states(self) : 

        """
        Return the new set of states of the miror automata

        """

        new_miror_states = set()
        for vector_index in range (2**len(self.states)) :
            new_miror_states.add(vector_index)
        return new_miror_states

    def miror_vectors(self):
        
        """
        returns a new set of vectors corresponding to the miror automata states 

        """

        new_states = {}
        dimension = len(self.states)

        for vector_index in range(2**dimension) :
            
            binary_states = bin(vector_index).ljust((dimension+2),"0")[:1:-1]
            new_vector = np.array(list(map(lambda x : [x], list(map(int,list(binary_states))))))
            new_states[vector_index] = new_vector

        return new_states

    def miror_final_states(self) :

        """
        returns the new final states vector of the miror automata, by the product of the initial states line vector 
        by the vectors representing the states of the miror automata.
        
        """

        new_final_states = np.array([[0]] * 2**(np.size(self.initial_states)), dtype=object)
        set_of_miror_vectors = self.miror_vectors()
        
        for index_vector in range (2**(len(self.states))) :
            vector = set_of_miror_vectors[index_vector]
            probability = np.dot(self.initial_states, vector)
            new_final_states[index_vector][0] = probability[0][0]

        return new_final_states
         
    def miror_initial_states(self) :
        
        """
        returns the new initial states vector of the miror automata

        """

        new_initial_state = np.array(np.array([[0] * (2**(np.size(self.final_states)))]), dtype=object)
        decomposed_vector_set = self.vector_decomposition(self.final_states)

        for probability in decomposed_vector_set :
            state = self.convertisor_column_vector_to_an_integer(decomposed_vector_set[probability])
            new_initial_state[0][state] = probability

        return new_initial_state
    
    def new_transition_matrix(self,label) :

        """
        return the new transition matrix of the automaton
        
        """
        
        set_of_miror_vectors = self.miror_vectors()
        new_transition_matrix = np.zeros((2**(np.size(self.initial_states)),2**(np.size(self.initial_states))))

        for vector_index in set_of_miror_vectors : 
            vector = set_of_miror_vectors[vector_index]
            new_vector = np.dot(self.set_of_transition_matrix[label],vector)
            decomposed_vector = self.vector_decomposition(new_vector)

            for probability in decomposed_vector :
                matrix_index = self.convertisor_column_vector_to_an_integer(decomposed_vector[probability])
                new_transition_matrix[vector_index][matrix_index] = probability

        return new_transition_matrix
    
    def new_set_of_transition_matrix(self) :

        """
        returns the new set of matrix transition of the miror automata

        """

        new_dict = {}
        for label in self.alphabet :
            new_dict[label] = self.new_transition_matrix(label)
        return new_dict
    
#If it is possible, we want to delete the useless state of our miror automaton.

    def check_if_line_and_column_null(self,matrix) :

        """
        It is just a test to check if a column and a line in the matrix are empty
        """

        sum_matrix_column_index = []
        sum_matrix_line_index = []

        for line_index in range(len(matrix[1])) :
            sum_matrix_line_index.append(np.sum(matrix[line_index]))

            sum_column = 0
            for column_index in range (len(matrix[1])) : 
                sum_column += matrix[column_index][line_index]
            sum_matrix_column_index.append(sum_column)
        
        sum_line_and_column =[]

        for i in range (len(matrix[1])) :
            sum_line_and_column.append(sum_matrix_column_index[i]+sum_matrix_line_index[i])
            
        return sum_line_and_column
        
    def cleared_useless_state(self) : 

        """
        returns a matrix without a line and a column fullof zero.

        """
        
        new_cleared_transition_state = {}
        global_verification = [0]*(np.size(self.initial_states))

        for label in self.alphabet :

            check = self.check_if_line_and_column_null(self.set_of_transition_matrix[label])
            
            for index in range (len(check)) :
                global_verification[index] = global_verification[index] + check[index]


        for label in self.alphabet :

            new_cleared_transition_state[label] = self.set_of_transition_matrix[label]
            new_cleared_initial_states = self.initial_states
            new_cleared_final_state = self.final_states

            for index_global_verification in range (0,len(global_verification)) :  
                
                if global_verification[index_global_verification] == 0 :
                    line_cleared_matrix = np.delete(self.set_of_transition_matrix[label],index_global_verification,0)
                    cleared_matrix = np.delete(line_cleared_matrix,index_global_verification,1)
                    new_cleared_transition_state[label] = cleared_matrix

                    new_cleared_initial_states = np.delete(self.initial_states, index_global_verification,1)

            new_cleared_states = set()
            for state_index in range (np.size(new_cleared_initial_states)) : 
                new_cleared_states.add(state_index)

        return ProbabilisticAutomata(self.alphabet,new_cleared_states,new_cleared_initial_states,new_cleared_transition_state,new_cleared_final_state)
        

#The new miror automaton

    def miror_automaton(self) :
        return ProbabilisticAutomata(self.alphabet, self.miror_states(), self.miror_initial_states(),self.new_set_of_transition_matrix(),self.miror_final_states())
    
    def miror_cleared_automaton(self) :
        if np.size(self.initial_states) == 1 :
            return self
        else :
            return self.cleared_useless_state()
    
#Check the acceptance probability of a word

    def word_probability(self, word) :
        Probability = self.initial_states
        for label in (word):
            if label in self.set_of_transition_matrix.keys() :
                Probability = np.dot(Probability, self.set_of_transition_matrix[label])
            else :
                return 0
        Probability = np.dot(Probability,self.final_states)
        return Probability