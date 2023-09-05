from collections import defaultdict
import numpy as np
import math


class ProbabilisticAutomata :

#Classical functions of an oriented object structure

    def __init__(self, alphabet, states, initial_states, set_of_transition_matrix, final_states):
        """
        :param alphabet: set of symbols
        :param states: set of states
        :param ini: set of initial states
        :param trans: dictionary giving transitions
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
        res += "Transition matrix"+ "\n"
        for source in self.set_of_transition_matrix :
            res += str(source) + "\n"
            for label in self.set_of_transition_matrix[source]:
                res += str(label) +"\n"
        return res


#Firstly, we have to built tools to implement our algorithm

    def min_greater_than_zero(self,vector) :

        """
        returns the strictly positive minimum of the vector coordinates

        """
        #aberant value, useful for the next function "vector_decomposition"

        min_vector = 1.1
        for index in vector :
            
            if index[0] < min_vector and index[0] > 0 :

                min_vector = index[0]
        return min_vector 

    def vector_decomposition(self,vector) :

        """
        returns the decomposition of a vector with factor in [0,1] 

        """
         
        min_vector = self.min_greater_than_zero(vector)
        list_of_new_vectors = []

        #use of the aberant value explained above

        while min_vector != 1.1 :
            new_vector = np.array([[0]] * (np.size(vector)))
            for index in range (0,np.size(vector)) :

                if vector[index, 0] != 0 :
                    new_vector[index, 0] = 1

            list_of_new_vectors.append((min_vector, new_vector))
            rest_vector = (vector - (new_vector * min_vector))
            vector = rest_vector
            min_vector = self.min_greater_than_zero(vector)
            
        print("list_of_new_vectors",list_of_new_vectors)
        return list_of_new_vectors

    def convertisor_column_vector_to_an_integer(self, vector):

        """
        returns the integer correspoonding to the binary vector

        """

        c = 0
        for index in range (0, np.size(vector)) :
            if vector [index, 0] == 1:
                c += 2**(index)
        return c
    
    def from_integer_to_binary_on_n_bits(self,n,bits):
        return format(n, 'b').zfill(bits)

    def minus_one_array(self,array) :
        L = []
        for index in range (len(array)) :
            L.append(-1)
        Array = [array_elt + minus_one_elt for minus_one_elt, array_elt in zip(L, array)]
        return Array
        

#Then we have to deal with the new caracteristics of the miror automata

#générer des vecteurs aléatoires probabilistes pour tester l'algo naif
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
            
            #We take the equivalent binary number with the python function bin() and we reverse it without the prefixe 0b

            #binary_states = bin(vector_index).ljust((dimension+2),"0")[:1:-1]

            binary_states = self.from_integer_to_binary_on_n_bits(vector_index, dimension)
            new_vector = np.array(list(map(lambda x : [x], list(map(int,list(reversed(binary_states)))))))
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
        decomposed_vector_list = self.vector_decomposition(self.final_states)
        for element in decomposed_vector_list :
            probability = element[0]
            vector = element[1]
            state = self.convertisor_column_vector_to_an_integer(vector)
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
            
            for element in decomposed_vector :
            
                probability = element[0]
                vector = element[1]
                matrix_index = self.convertisor_column_vector_to_an_integer(vector)
                new_transition_matrix[vector_index][matrix_index] = probability

            # for probability in decomposed_vector :

            #     for element in range (len(decomposed_vector[probability])) :

            #         matrix_index = self.convertisor_column_vector_to_an_integer(decomposed_vector[probability][element])
            #         new_transition_matrix[vector_index][matrix_index] = probability

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

            for index_global_verification in range (np.size(self.initial_states)) : 
                if global_verification[index_global_verification] == 0 :
                    line_cleared_matrix = np.delete(new_cleared_transition_state[label],index_global_verification,0)
                    cleared_matrix = np.delete(line_cleared_matrix,index_global_verification,1)
                    new_cleared_transition_state[label] = cleared_matrix
                    new_cleared_initial_states = np.delete(new_cleared_initial_states, index_global_verification,1)
                    new_cleared_final_state = np.delete(new_cleared_final_state,index_global_verification,0)
                    

            new_cleared_states = set()
            for state_index in range (np.size(new_cleared_initial_states)) : 
                new_cleared_states.add(state_index)
        
        return ProbabilisticAutomata(self.alphabet,new_cleared_states,new_cleared_initial_states,new_cleared_transition_state,new_cleared_final_state)
        
#The new miror automaton
    def miror_automaton(self) :
        
        return ProbabilisticAutomata(self.alphabet, self.miror_states(), self.miror_initial_states(),self.new_set_of_transition_matrix(),self.miror_final_states())

#First idea, before the function reachable_states, check if a line and a column are null for all the labels.
    def miror_cleared_automaton(self) :
        if np.size(self.initial_states) == 1 :
            
            return self
        else :
            
            self = self.miror_automaton()
            cleared_one = self.cleared_useless_state()
            return cleared_one

#Automata transversal, this function will keep only the reachable states and the designated initial_states      
    def reachable_states(self) :

        # global memory list 
        global_reachable_states = []
            
        # memory list for all the labels
        array_reachable_states = []

            # initialization with empty vector, this will be the memory list for a label
        array_present_element = []

        list_null = [0*(np.size(self.initial_states))]
        for label in self.alphabet :
            list_null += self.check_if_line_and_column_null(self.set_of_transition_matrix[label])
            
            # we check all the reachable state, via each coeficient of initial_states (line matrix ) not null
        for vector_element in range (np.size(self.initial_states)) :
            print("vector_element",vector_element)

            # We do not want the same index several times and if the initial states must be linked by a transition to the 
            # other states to be counted
            print(self.initial_states[0][vector_element])
            if self.initial_states[0][vector_element] not in array_reachable_states and self.initial_states[0][vector_element] != 0 :
                if list_null[vector_element] != 0 :
                    print("a")
                    array_reachable_states.append(vector_element)
                    array_present_element.append(vector_element)
            print("array_reachable_states",array_reachable_states)
            print("array_present_element",array_present_element)
            
        #Initialization
        #index_array design the index in array reachable_states
                
            while array_reachable_states != [] :
                 # coefficient_element_index an element in the line matrix 
                 for label in self.alphabet :

                    list_null = self.check_if_line_and_column_null(self.set_of_transition_matrix[label])
                    
                    if array_reachable_states != [] :
                        index_array = 0
                        matrix = self.set_of_transition_matrix[label]
                
                #matrix_index is a value in the global_reachable_state_array
                    print("index_array",index_array)
                    print("global_reachable_states",global_reachable_states)
                    matrix_index = array_reachable_states[index_array]
                # this the useless line of the matrix 
                    line_matrix = matrix[matrix_index]
                    counter_deleted_list_element = 0
                    for coefficient_element_index in range (np.size(self.initial_states)) :
                        coefficient_element = line_matrix[coefficient_element_index]

                        # We check if the matrix coefiecient representing the probabibility is not null (reachable states)

                        if coefficient_element_index not in array_present_element and coefficient_element != 0.0 :
                            array_reachable_states.append(coefficient_element_index)
                            array_present_element.append(coefficient_element_index)

                        # We delete the index_array because we have already check this
                    print(counter_deleted_list_element)
                    array_reachable_states.pop(index_array + counter_deleted_list_element)

                    counter_deleted_list_element -=1
                    
                        # avoid error "out of range"
                    if array_reachable_states != [] :
                        matrix_index = array_reachable_states[index_array + counter_deleted_list_element]
                        line_matrix = matrix[matrix_index]
                            
                        
                    else :
                        for index in range (len(array_present_element)) :
                        
                            if array_present_element[index] not in global_reachable_states :
                                global_reachable_states.append(array_present_element[index])

                    
                        break
                
                        # After this, we have to sort the reachable states in the global list, the reachable states of all the labels
                    
                    for index in range (len(array_present_element)) :
                        
                        if array_present_element[index] not in global_reachable_states :
                            global_reachable_states.append(array_present_element[index])

                    index_array+=1
            print("global_reachable_states",global_reachable_states)

        
             
        # To simplify the code, we just want the not reachable states
        array_not_accesible_state =[]
            
        for state in self.states :
            
            if state not in global_reachable_states :
                array_not_accesible_state.append(state)
        
        # We can now count how many states we have to construct the new states of the new automata

        new_states = {i for i in range (len(global_reachable_states))}
        self.states = new_states

        # We also have to change the initial states

        new_initial_states = self.initial_states
        counter_deleted_index = 0

        for matrix_index_initial_states in range (0,len(array_not_accesible_state)) :
            deleted_element = array_not_accesible_state[matrix_index_initial_states]+counter_deleted_index
            new_initial_states = np.delete(new_initial_states,deleted_element,1)
            counter_deleted_index -= 1
        


        self.initial_states = new_initial_states


        # For each label, we are checking the usefulness of each state, checking the transitions. 
        # If there is no usefull transition leading to a final state, this one is deleted

        #Initialization 
        new_cleared_final_state = self.final_states
        new_set_of_transition_matrix =  {}

        
        for label in self.alphabet :
            
            # Because the matrix size will change at each deletion, we need a counter
            counter_deleted_line=0
            matrix = self.set_of_transition_matrix[label]
            copy_global_reachable_states = global_reachable_states
            print("matrix",matrix)

            # We loop in array_not_accessible state to delete them after
            print("global_reachable_states",global_reachable_states)
            for line_index_matrix in range (0,len(global_reachable_states)) :
                print("a")

                if global_reachable_states != []:
                    print("b")
                    future_deleted_line = copy_global_reachable_states[line_index_matrix+counter_deleted_line]
                    print("c")
                    line_cleared_matrix = np.delete(matrix,future_deleted_line,0)
                    print("d")
                    cleared_matrix = np.delete(line_cleared_matrix,future_deleted_line,1)
                    print("e")
                    matrix = cleared_matrix
                    print("f")
                    new_cleared_final_state = np.delete(new_cleared_final_state,line_index_matrix + counter_deleted_line,0)
                    print("g")
                    copy_global_reachable_states.pop(line_index_matrix + counter_deleted_line)
                    print("h")
                    copy_global_reachable_states = self.minus_one_array(copy_global_reachable_states)
                    print("i")
                    counter_deleted_line -= 1
                    
            
            new_set_of_transition_matrix[label]= matrix    
            
        self.set_of_transition_matrix = new_set_of_transition_matrix
        self.final_states = new_cleared_final_state
        

        return self

# Global minimisation miror function
    def miror_minimised_automata(self):
        self = self.miror_automaton()
        return self.reachable_states()

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

#Potentiel de bug si pas dans l'ordre 