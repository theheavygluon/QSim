import numpy as np
from functools import reduce
# Matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
zero = np.array([[1],[0]])
one = np.array([[0],[1]])

#CX of a circuit


tensor = lambda *initial_state: reduce(lambda x, y: np.kron(x, y), initial_state)



def cx_matrix(num_qubits, control, target):
    I = np.identity(2)
    left = [I]*num_qubits
    right = [I]*num_qubits
    left[control] = np.dot(zero, zero.T)
    right[control] = np.dot(one, one.T)
    right[target] = X
    return tensor(*left) + tensor(*right)

# Eigenvectors of Pauli Matrices

plus = np.array([[1], [1]])/np.sqrt(2) # X plus basis state
minus = np.array([[1], [-1]])/np.sqrt(2) # X minus basis state

up = np.array([[1], [1j]])/np.sqrt(2) # Y plus basis state
down = np.array([[1], [-1j]])/np.sqrt(2) # Y plus basis state

# Bell States
B00 = np.array([[1], [0], [0], [1]])/np.sqrt(2) # Bell of 00
B01 = np.array([[1], [0], [0], [-1]])/np.sqrt(2) # Bell of 01
B10 = np.array([[0], [1], [1], [0]])/np.sqrt(2) # Bell of 10
B11 = np.array([[0], [-1], [1], [0]])/np.sqrt(2) # Bell of 11

# Rn Matrix Function
Rx = lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
Ry = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
Rz = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])



# return vector of size 2**num_qubits with all zeroes except first element which is 1
def get_ground_state(n):
    zero = [1,0]
    i = 0
    state = [1,0]
    while i<n-1:
        state = np.kron(state,zero)
        i+=1
    return state



# return unitary operator of size 2**n x 2**n for given gate and target qubits
#Problem: Trying to extend it to multiqubit targets (like CNOT)

def get_operator(total_qubits, gate_unitary, target_qubits):
    if type(target_qubits) == int:
        I = np.identity(total_qubits-1)
        i = 0
        state = I
        while i < target_qubits-1:
            state = np.kron(state, I)
            i+=1
        state = np.kron(state, gate_unitary)
        
        while i < total_qubits-2:
            state = np.kron(state, I)
            i+=1   
    else:
        state = print("Adam is working on it")
    return state

from functools import reduce
tensor = lambda *initial_state: reduce(lambda x, y: np.kron(x, y), initial_state)
import numpy as np
zero = np.array([[1],[0]])
one = np.array([[0],[1]])
I = np.identity(2)
def cx_matrix(num_qubits, control, target):

    left = [I]*num_qubits
    right = [I]*num_qubits

    left[control] = np.dot(zero, zero.T)
    right[control] = np.dot(one, one.T)

    right[target] = X

    return tensor(*left) + tensor(*right)


def run_program(initial_state, program):
    # read program, and for each gate:
    #   - calculate matrix operator
    #   - multiply state with operator
    # return final state
    return

def measure_all(state_vector):
    # choose element from state_vector using weighted random and return it's index
    return

def get_counts(state_vector, num_shots):
    # simply execute measure_all in a loop num_shots times and
    # return object with statistics in following form:
    #   {
    #      element_index: number_of_ocurrences,
    #      element_index: number_of_ocurrences,
    #      element_index: number_of_ocurrences,
    #      ...
    #   }
    # (only for elements which occoured - returned from measure_all)
    return

