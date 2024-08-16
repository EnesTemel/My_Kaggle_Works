import argparse  # Parse command line arguments and convert them into Python objects
import time  # Functions related to time

import pandas as pd  # Library for importing CSV files
import numpy as np  # Library for matrix operations
from tqdm import tqdm  # Progress bar library to show how much of the process has been executed
import os

# Inherits from Python's native RuntimeError, throws an exception if exceeding the maximum size
class ExceedMaxSizeError(RuntimeError):
    pass  # Placeholder, does nothing

# Get the shortest path, 'moves' are the feasible +/- operations for this puzzle type, K is the number of steps, max_size is an integer or None.
def get_shortest_path(moves, K, max_size):
    n = len(next(iter(moves.values())))  # Get the length of the first value in the dictionary

    state = tuple(range(n))  # Tuple of (0,1, …, n)
    cur_states = [state]  # Initial state [(0,1, …, n)]

    shortest_path = {}  # Dictionary for the shortest path
    shortest_path[state] = []  # {state: []}
    # Somewhat similar to a linked list
    for _ in range(100 if K is None else K):  # If no K, it's 100.
        next_states = []
        for state in cur_states:  # Take out the tuple (0,1,2, …, n)
            for move_name, perm in moves.items():  # The name of the operation and the specific operation
                if np.random.rand() < 0.5:  # random_prune
                    next_state = tuple(state[i] for i in perm)  # state[i] is the new state
                    # If next_state is in the shortest path, it means a loop is formed, so skip
                    if next_state in shortest_path:
                        continue
                    # If not, shortest_path = {state: [], next_state: [move_name]}
                    shortest_path[next_state] = shortest_path[state] + [move_name]
                    # If max_size is not None and the length of the path exceeds max_size, throw an exception
                    if (max_size is not None) and (len(shortest_path) > max_size):
                        raise ExceedMaxSizeError
                    # Add next_state to next_states
                    next_states.append(next_state)
        cur_states = next_states

    return shortest_path

# 'cube_2/2/2' Add both positive and negative operations of puzzle_type to the dictionary.
def get_moves(puzzle_type):
    # Here eval and json.load have the same effect, finding the "allowed_moves" dictionary corresponding to puzzle_type.
    moves = eval(pd.read_csv("puzzle_info.csv").set_index("puzzle_type").loc[puzzle_type, "allowed_moves"])
    # Add reverse moves to the dictionary
    for key in list(moves.keys()):
        # np.argsort: Indices of the sorted array, e.g., s=[2,0,1] -> s'=[1,2,0]
        # Originally, the value at position 2 of the array was assigned to position 0, now the value at position 0 of the array is assigned to position 2.
        moves["-" + key] = list(np.argsort(moves[key]))
    return moves

def solution():
    parser = argparse.ArgumentParser()  # Create a command-line argument parser
    parser.add_argument("--problem_id", type=int, required=True)  # The program must have an int type problem_id when running
    # Define the command-line argument time_limit, set as a float, default value is 2 hours
    parser.add_argument("--time_limit", type=float, default=2 * 60 * 60)
    args = parser.parse_args()  # Parse these command-line arguments, return a namespace

    # Import the file, set id as the index, and take the corresponding data based on the passed parameter args.problem_id
    puzzle = pd.read_csv("puzzles.csv").set_index("id").loc[args.problem_id]
    print(f"problem_id:{args.problem_id}")
    submission = pd.read_csv("submission.csv").set_index("id").loc[args.problem_id]
    # Convert the submission example's "r1.-f1" into ['r1', '-f1']
    sample_moves = submission["moves"].split(".")
    # Print the number of steps required for the submission example
    moves = get_moves(puzzle["puzzle_type"])
    # Print the number of moves

    K = 2
    while True:
        try:
            # When k=2, take 2 steps, no limit on max_size, if there are many paths limit it to 1000000
            shortest_path = get_shortest_path(moves, K, None if K == 2 else 1000000)
        except ExceedMaxSizeError:  # If an exception is thrown inside try
            break  # Stop
        K += 1
    # All states that can be reached in K steps, K's value depends on 1000000, here shortest_path is the shortest_path obtained by the last K before the exception is thrown.
    print(f"K: {K}, Number of shortest_path: {len(shortest_path)}")

    # Initial state
    current_state = puzzle["initial_state"].split(";")
    current_solution = list(sample_moves)  # The solution list of the submission example
    initial_score = len(current_solution)  # Initial score, the more steps, the higher the score
    start_time = time.time()  # Set the start time

    # Create a progress bar with tqdm(...), the number of iterations is len(current_solution) - K,
    # Description shown below the progress bar: Score is the length of the current solution, -0 is the tolerance count.
    with tqdm(total=len(current_solution) - K, desc=f"Score: {len(current_solution)} (-0)") as pbar:
        step = 0
        # Check if the time limit is exceeded
        # It checks the possibility of optimization in the steps [step, step+K+1].
        while step + K < len(current_solution) and (time.time() - start_time < args.time_limit):
            # Take K+1 actions of the current solution [step, step+K]
            replaced_moves = current_solution[step: step + K + 1]
            # Initialize state_before and state_after to the initial state
            state_before = current_state
            state_after = current_state
            # state_after reaches the Kth state (keeping the first K solutions unchanged)
            for move_name in replaced_moves:
                state_after = [state_after[i] for i in moves[move_name]]

            # shortest_path are all states reachable in K steps
            found_moves = None  # Found a better method
            # For perm: (0,1,2,3,4, …, n) move_names: ['f1', 'r1', …]
            for perm, move_names in shortest_path.items():
                # For perm=(1,2,0), (i,j)=(0,1),(1,2),(2,0), i is the index of perm, j is the value of perm
                for i, j in enumerate(perm):
                    if state_after[i] != state_before[j]:  # If any are not equal, break the inner for loop
                        break
                else:  # state_after is reached by state_before in K+1 steps, but perm is a method within K steps
                    found_moves = move_names  # Found a more optimal method
                    break

            if found_moves is not None:  # If a better method is found
                length_before = len(current_solution)  # Number of steps in the previous method
                # The current solution is: first step unchanged + new method found + later solution unchanged
                current_solution = current_solution[:step] + list(found_moves) + current_solution[step + K + 1:]
                pbar.update(length_before - len(current_solution))  # Progress bar moves forward by the reduced number of steps
                # Progress bar shows current score, how much the current solution is optimized compared to the initial solution
                pbar.set_description(f"Score: {len(current_solution)} ({len(current_solution) - initial_score})")
                for _ in range(K):
                    if step:  # If step is not 0, it can continue to go back
                        step -= 1  # Go back one step
                        pbar.update(-1)  # Progress bar moves back one step
                        move_name = current_solution[step]  # Take out this step's solution
                        # If there's a '-', remove it, if not, add it
                        move_name = move_name[1:] if move_name.startswith("-") else f"-{move_name}"
                        # Go back to the previous state
                        current_state = [current_state[i] for i in moves[move_name]]
            else:  # If no better method is found, move forward one step
                current_state = [current_state[i] for i in moves[current_solution[step]]]
                step += 1
                pbar.update(1)  # Progress bar moves forward one unit
    # Write the final solution found, joining with '.'
    solutions_dir = r"C:\Users\enesm\OneDrive\Masaüstü\KAGGLE\Guts\kaggle\working\solutions"
    os.makedirs(solutions_dir, exist_ok=True)  # Ensure the directory exists

    solution_file = f"kaggle/working/solutions/{args.problem_id}.txt"


    # Write the final solution found, joining with '.'
    try:
        with open(solution_file, "w") as f:
            f.write(".".join(current_solution))
        print(f"Solution written to {solution_file}")
    except Exception as e:
        print(f"Failed to write solution: {e}")
#调用解决问题的函数
solution()
