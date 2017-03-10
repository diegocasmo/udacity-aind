from utils import *
assignments = []

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    for unit in unitlist:
        # Find all instances of naked twins in unit
        naked_twins = find_naked_twins(values, unit)
        if len(naked_twins) > 0:
            for box in unit:
                # Eliminate the naked twins as possibilities for this unit
                values = eliminate_naked_twins_from_unit(values, unit, naked_twins)
    return values

def eliminate_naked_twins_from_unit(values, unit, naked_twins):
    "Eliminate naked twin values from unit"
    for box in unit:
        if len(values[box]) > 2:
            for naked_twin in naked_twins:
                values = assign_value(values, box, eliminate_value(values[box], naked_twin))
    return values

def find_naked_twins(values, unit):
    "Return a string which contains the numbers that belong to this unit naked twins"
    tuples = {}
    for box in unit:
        if len(values[box]) == 2:
            tuples[box] = values[box]
    tuple_values = list(tuples.values())
    naked_twins = [tuple for tuple in tuple_values if has_naked_twin(tuple_values, tuple)]
    return ''.join(set(naked_twins))

def has_naked_twin(value_list, tuple):
    "Return true if 'tuple' has a naked twin in 'value_list'"
    return value_list.count(tuple) > 1 or value_list.count(reverse_string(tuple)) > 1

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    assert len(grid) == 81, "Input grid must be a string of length 81 (9x9)"
    values = dict(zip(boxes, grid))
    empty_box_value = '123456789'
    for box, value in values.items():
        if(value == '.'):
            assign_value(values, box, empty_box_value)
    return values

def eliminate(values):
    """Eliminate values from peers of each box with a single value.
    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        value_to_eliminate = values[box]
        for peer in peers[box]:
            assign_value(values, peer, eliminate_value(values[peer], value_to_eliminate))
    return values

def eliminate_value(curr_value, value_to_eliminate):
    """Eliminate a single value from a string
    Args:
        curr_value: Current value of the string
        value_to_eliminate: Value to be eliminated from the string
    Returns:
        The string with the desired value eliminated from it
    """
    return curr_value.replace(value_to_eliminate, '');

def only_choice(values):
    """Finalize all values that are the only choice for a unit.
    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                assign_value(values, dplaces[0], digit)
    return values

def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = get_num_of_solved_values(values)
        # Use the Eliminate Strategy
        values = eliminate(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Use the Naked Twins Strategy
        values = naked_twins(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = get_num_of_solved_values(values)
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def get_num_of_solved_values(values):
    """Return the number of boxes with a single value
    Input: Sudoku in dictionary form.
    Output: Number of boxes with a single value
    """
    return len([box for box in values.keys() if len(values[box]) == 1])

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # Reduce the puzzle using the previous function
    reduced_values = reduce_puzzle(values)

    # Verify if sudoku was solved
    if reduced_values is False:
        return False # This branch does not lead to a solution
    if is_solved(reduced_values):
        return reduced_values

    # Choose one of the unfilled squares with the fewest possibilities
    unfilled_box = choose_unfilled_box(reduced_values)

    # Use recursion to solve each one of the resulting sudokus
    for value in reduced_values[unfilled_box]:
        new_values = reduced_values.copy()
        new_values[unfilled_box] = value
        solution = search(new_values)
        if solution:
            return solution

def choose_unfilled_box(values):
    "Return sudoku box with the fewest amount of possibilities and its values"
    return min((len(values[box]), box) for box in boxes if len(values[box]) > 1)[1]

def is_solved(values):
    "Return true if sudoku is solved, false otherwise"
    return all(len(values[box]) == 1 for box in values.keys())

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    return search(grid_values(grid))

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
