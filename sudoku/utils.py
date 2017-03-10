rows = 'ABCDEFGHI'
cols = '123456789'

def cross(a, b):
    "Cross product of elements in 'a' and elements in 'b'."
    return [s+t for s in a for t in b]

def reverse_string(string):
    "Return the backwards representation of a string"
    return string[::-1]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
right_diagonal_units = [sum([cross(r, str(i + 1)) for i, r in enumerate(rows)], [])]
left_diagonal_units = [sum([cross(r, str(i + 1)) for i, r in enumerate(reverse_string(rows))], [])]
unitlist = row_units + column_units + square_units + right_diagonal_units + left_diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return
