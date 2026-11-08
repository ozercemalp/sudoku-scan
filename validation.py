"""Check the recognised givens for row, column, and box conflicts."""


def validate_grid(grid):
    if len(grid) != 9:
        raise ValueError('expected nine rows')
    for row in grid:
        if len(row) != 9 or any(type(digit) is not int or digit < 0 or digit > 9 for digit in row):
            raise ValueError('each row must contain nine integers in 0..9')
    # no repeated digits in a row column or 3x3 box
    units = []
    for row in range(9):
        units.append([(row, column) for column in range(9)])
    for column in range(9):
        units.append([(row, column) for row in range(9)])
    for box_row in range(0, 9, 3):
        for box_column in range(0, 9, 3):
            cells = []
            for row in range(box_row, box_row + 3):
                for column in range(box_column, box_column + 3):
                    cells.append((row, column))
            units.append(cells)
    # same cell might conflict in multiple groups so use a set
    conflicts = set()
    for unit in units:
        # skip zeros since empty cells can repeat
        for digit in range(1, 10):
            matches = []
            for row, column in unit:
                if grid[row][column] == digit:
                    matches.append((row, column))
            if len(matches) > 1:
                # include all cells with this digit even the first one
                conflicts.update(matches)
    verdict = 'invalid' if conflicts else 'valid'
    # sort so the cell order stays the same between runs
    return verdict, [list(cell) for cell in sorted(conflicts)]
