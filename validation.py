"""Check the recognised givens for row, column, and box conflicts."""


def validate_grid(grid):
    # Check the 9x9 shape and allow only integers 0-9, with 0 meaning empty.
    if len(grid) != 9:
        raise ValueError('expected nine rows')
    for row in grid:
        if len(row) != 9 or any(type(digit) is not int or digit < 0 or digit > 9 for digit in row):
            raise ValueError('each row must contain nine integers in 0..9')
    # Each row, column, and 3x3 box is a group that cannot repeat a given digit.
    units = []
    for row in range(9):
        units.append([(row, column) for column in range(9)])
    for column in range(9):
        units.append([(row, column) for row in range(9)])
    # Step by three to visit the top-left cell of each box.
    for box_row in range(0, 9, 3):
        for box_column in range(0, 9, 3):
            cells = []
            for row in range(box_row, box_row + 3):
                for column in range(box_column, box_column + 3):
                    cells.append((row, column))
            units.append(cells)
    # A set avoids listing a cell twice if it conflicts in more than one group.
    conflicts = set()
    for unit in units:
        # Ignore zeros: several empty cells in a group are allowed.
        for digit in range(1, 10):
            matches = []
            for row, column in unit:
                if grid[row][column] == digit:
                    matches.append((row, column))
            if len(matches) > 1:
                # Report every cell with the repeated digit, including the first one.
                conflicts.update(matches)
    # This checks the recognised givens, without solving the puzzle.
    verdict = 'invalid' if conflicts else 'valid'
    # Sort coordinates so the returned conflict list has a consistent order.
    return verdict, [list(cell) for cell in sorted(conflicts)]
