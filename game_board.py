import numpy as np
from numba import jit

def generate_dirs(grid_len):
    """
    Generate the possible movement directions.
    0: Up, 1: Down, 2: Left, 3: Right
    """
    return list(range(4))

@jit(nopython=True)
def merge(a):
    """
    Merges adjacent identical tiles in each row.
    When two identical values are adjacent, the first one doubles and the second becomes zero.
    JIT-compiled for performance optimization.
    """
    for i in range(a.shape[0]):
        for j in range(a.shape[1] - 1):
            if a[i][j] == a[i][j + 1] and a[i][j] != 0:
                a[i][j] *= 2
                a[i][j + 1] = 0   
    return a

@jit(nopython=True)
def justify_left(a, out):
    """
    Shifts all non-zero values to the left side of each row,
    removing gaps between tiles.
    JIT-compiled for performance optimization.
    """
    for i in range(a.shape[0]):
        c = 0
        for j in range(a.shape[1]):
            if a[i][j] != 0:
                out[i][c] = a[i][j]
                c += 1
    return out

@jit(nopython=True)
def get_available_from_zeros(a):
    """
    Determines possible moves based on zero positions.
    Returns a list of booleans representing possible moves:
    [up, down, left, right]
    JIT-compiled for performance optimization.
    """
    uc, dc, lc, rc = False, False, False, False

    v_saw_0 = np.zeros(a.shape[1], dtype=np.bool_)
    v_saw_1 = np.zeros(a.shape[1], dtype=np.bool_)

    for i in range(a.shape[0]):
        saw_0 = False
        saw_1 = False

        for j in range(a.shape[1]):
            if a[i][j] == 0:
                saw_0 = True
                v_saw_0[j] = True

                if saw_1:
                    rc = True
                if v_saw_1[j]:
                    dc = True

            if a[i][j] > 0:
                saw_1 = True
                v_saw_1[j] = True

                if saw_0:
                    lc = True
                if v_saw_0[j]:
                    uc = True

    return [uc, dc, lc, rc]

class GameBoard:
    def __init__(self, grid_len=4):
        """
        Initialize the 2048 game board with a custom grid size.
        Default size is 4x4 (standard 2048 game).
        """
        self.grid_len = grid_len
        self.grid = np.zeros((grid_len, grid_len))
        self.dirs = generate_dirs(grid_len)

    def clone(self):
        """
        Create a deep copy of the current game board.
        Used for move simulation and evaluation.
        """
        grid_copy = GameBoard(self.grid_len)
        grid_copy.grid = np.copy(self.grid)
        return grid_copy

    def insert_tile(self, pos, value):
        """
        Insert a new tile at the specified position with the given value.
        Typically used to add 2 or 4 tiles after each move.
        """
        self.grid[pos[0]][pos[1]] = value

    def get_available_cells(self):
        """
        Find all empty cells (zeros) in the grid.
        Returns a list of (x,y) coordinates of empty cells.
        """
        cells = []
        for x in range(self.grid_len):
            for y in range(self.grid_len):
                if self.grid[x][y] == 0:
                    cells.append((x,y))
        return cells

    def get_max_tile(self):
        """
        Returns the value of the highest tile on the board.
        Used to track game progress and for evaluation.
        """
        return np.amax(self.grid)

    def move(self, dir, get_avail_call=False):
        """
        Move tiles in the specified direction and merge when possible.
        
        Parameters:
        - dir: Direction (0: Up, 1: Down, 2: Left, 3: Right)
        - get_avail_call: If True, returns whether the move changed the board
        
        Returns:
        - Boolean indicating if the move changed the board state (if get_avail_call is True)
        - None otherwise
        """
        if get_avail_call:
            clone = self.clone()

        z1 = np.zeros((self.grid_len, self.grid_len))
        z2 = np.zeros((self.grid_len, self.grid_len))

        # UP: Transpose and reverse, apply left operations, then reverse transpose
        if dir == 0:
            self.grid = self.grid[:,::-1].T
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid.T[:,::-1]
        # DOWN: Reverse transpose, apply left operations, then transpose reverse
        elif dir == 1:
            self.grid = self.grid.T[:,::-1]
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid[:,::-1].T
        # LEFT: Apply operations directly (left is our base operation)
        elif dir == 2:
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
        # RIGHT: Double reverse, apply left operations, then double reverse again
        elif dir == 3:
            self.grid = self.grid[:,::-1]
            self.grid = self.grid[::-1,:]
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid[:,::-1]
            self.grid = self.grid[::-1,:]

        if get_avail_call:
            return not (clone.grid == self.grid).all()
        else:
            return None

    def get_available_moves(self, dirs=None):
        """
        Determine which moves are currently possible.
        Returns a list of valid move directions (0: Up, 1: Down, 2: Left, 3: Right).
        """
        if dirs is None:
            dirs = self.dirs
        
        available_moves = []
        
        a1 = get_available_from_zeros(self.grid)

        for x in dirs:
            if not a1[x]:
                board_clone = self.clone()

                if board_clone.move(x, True):
                    available_moves.append(x)

            else:
                available_moves.append(x)

        return available_moves

    def get_cell_value(self, pos):
        """
        Get the value of the tile at the specified position.
        """
        return self.grid[pos[0]][pos[1]]
    
    def calculate_move_score(self, original_grid):
        """
        Calculate score earned from a move by comparing with the original grid.
        Scores are based on the values of merged tiles.
        
        Parameters:
        - original_grid: The grid state before the move
        
        Returns:
        - The score earned from merges during this move
        """
        score = 0
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                current_value = self.grid[i][j]
                original_value = original_grid[i][j]
                
                # If current value is larger and not zero, it means a merge happened
                if current_value > original_value and current_value != 0:
                    score += current_value
        
        return score