import enum
import math
import time
import numpy as np
import random

class AIStrategy(enum.Enum):
    """Enumeration of available AI strategies."""
    EXPECTIMAX = 1
    MINIMAX = 2
    MCTS = 3

class MCTSNode:
    """Node representation for Monte Carlo Tree Search."""
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.total_score = 0
        self.untried_moves = board.get_available_moves()

    def uct_score(self, exploration_constant=1.414):
        """Calculate UCT score for node selection."""
        if self.visits == 0:
            return float('inf')
        return (self.total_score / self.visits) + \
               exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

class AI:
    """AI class implementing different algorithms for playing 2048."""
    def __init__(self, strategy=AIStrategy.EXPECTIMAX):
        self.strategy = strategy

    def get_move(self, board):
        """Select best move based on chosen strategy."""
        if self.strategy == AIStrategy.EXPECTIMAX:
            best_move, _ = self.maximize(board, max_depth=3)
            return best_move
        elif self.strategy == AIStrategy.MINIMAX:
            return self.minimax_move(board, max_depth=3)
        elif self.strategy == AIStrategy.MCTS:
            return self.mcts_move(board, time_budget=0.05)
        else:
            raise ValueError("Invalid strategy selected")

    def eval_board(self, board, n_empty): 
        """Evaluate board state based on multiple heuristics."""
        grid = board.grid

        utility = 0
        smoothness = 0

        # Calculate total weighted tiles
        big_t = np.sum(np.power(grid, 2))
        
        # Calculate smoothness by comparing adjacent tiles
        s_grid = np.sqrt(grid)
        
        # Horizontal smoothness
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]-1):
                smoothness -= np.abs(s_grid[i,j] - s_grid[i,j+1])
        
        # Vertical smoothness
        for j in range(grid.shape[1]):
            for i in range(grid.shape[0]-1):
                smoothness -= np.abs(s_grid[i,j] - s_grid[i+1,j])
        
        # Weights for different aspects of the board
        empty_w = 100000
        smoothness_w = 3

        # Calculate utility components
        empty_u = n_empty * empty_w
        smooth_u = smoothness ** smoothness_w
        big_t_u = big_t

        utility += big_t
        utility += empty_u
        utility += smooth_u

        return (utility, empty_u, smooth_u, big_t_u)

    def maximize(self, board, depth=0, max_depth=3):
        """Expectimax algorithm's maximizing step."""
        # Early termination if max depth reached
        if depth >= max_depth:
            return None, self.eval_board(board, len(board.get_available_cells()))
        
        moves = board.get_available_moves()
        moves_boards = []

        for m in moves:
            m_board = board.clone()
            m_board.move(m)
            moves_boards.append((m, m_board))

        max_utility = (float('-inf'),0,0,0)
        best_direction = None

        for mb in moves_boards:
            utility = self.chance(mb[1], depth + 1)

            if utility[0] >= max_utility[0]:
                max_utility = utility
                best_direction = mb[0]

        return best_direction, max_utility

    def chance(self, board, depth = 0):
        """Expectimax algorithm's chance step."""
        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        # Depth limits adjusted based on grid complexity
        complexity_factor = board.grid_len / 4  # Normalize to 4x4 grid
        max_depth = max(3, int(5 / complexity_factor))

        if n_empty >= board.grid_len * 2 and depth >= max_depth:
            return self.eval_board(board, n_empty)

        if n_empty == 0:
            _, utility = self.maximize(board, depth + 1)
            return utility

        possible_tiles = []

        chance_2 = (.9 * (1 / n_empty))
        chance_4 = (.1 * (1 / n_empty))
        
        for empty_cell in empty_cells:
            possible_tiles.append((empty_cell, 2, chance_2))
            possible_tiles.append((empty_cell, 4, chance_4))

        utility_sum = [0, 0, 0, 0]

        for t in possible_tiles:
            t_board = board.clone()
            t_board.insert_tile(t[0], t[1])
            _, utility = self.maximize(t_board, depth + 1)

            for i in range(4):
                utility_sum[i] += utility[i] * t[2]

        return tuple(utility_sum)

    def minimax_move(self, board, max_depth=4):
        """Minimax algorithm for move selection."""
        def minimax(board, depth, is_maximizing):
            # Check terminal conditions
            if depth == 0 or len(board.get_available_moves()) == 0:
                return self.eval_board(board, len(board.get_available_cells()))[0]
            
            if is_maximizing:
                # Player's turn (maximizing)
                max_eval = float('-inf')
                moves = board.get_available_moves()
                
                for move in moves:
                    board_copy = board.clone()
                    board_copy.move(move)
                    
                    # Simulate worst-case tile placement (always place 4)
                    empty_cells = board_copy.get_available_cells()
                    if empty_cells:
                        worst_tile_pos = empty_cells[0]  # Pessimistic approach
                        board_copy.insert_tile(worst_tile_pos, 4)
                    
                    eval_score = minimax(board_copy, depth - 1, False)
                    max_eval = max(max_eval, eval_score)
                
                return max_eval
        
            else:
                # Tile placement turn (minimizing)
                min_eval = float('inf')
                empty_cells = board.get_available_cells()
                
                for cell in empty_cells:
                    board_copy = board.clone()
                    board_copy.insert_tile(cell, 4)  # Worst tile for player
                    
                    eval_score = minimax(board_copy, depth - 1, True)
                    min_eval = min(min_eval, eval_score)
                
                return min_eval
    
        # Select best move by evaluating each possible move
        moves = board.get_available_moves()
        best_move = None
        best_score = float('-inf')
        
        for move in moves:
            board_copy = board.clone()
            board_copy.move(move)
            
            # Simulate worst-case tile placement
            empty_cells = board_copy.get_available_cells()
            if empty_cells:
                worst_tile_pos = empty_cells[0]
                board_copy.insert_tile(worst_tile_pos, 4)
            
            # Evaluate the move
            score = minimax(board_copy, max_depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    def mcts_move(self, board, time_budget=0.05, exploration_constant=1.414):   
        """MCTS for move selection."""
        def simulate_game(board):
            """Simulate a game from the current board state until no moves are possible."""
            simulation_board = board.clone()
            while len(simulation_board.get_available_moves()) > 0:
                move = random.choice(simulation_board.get_available_moves())
                simulation_board.move(move)
                
                # Add random tile
                empty_cells = simulation_board.get_available_cells()
                if empty_cells:
                    tile_value = 2 if random.random() < 0.9 else 4
                    tile_pos = random.choice(empty_cells)
                    simulation_board.insert_tile(tile_pos, tile_value)
            
            # Evaluate final board state
            return self.eval_board(simulation_board, len(simulation_board.get_available_cells()))[0]
    
        def select_and_expand(node):
            """Select a node to expand using UCT."""
            while node.untried_moves == [] and node.children:
                # Use UCT to select child
                node = max(node.children, key=lambda c: c.uct_score(exploration_constant))
            
            if node.untried_moves:
                # Expand a new node
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                new_board = node.board.clone()
                new_board.move(move)
                
                # Add random tile
                empty_cells = new_board.get_available_cells()
                if empty_cells:
                    tile_value = 2 if random.random() < 0.9 else 4
                    tile_pos = random.choice(empty_cells)
                    new_board.insert_tile(tile_pos, tile_value)
                
                child = MCTSNode(new_board, parent=node, move=move)
                node.children.append(child)
                return child
            
            return node
    
        def backpropagate(node, result):
            """Update visit counts and scores up the tree"""
            while node is not None:
                node.visits += 1
                node.total_score += result
                node = node.parent
    
        # Create root node
        root = MCTSNode(board)
        
        # Run MCTS within time budget
        start_time = time.time()
        while time.time() - start_time < time_budget:
            # Selection and expansion
            leaf = select_and_expand(root)
            
            # Simulation
            simulation_result = simulate_game(leaf.board)
            
            # Backpropagation
            backpropagate(leaf, simulation_result)
        
        # Select best move based on most visited child
        best_move = max(root.children, key=lambda c: c.visits).move if root.children else None
        
        return best_move