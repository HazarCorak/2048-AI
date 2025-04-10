import tkinter as tk
import numpy as np
from tkinter import messagebox, simpledialog
from random import randint
import time
from time import perf_counter

from game_board import GameBoard
from ai import AI, AIStrategy

# Configuration constants
DEFAULT_SIZE = 500  # Default window size in pixels
DEFAULT_GRID_LEN = 4  # Default grid size (4x4 for classic 2048)
GRID_PADDING = 10  # Padding between grid cells

# UI color definitions
BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")
SCORE_FONT = ("Verdana", 16)

class MainMenu:
    """Main configuration screen for the 2048 AI game."""
    def __init__(self, master):
        self.master = master
        self.master.title("2048 AI - Game Configuration")
        self.master.geometry("400x650")
        self.master.configure(bg=BACKGROUND_COLOR_GAME)

        self.create_widgets()

    def create_widgets(self):
        """Create all the UI elements for the configuration screen."""
        # Title
        title_label = tk.Label(
            self.master, 
            text="2048 AI", 
            font=("Verdana", 24, "bold"), 
            bg=BACKGROUND_COLOR_GAME, 
            fg="white"
        )
        title_label.pack(pady=20)
    
        # Configuration Frame
        config_frame = tk.Frame(self.master, bg=BACKGROUND_COLOR_GAME)
        config_frame.pack(expand=True)
    
        # Board Size Selection
        size_label = tk.Label(
            config_frame, 
            text="Select Board Size", 
            font=("Verdana", 16), 
            bg=BACKGROUND_COLOR_GAME, 
            fg="white"
        )
        size_label.pack(pady=10)
    
        # Predefined board sizes
        board_sizes = [
            ("3x3", 3),
            ("4x4 (Classic)", 4),
            ("5x5", 5),
            ("Custom Size", 0)
        ]
    
        self.board_size_var = tk.IntVar(value=4)
    
        for label, size in board_sizes:
            btn = tk.Radiobutton(
                config_frame, 
                text=label, 
                variable=self.board_size_var,
                value=size,
                font=("Verdana", 12),
                bg=BACKGROUND_COLOR_GAME,
                activebackground=BACKGROUND_COLOR_GAME
            )
            btn.pack(pady=5)
            
        # Mode Selection Label
        mode_label = tk.Label(
            config_frame, 
            text="Select Game Mode", 
            font=("Verdana", 16), 
            bg=BACKGROUND_COLOR_GAME, 
            fg="white"
        )
        mode_label.pack(pady=10)

        # Game Modes
        modes = [
            ("Continuous Mode", False),
            ("Step-by-Step Mode", True)
        ]

        self.step_mode_var = tk.BooleanVar(value=False)

        for label, step_mode in modes:
            btn = tk.Radiobutton(
                config_frame, 
                text=label, 
                variable=self.step_mode_var,
                value=step_mode,
                font=("Verdana", 12),
                bg=BACKGROUND_COLOR_GAME,
                activebackground=BACKGROUND_COLOR_GAME
            )
            btn.pack(pady=5)

        # Strategy Selection Label
        strategy_label = tk.Label(
            config_frame, 
            text="Select AI Strategy", 
            font=("Verdana", 16), 
            bg=BACKGROUND_COLOR_GAME, 
            fg="white"
        )
        strategy_label.pack(pady=10)

        # Strategy Buttons
        strategies = [
            ("Expectimax", AIStrategy.EXPECTIMAX),
            ("Minimax", AIStrategy.MINIMAX),
            ("Monte Carlo Tree Search", AIStrategy.MCTS)
        ]

        self.strategy_var = tk.StringVar()

        for label, strategy in strategies:
            btn = tk.Radiobutton(
                config_frame, 
                text=label, 
                variable=self.strategy_var,
                value=strategy.name,
                font=("Verdana", 12),
                bg=BACKGROUND_COLOR_GAME,
                activebackground=BACKGROUND_COLOR_GAME
            )
            btn.pack(pady=5)

        # Start Game Button
        start_btn = tk.Button(
            config_frame, 
            text="Start Game", 
            command=self.start_game,
            width=25,
            font=("Verdana", 12)
        )
        start_btn.pack(pady=10)

    def start_game(self):
        """Validate configuration and start the game."""
        # Validate strategy selection
        if not self.strategy_var.get():
            messagebox.showerror("Error", "Please select an AI strategy")
            return
    
        # Handle board size selection
        board_size = self.board_size_var.get()
        if board_size == 0:
            # Open custom size dialog
            board_size = simpledialog.askinteger(
                "Custom Board Size", 
                "Enter board size (min 3, max 10):", 
                minvalue=3,
                maxvalue=10, 
                initialvalue=4
            )
            if board_size is None:
                return  # User canceled

        # Get selected strategy
        strategy = AIStrategy[self.strategy_var.get()]
        step_mode = self.step_mode_var.get()

        # Close the main menu
        self.master.destroy()
        
        # Create game window
        root = tk.Tk()
        game = GameGrid(root, strategy, step_mode, board_size)
        root.mainloop()

class GameGrid(tk.Frame):
    """Main game grid class that handles game logic and UI."""
    def __init__(self, master, strategy, step_mode, grid_len=DEFAULT_GRID_LEN):
        tk.Frame.__init__(self, master)

        self.master = master
        self.master.title('2048 AI')
        self.grid_cells = []

        # Store the selected strategy, mode, and grid size
        self.strategy = strategy
        self.step_mode = step_mode
        self.grid_len = grid_len

        # Dynamically calculate size based on grid length
        self.size = DEFAULT_SIZE if grid_len <= 6 else DEFAULT_SIZE * (6 / grid_len)

        # Game tracking variables
        self.score = 0
        self.move_times = []
        self.move_count = 0
        self.max_tile = 0
        self.game_over = False
        
        # Pause and step controls
        self.is_paused = False
        self.waiting_for_step = False

        # Score display frame
        self.score_frame = tk.Frame(master, bg=BACKGROUND_COLOR_GAME)
        self.score_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.score_label = tk.Label(
            self.score_frame, 
            text=f"Score: {int(self.score)}", 
            font=SCORE_FONT, 
            bg=BACKGROUND_COLOR_GAME, 
            fg="white"
        )
        self.score_label.pack(side=tk.LEFT)

        self.strategy_label = tk.Label(
            self.score_frame, 
            text=f"Strategy: {strategy.name} ({grid_len}x{grid_len}) ({('Continuous' if not step_mode else 'Step-by-Step')} Mode)", 
            font=SCORE_FONT, 
            bg=BACKGROUND_COLOR_GAME, 
            fg="white"
        )
        self.strategy_label.pack(side=tk.RIGHT)

        # Control buttons frame
        self.control_frame = tk.Frame(master, bg=BACKGROUND_COLOR_GAME)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Pause/Resume button
        self.pause_resume_btn = tk.Button(
            self.control_frame, 
            text="Pause", 
            command=self.toggle_pause,
            font=SCORE_FONT
        )
        self.pause_resume_btn.pack(side=tk.LEFT, padx=5)

        # Step button (only visible in step mode)
        self.step_btn = tk.Button(
            self.control_frame, 
            text="Next Move", 
            command=self.step_move,
            font=SCORE_FONT,
            state=tk.NORMAL if step_mode else tk.DISABLED
        )
        self.step_btn.pack(side=tk.LEFT, padx=5)

        # Initialize grid and game state
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        
        # Pack the grid
        self.pack(expand=True, fill=tk.BOTH)

        # Initialize AI with selected strategy
        self.AI = AI(strategy=strategy)

        # Bind key press for game over
        self.master.bind('<Key>', self.handle_key_press)

        # Use after method for game loop to prevent freezing
        self.master.after(10, self.run_game)
        
        # Add a game start time
        self.game_start_time = perf_counter()
        self.move_times = []

    def init_grid(self):
        """Initialize the visual grid with empty cells."""
        background = tk.Frame(self.master, bg=BACKGROUND_COLOR_GAME, width=self.size, height=self.size)
        background.pack(expand=True, fill=tk.BOTH)

        for i in range(self.grid_len):
            grid_row = []

            for j in range(self.grid_len):
                cell = tk.Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=self.size/self.grid_len, height=self.size/self.grid_len)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING, sticky="nsew")
                
                # Configure grid weights to make cells expand
                background.grid_rowconfigure(i, weight=1)
                background.grid_columnconfigure(j, weight=1)
                
                # Adjust font size based on grid size
                font_size = 40 if self.grid_len <= 6 else max(10, int(40 * (6 / self.grid_len)))
                cell_font = ("Verdana", font_size, "bold")
                
                t = tk.Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=tk.CENTER, font=cell_font, width=4, height=2)
                t.pack(expand=True, fill=tk.BOTH)
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        """Initialize the game board with starting tiles."""
        # Create a new GameBoard with custom grid length
        self.board = GameBoard(self.grid_len)
        self.add_random_tile()
        self.add_random_tile()

    def update_grid_cells(self):
        """Update the visual grid to match the current game state."""
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                new_number = int(self.board.grid[i][j])
                cell = self.grid_cells[i][j]
                
                if new_number == 0:
                    # Minimal updates only when necessary
                    if cell['text'] != "" or cell['bg'] != BACKGROUND_COLOR_CELL_EMPTY:
                        cell.configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    # Get appropriate colors based on tile value
                    n = new_number
                    c = min(new_number, 2048)
                    
                    cell.configure(
                        text=str(n), 
                        bg=BACKGROUND_COLOR_DICT.get(c, BACKGROUND_COLOR_DICT[2048]), 
                        fg=CELL_COLOR_DICT.get(c, CELL_COLOR_DICT[2048])
                    )
        
        # Update the display
        self.master.update_idletasks()
    
    def show_game_stats(self):
        """Display game statistics in a new window."""
        # Calculate statistics
        ai_total_time = sum(self.move_times)
        ai_avg_time = ai_total_time / len(self.move_times) if self.move_times else 0
        
        # Calculate total game duration
        game_duration = perf_counter() - self.game_start_time
        
        # Create a new top-level window for stats
        stats_window = tk.Toplevel(self.master)
        stats_window.title("Game Statistics")
        stats_window.geometry("400x530")
        stats_window.configure(bg=BACKGROUND_COLOR_GAME)
    
        # Create labels for statistics
        stats = [
            f"Strategy: {self.strategy.name}",
            f"Board Size: {self.grid_len}x{self.grid_len}",
            f"Mode: {'Continuous' if not self.step_mode else 'Step-by-Step'}",
            f"Final Score: {int(self.score)}",
            f"Moves: {self.move_count}",
            f"Max Tile: {int(self.max_tile)}",
            f"Game Duration: {game_duration:.2f}s",
            f"AI Total Calculation Time: {ai_total_time:.4f}s",
            f"AI Avg Calculation Time: {ai_avg_time:.4f}s"
        ]
    
        for i, stat in enumerate(stats):
            label = tk.Label(
                stats_window, 
                text=stat, 
                font=("Verdana", 16), 
                bg=BACKGROUND_COLOR_GAME, 
                fg="white"
            )
            label.pack(pady=10)
    
        # Close button
        close_btn = tk.Button(
            stats_window, 
            text="Close", 
            command=stats_window.destroy
        )
        close_btn.pack(pady=20)
        
    def toggle_pause(self):
        """Toggle pause state of the game."""
        self.is_paused = not self.is_paused
            
        # Update button text
        if self.is_paused:
            self.pause_resume_btn.config(text="Resume")
        else:
            self.pause_resume_btn.config(text="Pause")
        
    def step_move(self):
        """Advance to the next move in step-by-step mode."""
        # Ensure the game is in step mode and paused
        if not self.step_mode:
            messagebox.showwarning("Step Mode", "Step mode is not enabled.")
            return
        
        if self.game_over:
            messagebox.showinfo("Game Over", "The game has ended.")
            return
        
        # Temporarily disable pause to allow the move
        was_paused = self.is_paused
        self.is_paused = False
        
        try:
            # Store original grid for score calculation
            original_grid = np.copy(self.board.grid)
            
            # Measure AI decision time
            start_time = perf_counter()
            
            # Get AI move
            move = self.AI.get_move(self.board)
            
            # Record move time
            move_time = perf_counter() - start_time
            self.move_times.append(move_time)
            
            # Apply move
            self.board.move(move)
            
            # Calculate and add score from move
            move_score = self.board.calculate_move_score(original_grid)
            self.score += move_score
            
            # Increment move counter
            self.move_count += 1
            
            # Add random tile
            self.add_random_tile()
    
            # Update max tile
            current_max_tile = self.board.get_max_tile()
            self.max_tile = max(self.max_tile, current_max_tile)
            
            # Update grid and score display
            self.update_grid_cells()
            self.score_label.config(text=f"Score: {int(self.score)}")
    
            # Check game over condition
            if len(self.board.get_available_moves()) == 0:
                self.game_over = True
                messagebox.showinfo("Game Over", f"Game ended! Final score: {int(self.score)}")
                self.show_game_stats()
        
        except Exception as e:
            messagebox.showerror("Move Error", str(e))
        
        # Restore paused state
        self.is_paused = was_paused
    
    def run_game(self):
        """Main game loop with score tracking."""
        # Check if game is over
        if self.game_over:
            return
        
        # Check pause states
        if self.is_paused and not (self.step_mode and self.waiting_for_step):
            self.master.after(50, self.run_game)
            return
        
        # Reset waiting for step if in step mode
        if self.step_mode and self.waiting_for_step:
            self.waiting_for_step = False
        
        try:
            # Store original grid for score calculation
            original_grid = np.copy(self.board.grid)
            
            # Measure AI decision time
            start_time = perf_counter()
            
            # Get AI move
            move = self.AI.get_move(self.board)
            
            # Record move time
            move_time = perf_counter() - start_time
            self.move_times.append(move_time)
            
            # Apply move
            self.board.move(move)
            
            # Calculate and add score from move
            move_score = self.board.calculate_move_score(original_grid)
            self.score += move_score
            
            # Increment move counter
            self.move_count += 1
            
            # Add random tile
            self.add_random_tile()
    
            # Update max tile
            current_max_tile = self.board.get_max_tile()
            self.max_tile = max(self.max_tile, current_max_tile)
            
            # Update grid and score display
            self.update_grid_cells()
            self.score_label.config(text=f"Score: {int(self.score)}")
    
            # Check game over condition
            if len(self.board.get_available_moves()) == 0:
                self.game_over = True
                self.master.after(200, self.show_game_over)
                return
    
        except Exception as e:
            print(f"Error in game loop: {e}")
            return
        
        # Scheduling next move based on game mode
        if not self.step_mode:
            self.master.after(0, self.run_game)
        else:
            # In step mode, wait until manually stepped
            pass
        
    def show_game_over(self):
        """Show game over dialog and statistics."""
        messagebox.showinfo("Game Over", f"Game ended! Final score: {int(self.score)}")
        self.show_game_stats()
    
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = self.board.get_available_cells()
            
        if not empty_cells:
            return False
            
        # 90% chance of 2, 10% chance of 4
        new_tile_value = 2 if randint(0, 10) < 9 else 4
        chosen_cell = empty_cells[randint(0, len(empty_cells) - 1)]
            
        # Insert tile and update board
        self.board.insert_tile(chosen_cell, new_tile_value)
        return True
        
    def handle_key_press(self, event):
        """Handle key press for game control."""
        if event.char == 'p' or event.char == 'P':
            # Toggle pause when 'p' is pressed
            self.toggle_pause()
        elif event.char == 's' or event.char == 'S':
            # Trigger step move when in step mode
            if self.step_mode:
                self.step_move()

def main():
    """Main function to start the application."""
    root = tk.Tk()
    MainMenu(root)
    root.mainloop()

if __name__ == "__main__":
    main()