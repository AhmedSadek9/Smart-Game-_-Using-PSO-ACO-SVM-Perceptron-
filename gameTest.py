import pygame
import numpy as np
import random
import math
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Initialize pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Game Agent")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

class Game:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.agent_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.treasures = []
        self.obstacles = []
        self.score = 0
        self.steps = 0
        self.generate_environment()
        self.agent_path = [tuple(self.agent_pos)]
        self.game_over = False
        
    def generate_environment(self):
        # Generate 10 treasures at random positions
        self.treasures = []
        for _ in range(10):
            pos = [random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)]
            while pos == self.agent_pos or pos in self.treasures:
                pos = [random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)]
            self.treasures.append(pos)
        
        # Generate 20 obstacles at random positions
        self.obstacles = []
        for _ in range(20):
            pos = [random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)]
            while pos == self.agent_pos or pos in self.treasures or pos in self.obstacles:
                pos = [random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)]
            self.obstacles.append(pos)
    
    def move_agent(self, direction):
        if self.game_over:
            return
            
        self.steps += 1
        new_pos = self.agent_pos.copy()
        
        # 0: up, 1: right, 2: down, 3: left
        if direction == 0 and new_pos[1] > 0:
            new_pos[1] -= 1
        elif direction == 1 and new_pos[0] < GRID_WIDTH - 1:
            new_pos[0] += 1
        elif direction == 2 and new_pos[1] < GRID_HEIGHT - 1:
            new_pos[1] += 1
        elif direction == 3 and new_pos[0] > 0:
            new_pos[0] -= 1
            
        # Check if new position is valid
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
            self.agent_path.append(tuple(new_pos))
            
            # Check if treasure collected
            if new_pos in self.treasures:
                self.treasures.remove(new_pos)
                self.score += 10
                
            # Check game over condition
            if len(self.treasures) == 0 or self.steps >= 200:
                self.game_over = True
    
    def get_state(self):
        # Create a simple state representation
        state = []
        
        # Agent position
        state.extend(self.agent_pos)
        
        # Distance to nearest treasure
        if self.treasures:
            min_dist = min(math.dist(self.agent_pos, treasure) for treasure in self.treasures)
            state.append(min_dist)
        else:
            state.append(0)
            
        # Number of treasures in each direction
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            count = 0
            for treasure in self.treasures:
                if (dx > 0 and treasure[0] > self.agent_pos[0]) or \
                   (dx < 0 and treasure[0] < self.agent_pos[0]) or \
                   (dy > 0 and treasure[1] > self.agent_pos[1]) or \
                   (dy < 0 and treasure[1] < self.agent_pos[1]):
                    count += 1
            state.append(count)
            
        return np.array(state)
    
    def draw(self):
        screen.fill(BLACK)
        
        # Draw grid
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            rect = pygame.Rect(obstacle[0] * GRID_SIZE, obstacle[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, RED, rect)
        
        # Draw treasures
        for treasure in self.treasures:
            rect = pygame.Rect(treasure[0] * GRID_SIZE, treasure[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, YELLOW, rect)
        
        # Draw agent
        agent_rect = pygame.Rect(self.agent_pos[0] * GRID_SIZE, self.agent_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, GREEN, agent_rect)
        
        # Draw path
        if len(self.agent_path) > 1:
            points = [(x * GRID_SIZE + GRID_SIZE//2, y * GRID_SIZE + GRID_SIZE//2) for x, y in self.agent_path]
            pygame.draw.lines(screen, BLUE, False, points, 2)
        
        # Draw score and steps
        score_text = font.render(f"Score: {self.score} | Steps: {self.steps}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        if self.game_over:
            game_over_text = font.render("Game Over! Press R to reset", True, WHITE)
            screen.blit(game_over_text, (WIDTH//2 - 100, HEIGHT//2))

# Particle Swarm Optimization (PSO) for path planning
class PSOAgent:
    def __init__(self, game):
        self.game = game
        self.num_particles = 20
        self.max_iter = 50
        self.particles = []
        self.global_best = None
        self.global_best_score = -float('inf')
        
    def initialize_particles(self):
        self.particles = []
        for _ in range(self.num_particles):
            particle = {
                'position': [random.randint(0, 3) for _ in range(5)],  # Sequence of 5 moves
                'velocity': [random.uniform(-1, 1) for _ in range(5)],
                'best_position': None,
                'best_score': -float('inf')
            }
            self.particles.append(particle)
    
    def evaluate_particle(self, particle):
        # Simulate the moves and calculate the score
        game_copy = Game()
        game_copy.agent_pos = self.game.agent_pos.copy()
        game_copy.treasures = [t.copy() for t in self.game.treasures]
        game_copy.obstacles = [o.copy() for o in self.game.obstacles]
        
        for move in particle['position']:
            game_copy.move_agent(int(round(move)) % 4)
            if game_copy.game_over:
                break
        
        # Calculate score based on treasures collected and path length
        score = game_copy.score - 0.1 * len(game_copy.agent_path)
        return score
    
    def update_particles(self):
        w = 0.7  # inertia weight
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient
        
        for particle in self.particles:
            # Update velocity
            for i in range(len(particle['velocity'])):
                r1 = random.random()
                r2 = random.random()
                
                cognitive = c1 * r1 * (particle['best_position'][i] - particle['position'][i])
                social = c2 * r2 * (self.global_best[i] - particle['position'][i])
                particle['velocity'][i] = w * particle['velocity'][i] + cognitive + social
            
            # Update position
            for i in range(len(particle['position'])):
                particle['position'][i] += particle['velocity'][i]
                # Keep within move bounds (0-3)
                particle['position'][i] = max(0, min(3, particle['position'][i]))
            
            # Evaluate new position
            score = self.evaluate_particle(particle)
            
            # Update personal best
            if score > particle['best_score']:
                particle['best_score'] = score
                particle['best_position'] = particle['position'].copy()
            
            # Update global best
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best = particle['position'].copy()
    
    def get_next_move(self):
        self.initialize_particles()
        
        # Initialize personal and global bests
        for particle in self.particles:
            score = self.evaluate_particle(particle)
            particle['best_score'] = score
            particle['best_position'] = particle['position'].copy()
            
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best = particle['position'].copy()
        
        # Run PSO iterations
        for _ in range(self.max_iter):
            self.update_particles()
        
        # Return the first move of the best sequence
        return int(round(self.global_best[0])) % 4

# Ant Colony Optimization (ACO) for path planning
class ACOAgent:
    def __init__(self, game):
        self.game = game
        self.num_ants = 10
        self.max_iter = 30
        self.pheromones = np.ones((GRID_WIDTH, GRID_HEIGHT, 4))  # Pheromone for each grid cell and direction
        self.evaporation = 0.9
        self.alpha = 1.0
        self.beta = 2.0
        
    def get_possible_moves(self, pos):
        moves = []
        for direction in range(4):
            new_pos = pos.copy()
            if direction == 0 and new_pos[1] > 0:
                new_pos[1] -= 1
            elif direction == 1 and new_pos[0] < GRID_WIDTH - 1:
                new_pos[0] += 1
            elif direction == 2 and new_pos[1] < GRID_HEIGHT - 1:
                new_pos[1] += 1
            elif direction == 3 and new_pos[0] > 0:
                new_pos[0] -= 1
                
            if new_pos not in self.game.obstacles:
                moves.append(direction)
        return moves
    
    def evaluate_path(self, path):
        game_copy = Game()
        game_copy.agent_pos = self.game.agent_pos.copy()
        game_copy.treasures = [t.copy() for t in self.game.treasures]
        game_copy.obstacles = [o.copy() for o in self.game.obstacles]
        
        for move in path:
            game_copy.move_agent(move)
            if game_copy.game_over:
                break
        
        # Calculate score based on treasures collected and path length
        score = game_copy.score - 0.1 * len(game_copy.agent_path)
        return score
    
    def update_pheromones(self, ants):
        # Evaporate pheromones
        self.pheromones *= self.evaporation
        
        # Deposit new pheromones
        for ant in ants:
            path = ant['path']
            score = ant['score']
            
            # Deposit pheromones along the path
            current_pos = self.game.agent_pos.copy()
            for move in path:
                x, y = current_pos
                self.pheromones[x, y, move] += score
                
                # Update position
                if move == 0:
                    current_pos[1] -= 1
                elif move == 1:
                    current_pos[0] += 1
                elif move == 2:
                    current_pos[1] += 1
                elif move == 3:
                    current_pos[0] -= 1
    
    def get_next_move(self):
        best_move = None
        best_score = -float('inf')
        
        for _ in range(self.max_iter):
            ants = []
            
            for _ in range(self.num_ants):
                ant = {'path': [], 'score': 0}
                current_pos = self.game.agent_pos.copy()
                path = []
                
                for _ in range(5):  # Plan 5 moves ahead
                    possible_moves = self.get_possible_moves(current_pos)
                    if not possible_moves:
                        break
                    
                    # Calculate probabilities for each move
                    x, y = current_pos
                    probabilities = []
                    total = 0
                    
                    for move in possible_moves:
                        pheromone = self.pheromones[x, y, move]
                        # Simple heuristic: prefer moves toward nearest treasure
                        heuristic = 1.0
                        
                        if self.game.treasures:
                            # Calculate direction to nearest treasure
                            nearest = min(self.game.treasures, key=lambda t: math.dist(current_pos, t))
                            desired_dx = nearest[0] - current_pos[0]
                            desired_dy = nearest[1] - current_pos[1]
                            
                            if (move == 0 and desired_dy < 0) or (move == 1 and desired_dx > 0) or \
                               (move == 2 and desired_dy > 0) or (move == 3 and desired_dx < 0):
                                heuristic = 2.0
                        
                        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                        probabilities.append(prob)
                        total += prob
                    
                    if total == 0:
                        move = random.choice(possible_moves)
                    else:
                        probabilities = [p/total for p in probabilities]
                        move = random.choices(possible_moves, weights=probabilities)[0]
                    
                    path.append(move)
                    
                    # Update position
                    if move == 0:
                        current_pos[1] -= 1
                    elif move == 1:
                        current_pos[0] += 1
                    elif move == 2:
                        current_pos[1] += 1
                    elif move == 3:
                        current_pos[0] -= 1
                
                ant['path'] = path
                ant['score'] = self.evaluate_path(path)
                ants.append(ant)
                
                # Track the best move
                if ant['score'] > best_score:
                    best_score = ant['score']
                    best_move = path[0] if path else random.randint(0, 3)
            
            self.update_pheromones(ants)
        
        return best_move if best_move is not None else random.randint(0, 3)

# Support Vector Machine (SVM) agent
class SVMAgent:
    def __init__(self, game):
        self.game = game
        self.model = svm.SVC()
        self.scaler = StandardScaler()
        self.trained = False
        self.collect_training_data()
        
    def collect_training_data(self):
        # Generate training data by playing random games
        X = []
        y = []
        
        for _ in range(100):  # 100 training games
            game = Game()
            game.generate_environment()
            
            while not game.game_over:
                state = game.get_state()
                
                # Find the best move (toward nearest treasure)
                if game.treasures:
                    nearest = min(game.treasures, key=lambda t: math.dist(game.agent_pos, t))
                    dx = nearest[0] - game.agent_pos[0]
                    dy = nearest[1] - game.agent_pos[1]
                    
                    if abs(dx) > abs(dy):
                        best_move = 1 if dx > 0 else 3
                    else:
                        best_move = 2 if dy > 0 else 0
                else:
                    best_move = random.randint(0, 3)
                
                # Check if best move is valid
                new_pos = game.agent_pos.copy()
                if best_move == 0:
                    new_pos[1] -= 1
                elif best_move == 1:
                    new_pos[0] += 1
                elif best_move == 2:
                    new_pos[1] += 1
                elif best_move == 3:
                    new_pos[0] -= 1
                
                if new_pos in game.obstacles:
                    # Choose a random valid move if best move is invalid
                    possible_moves = []
                    for move in range(4):
                        test_pos = game.agent_pos.copy()
                        if move == 0 and test_pos[1] > 0:
                            test_pos[1] -= 1
                        elif move == 1 and test_pos[0] < GRID_WIDTH - 1:
                            test_pos[0] += 1
                        elif move == 2 and test_pos[1] < GRID_HEIGHT - 1:
                            test_pos[1] += 1
                        elif move == 3 and test_pos[0] > 0:
                            test_pos[0] -= 1
                        
                        if test_pos not in game.obstacles:
                            possible_moves.append(move)
                    
                    if possible_moves:
                        best_move = random.choice(possible_moves)
                    else:
                        best_move = random.randint(0, 3)
                
                X.append(state)
                y.append(best_move)
                
                # Make the move
                game.move_agent(best_move)
        
        # Train the SVM
        X = np.array(X)
        y = np.array(y)
        
        # Scale the features
        X = self.scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"SVM Training complete. Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
    
    def get_next_move(self):
        if not self.trained:
            return random.randint(0, 3)
            
        state = self.game.get_state().reshape(1, -1)
        state = self.scaler.transform(state)
        return self.model.predict(state)[0]

# Evolutionary Algorithm agent
class EvolutionaryAgent:
    def __init__(self, game):
        self.game = game
        self.population_size = 50
        self.generations = 20
        self.mutation_rate = 0.1
        self.population = []
        
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            # Each individual is a sequence of 5 moves
            individual = [random.randint(0, 3) for _ in range(5)]
            self.population.append(individual)
    
    def evaluate_individual(self, individual):
        game_copy = Game()
        game_copy.agent_pos = self.game.agent_pos.copy()
        game_copy.treasures = [t.copy() for t in self.game.treasures]
        game_copy.obstacles = [o.copy() for o in self.game.obstacles]
        
        for move in individual:
            game_copy.move_agent(move)
            if game_copy.game_over:
                break
        
        # Fitness based on score and path length
        fitness = game_copy.score - 0.1 * len(game_copy.agent_path)
        return fitness
    
    def selection(self, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(self.population_size):
            # Pick 2 random individuals
            a, b = random.sample(range(self.population_size), 2)
            # Select the one with higher fitness
            winner = a if fitnesses[a] > fitnesses[b] else b
            selected.append(self.population[winner].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        # Single-point crossover
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, 3)
        return individual
    
    def evolve(self):
        # Evaluate current population
        fitnesses = [self.evaluate_individual(ind) for ind in self.population]
        
        # Selection
        selected = self.selection(fitnesses)
        
        # Crossover
        new_population = []
        for i in range(0, len(selected), 2):
            if i+1 < len(selected):
                child1, child2 = self.crossover(selected[i], selected[i+1])
                new_population.extend([child1, child2])
            else:
                new_population.append(selected[i])
        
        # Mutation
        for i in range(len(new_population)):
            new_population[i] = self.mutate(new_population[i])
        
        self.population = new_population
    
    def get_next_move(self):
        self.initialize_population()
        
        # Run evolution
        for _ in range(self.generations):
            self.evolve()
        
        # Find the best individual
        best_individual = max(self.population, key=lambda ind: self.evaluate_individual(ind))
        return best_individual[0]

# Perceptron (Simple Neural Network) agent
class PerceptronAgent:
    def __init__(self, game):
        self.game = game
        self.model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.collect_training_data()
        
    def collect_training_data(self):
        # Similar to SVM, generate training data
        X = []
        y = []
        
        for _ in range(100):  # 100 training games
            game = Game()
            game.generate_environment()
            
            while not game.game_over:
                state = game.get_state()
                
                # Find the best move (toward nearest treasure)
                if game.treasures:
                    nearest = min(game.treasures, key=lambda t: math.dist(game.agent_pos, t))
                    dx = nearest[0] - game.agent_pos[0]
                    dy = nearest[1] - game.agent_pos[1]
                    
                    if abs(dx) > abs(dy):
                        best_move = 1 if dx > 0 else 3
                    else:
                        best_move = 2 if dy > 0 else 0
                else:
                    best_move = random.randint(0, 3)
                
                # Check if best move is valid
                new_pos = game.agent_pos.copy()
                if best_move == 0:
                    new_pos[1] -= 1
                elif best_move == 1:
                    new_pos[0] += 1
                elif best_move == 2:
                    new_pos[1] += 1
                elif best_move == 3:
                    new_pos[0] -= 1
                
                if new_pos in game.obstacles:
                    # Choose a random valid move if best move is invalid
                    possible_moves = []
                    for move in range(4):
                        test_pos = game.agent_pos.copy()
                        if move == 0 and test_pos[1] > 0:
                            test_pos[1] -= 1
                        elif move == 1 and test_pos[0] < GRID_WIDTH - 1:
                            test_pos[0] += 1
                        elif move == 2 and test_pos[1] < GRID_HEIGHT - 1:
                            test_pos[1] += 1
                        elif move == 3 and test_pos[0] > 0:
                            test_pos[0] -= 1
                        
                        if test_pos not in game.obstacles:
                            possible_moves.append(move)
                    
                    if possible_moves:
                        best_move = random.choice(possible_moves)
                    else:
                        best_move = random.randint(0, 3)
                
                X.append(state)
                y.append(best_move)
                
                # Make the move
                game.move_agent(best_move)
        
        # Train the Perceptron
        X = np.array(X)
        y = np.array(y)
        
        # Scale the features
        X = self.scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Perceptron Training complete. Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
    
    def get_next_move(self):
        if not self.trained:
            return random.randint(0, 3)
            
        state = self.game.get_state().reshape(1, -1)
        state = self.scaler.transform(state)
        return self.model.predict(state)[0]

# Main game loop
def main():
    game = Game()
    agents = {
        "PSO": PSOAgent(game),
        "ACO": ACOAgent(game),
        "SVM": SVMAgent(game),
        "Evolutionary": EvolutionaryAgent(game),
        "Perceptron": PerceptronAgent(game)
    }
    current_agent = "PSO"
    auto_play = False
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_1:
                    current_agent = "PSO"
                    print("Switched to PSO Agent")
                elif event.key == pygame.K_2:
                    current_agent = "ACO"
                    print("Switched to ACO Agent")
                elif event.key == pygame.K_3:
                    current_agent = "SVM"
                    print("Switched to SVM Agent")
                elif event.key == pygame.K_4:
                    current_agent = "Evolutionary"
                    print("Switched to Evolutionary Agent")
                elif event.key == pygame.K_5:
                    current_agent = "Perceptron"
                    print("Switched to Perceptron Agent")
                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                    print(f"Auto-play {'enabled' if auto_play else 'disabled'}")
                elif not auto_play:
                    if event.key == pygame.K_UP:
                        game.move_agent(0)
                    elif event.key == pygame.K_RIGHT:
                        game.move_agent(1)
                    elif event.key == pygame.K_DOWN:
                        game.move_agent(2)
                    elif event.key == pygame.K_LEFT:
                        game.move_agent(3)
        
        if auto_play and not game.game_over:
            # Get move from current AI agent
            move = agents[current_agent].get_next_move()
            game.move_agent(move)
            
            # Small delay for visualization
            pygame.time.delay(100)
        
        # Draw everything
        game.draw()
        
        # Display current agent
        agent_text = font.render(f"Agent: {current_agent} (1-5 to change, SPACE for auto)", True, WHITE)
        screen.blit(agent_text, (10, 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()