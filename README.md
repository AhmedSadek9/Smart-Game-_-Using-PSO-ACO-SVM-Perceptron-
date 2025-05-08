# AI Game Agent: Treasure Collector

![Game Screenshot](screenshot.png) <!-- Replace with your actual screenshot -->

A Python game demonstrating various AI techniques for pathfinding and decision-making in a grid-based treasure collection game. The project showcases implementations of Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Support Vector Machines (SVM), Evolutionary Algorithms, and Neural Networks (Perceptron) as game agents.

---

## Features

- **Interactive grid-based game environment** with treasures, obstacles, and an AI agent
- **Multiple AI agent implementations**:
  - Particle Swarm Optimization (PSO)
  - Ant Colony Optimization (ACO)
  - Support Vector Machine (SVM)
  - Evolutionary Algorithm
  - Multi-layer Perceptron (Neural Network)
- **Real-time visualization** of agent movement and pathfinding
- **Comparative analysis** of different AI approaches
- **Interactive controls** for manual play or AI auto-play

---

## Requirements

- Python 3.7+
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)

Install all dependencies using:
```bash
pip install pygame numpy scikit-learn
```

---

## How to Run

```bash
python ai_game_agent.py
```

---

## Controls

- **Arrow keys**: Manual movement (when auto-play is off)
- **1-5**: Switch between different AI agents  
  - 1: PSO Agent  
  - 2: ACO Agent  
  - 3: SVM Agent  
  - 4: Evolutionary Agent  
  - 5: Perceptron Agent  
- **SPACE**: Toggle auto-play mode
- **R**: Reset the game

---

## AI Techniques Implemented

### 1. Particle Swarm Optimization (PSO)
- Simulates swarm intelligence to find optimal paths
- Particles represent potential move sequences
- Updates based on personal and global best positions

### 2. Ant Colony Optimization (ACO)
- Inspired by ant foraging behavior
- Uses pheromone trails to mark good paths
- Balances exploration and exploitation

### 3. Support Vector Machine (SVM)
- Supervised learning model trained on optimal moves
- Uses game state features for decision making
- Pre-trained on simulated games

### 4. Evolutionary Algorithm
- Genetic algorithm approach
- Population of move sequences evolves over generations
- Utilizes selection, crossover, and mutation operators

### 5. Perceptron (Neural Network)
- Simple multi-layer perceptron classifier
- Trained on optimal moves from simulations
- Learns to map game states to actions

---

## Game Mechanics

- Agent starts in the center of the grid
- 10 treasures (yellow) and 20 obstacles (red) randomly placed
- **Goal**: Collect all treasures in minimum steps (max 200 steps)
- **Score**: +10 per treasure, -0.1 per step
- Game ends when all treasures are collected or step limit is reached

---

## File Structure

```
ai_game_agent.py      # Main game loop and AI agent implementations
README.md             # Project documentation
screenshot.png        # Game screenshot (replace with actual image)
```

---

## Future Enhancements

- More sophisticated state representations
- Deep Reinforcement Learning (e.g., DQN)
- Enhanced visualizations of AI decision-making
- Performance metrics and comparison charts
- Configurable AI hyperparameters via UI or config file

---

## License

This project is open-source under the [MIT License](LICENSE).

---

## Acknowledgements

- [Pygame](https://www.pygame.org/) for game development
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- Tutorials and literature on AI optimization techniques

---

## References (Optional)

- Add relevant research papers, blogs, or GitHub projects that inspired your implementations.