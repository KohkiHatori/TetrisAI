# Deep Q-Learning Tetris AI Agent

A sophisticated reinforcement learning agent that masters Tetris through Deep Q-Learning, featuring advanced feature engineering and reward shaping techniques.

## 🎯 Project Overview

This project implements an intelligent Tetris-playing agent using Deep Q-Learning (DQN). The agent learns to play Tetris by developing strategies for piece placement, line clearing, and long-term board management through reinforcement learning.

**Game Mechanics:** This Tetris implementation focuses on strategic placement rather than manual control. The agent is provided with all valid final positions for each piece and selects the optimal placement, eliminating the need to learn movement controls and allowing focus on high-level strategic decision making.

**My Contribution:** Complete implementation of the Q-Learning agent (`TetrisQAgent.java`), including neural network architecture, feature engineering, exploration strategies, and reward function design.

## 🧠 Key Technical Implementations

### 1. Advanced Feature Engineering (55-Dimensional State Space)
Designed a comprehensive feature vector that captures critical game state information:

- **Board Analysis Features:**
  - Block density and distribution patterns
  - Column height variations and average height
  - Surface bumpiness and flatness metrics
  - Highest occupied row detection

- **Strategic Features:**
  - Unreachable block detection using flood-fill algorithm
  - Hole analysis and "roofed" block identification
  - T-spin detection for advanced scoring opportunities
  - Line clearing potential assessment

- **Piece Information:**
  - Current piece type and orientation encoding
  - Next three pieces preview integration
  - Piece position coordinates (normalized)

### 2. Neural Network Architecture
```java
// Custom 3-layer feedforward network
Sequential qFunction = new Sequential();
qFunction.add(new Dense(55, 64));      // Input layer: 55 features → 64 neurons
qFunction.add(new ReLU());             // Activation function
qFunction.add(new Dense(64, 64));      // Hidden layer: 64 → 64 neurons  
qFunction.add(new ReLU());             // Activation function
qFunction.add(new Dense(64, 1));       // Output layer: 64 → 1 Q-value
```

### 3. Intelligent Exploration Strategy
Implemented adaptive exploration that balances exploitation vs exploration:
```java
// Dynamic exploration probability that decreases over time
double explorationProb = Math.pow(0.99, (currentCycleIdx/3 + currentGameIdx/3));
```
- **Early Training:** High exploration for diverse experience gathering
- **Late Training:** Focused exploitation of learned strategies
- **Guided Exploration:** Non-random action selection during exploration phases

### 4. Sophisticated Reward Function
Engineered a multi-component reward system addressing sparse reward challenges:

```java
double reward = score*2                           // Direct game score
              + holesDiff                         // Hole creation/filling
              + flatness                          // Surface smoothness
              + avgHeightRatioReverse            // Height management
              + ratioHighest                      // Avoiding game over
              + unreachableBlockRatioReverse     // Accessibility optimization
              + unreachableRowsRatioReverse      // Row reachability
              - (game.didAgentLose() ? 20.0 : 0.0) // Game over penalty
              - roofedRatio;                      // Trapped space penalty
```

### 5. Optimized Algorithms
- **Iterative Flood Fill:** Efficient reachability analysis using custom stack implementation
- **T-Spin Detection:** Advanced pattern recognition for bonus scoring opportunities
- **Board State Analysis:** Real-time evaluation of game state quality

## 🔧 Technical Skills Demonstrated

- **Machine Learning:** Deep Q-Learning, Neural Networks, Reinforcement Learning
- **Algorithm Design:** Flood-fill algorithms, feature extraction, state space design
- **Java Programming:** Object-oriented design, performance optimization
- **Problem Solving:** Sparse reward handling, exploration-exploitation balance
- **Game AI:** Strategic planning, pattern recognition, decision making

## 📊 Performance Results

**Final Model Performance:** Average score of **63.876** across 500 evaluation games

### Performance Features

- **Adaptive Learning:** Dynamic exploration rate adjustment based on training progress
- **Memory Efficiency:** Reusable data structures for flood-fill operations
- **Strategic Play:** Recognition of advanced Tetris techniques (T-spins, line clearing optimization)
- **Robust Decision Making:** Multi-factor evaluation considering both immediate and long-term consequences
- **Consistent Performance:** Stable scoring across large-scale evaluation (500 games)

## 🎮 Agent Capabilities

The trained agent demonstrates:
- **Strategic Piece Placement:** Optimal positioning considering future pieces
- **Hole Management:** Minimizing and strategically filling gaps
- **Height Control:** Maintaining manageable stack heights
- **Line Clearing Optimization:** Maximizing scoring opportunities
- **Advanced Techniques:** T-spin recognition and execution

## 🚀 Technical Highlights

- **State Space Design:** 55-dimensional feature vector capturing game complexity
- **Reward Shaping:** Multi-objective function balancing various game aspects  
- **Exploration Strategy:** Intelligent action selection during learning phases
- **Performance Optimization:** Efficient algorithms for real-time decision making
- **Neural Architecture:** Custom deep network tailored for Q-value estimation

## 🚀 Getting Started

### Prerequisites
- Java 8 or higher (developed and tested with Java 1.8.0_382)
- Terminal/Command line access

### Running the Tetris AI Agent

1. **Compile the project:**
   ```bash
   java -cp "./lib/*:." edu.bu.pas.tetris.Main
   ```

2. **Run the trained model:**
   ```bash
   java -cp "./lib/*:." edu.bu.pas.tetris.Main -p 1 -t 0 -v 1 -a src.pas.tetris.agents.TetrisQAgent -i params.model
   ```

### Command Line Arguments
- `-p 1`: Run 1 game
- `-t 0`: No training games (evaluation only)
- `-v 1`: 1 evaluation game
- `-a src.pas.tetris.agents.TetrisQAgent`: Use the custom Q-Learning agent
- `-i params.model`: Load pre-trained model parameters

This will demonstrate the trained agent playing a single game of Tetris, showcasing the strategic decision-making capabilities developed through reinforcement learning.

---

*This project showcases advanced reinforcement learning techniques, sophisticated feature engineering, and strategic AI development for complex game environments.*
