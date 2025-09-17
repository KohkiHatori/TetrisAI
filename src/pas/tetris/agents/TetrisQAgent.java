package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.*;


// JAVA PROJECT IMPORTS
import edu.bu.pas.tetris.agents.QAgent;
import edu.bu.pas.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.pas.tetris.game.Block;
import edu.bu.pas.tetris.game.Board;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.Model;
import edu.bu.pas.tetris.nn.LossFunction;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.models.Sequential;
import edu.bu.pas.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.pas.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.pas.tetris.training.data.Dataset;
import edu.bu.pas.tetris.utils.Coordinate;
import edu.bu.pas.tetris.utils.Pair;


public class TetrisQAgent
        extends QAgent
{

    public static final int INPUT_DIM = 55;
    public static final int HIDDEN_DIM = 64;
    public int numUnreachablePrevious = 0;
    private static final int ROWS = Board.NUM_ROWS;   // 22
    private static final int COLS = Board.NUM_COLS;   // 10
    private static final int STACK_CAP = ROWS * COLS*5;
    private final int[] stackX = new int[STACK_CAP];        // reused flood-fill stack
    private final int[] stackY = new int[STACK_CAP];

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(INPUT_DIM, HIDDEN_DIM));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(HIDDEN_DIM, HIDDEN_DIM));
        qFunction.add(new ReLU());
//        qFunction.add(new Dense(HIDDEN_DIM, HIDDEN_DIM));
//        qFunction.add(new ReLU());
        qFunction.add(new Dense(HIDDEN_DIM, 1));
        return qFunction;
    }

    public boolean isOver(boolean[][] occupied) {
        for (int x = 0; x < COLS; x++)
            if (occupied[0][x] || occupied[1][x]) return true;
        return false;
    }

    private boolean wasTSpin(Board board, Mino potentialAction) {
        if (potentialAction.getType() != Mino.MinoType.T) return false;
        Coordinate p = potentialAction.getPivotBlockCoordinate();
        int[][] c = {{-1,-1},{1,-1},{-1,1},{1,1}};
        int occ = 0;
        for (int[] d : c) {
            int x = p.getXCoordinate() + d[0];
            int y = p.getYCoordinate() + d[1];
            if (x < 0 || x >= Board.NUM_COLS || y < 0 || y >= Board.NUM_ROWS || board.isCoordinateOccupied(x, y)) occ++;
        }
        return occ >= 3;
    }

    private boolean wasDoubleTSpin(Board board, Mino potentialAction) {
        if (!wasTSpin(board, potentialAction)) return false;
        int full = 0;
        for (int y = 0; y < Board.NUM_ROWS; y++) {
            boolean line = true;
            for (int x = 0; x < Board.NUM_COLS && line; x++) {
                if (!board.isCoordinateOccupied(x, y)) line = false;
            }
            if (line) full++;
        }
        return full == 2;
    }



    /**
     This function is for you to figure out what your features
     are. This should end up being a single row-vector, and the
     dimensions should be what your qfunction is expecting.
     One thing we can do is get the grayscale image
     where squares in the image are 0.0 if unoccupied, 0.5 if
     there is a "background" square (i.e. that square is occupied
     but it is not the current piece being placed), and 1.0 for
     any squares that the current piece is being considered for.

     We can then flatten this image to get a row-vector, but we
     can do more than this! Try to be creative: how can you measure the
     "state" of the game without relying on the pixels? If you were given
     a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        boolean[][] occupied = new boolean[ROWS][COLS];
        Board board = new Board(game.getBoard());
        board.addMino(potentialAction);

        for (int y=0; y<ROWS; y++)
            for (int x=0; x<COLS; x++)
                occupied[y][x] = board.isCoordinateOccupied(x,y);

        int[] columnHeight = new int[COLS];
        int   highest   = ROWS, numBlocks = 0;
        boolean[][] reachable = new boolean[ROWS][COLS];
        for (int x=0; x<COLS; x++)
            if (!occupied[0][x])
                floodFill(occupied, reachable,0,x);   // reuse iterative fill
        double unreachableBlocks = 0;
        double unreachableRows = 0;
        double roofed = 0;
        for (int y=0; y<ROWS; y++) {
            int counter = 0;
            boolean rowRechable = true;
            for (int x=0; x<COLS; x++) {
                if (occupied[y][x]) {
                    numBlocks++;
                    counter++;
                    if (columnHeight[x] == 0)
                        columnHeight[x]= ROWS - y;
                }
                if (!occupied[y][x]){
                    if (!reachable[y][x]){
                        unreachableBlocks++;
                        rowRechable = false;
                    }
                    if (columnHeight[x] != 0){
                        roofed++;
                    }
                }
            }
            if (counter>0 && y<highest)
                highest=y;
            if (!rowRechable)
                unreachableRows++;
        }

        double avgHeight = 0;
        int bumpiness = 0;
        for (int x=0; x<COLS; x++) {
            avgHeight += columnHeight[x];
            if (x <COLS-1){
                bumpiness += Math.abs(columnHeight[x] - columnHeight[x + 1]);
            }
        }
        avgHeight /= COLS;
        double flatness = 20 - bumpiness;

        Matrix input = Matrix.zeros(1, INPUT_DIM);
        input.set(0, 0, numBlocks/(double)(ROWS*COLS));
        input.set(0, 2, avgHeight/(double)ROWS);
        input.set(0, 3, 1.0 - highest/(double)ROWS);
        input.set(0, 4, 1 - unreachableBlocks/(double) numBlocks);
        input.set(0, 5, 1 - unreachableRows / (double) (ROWS-highest));
        input.set(0, 6, flatness/20.0);
        input.set(0, 7, isOver(occupied) ? 1.0 : 0.0);
        input.set(0, 8, wasTSpin(board, potentialAction) ? 1.0 : 0.0);
        input.set(0, 9, wasDoubleTSpin(board, potentialAction) ? 1.0 : 0.0);

        input.set(0, 1, board.clearFullLines().size());

        input.set(0, 10, roofed/(double) numBlocks);
        input.set(0, 11 + potentialAction.getType().ordinal(), 1.0);
        List<Mino.MinoType> q = game.getNextThreeMinoTypes();
        for (int k=0; k < q.size(); ++k)
            input.set(0, 18 + k*7 + q.get(k).ordinal(), 1.0);
        for (int x=0; x<COLS; x++){
            input.set(0, 39+x, columnHeight[x]/(double)ROWS);
        }
        input.set(0, 49 + potentialAction.getOrientation().ordinal(), 1.0);
        Coordinate c = potentialAction.getPivotBlockCoordinate();
        input.set(0, 53, (double) c.getXCoordinate() /COLS);
        input.set(0, 54, (double) c.getYCoordinate() /ROWS);
        return input;
    }

    private void floodFill(boolean[][] occ, boolean[][] mark, int y, int x){
        int top=0;
        stackX[top]=x; stackY[top++]=y;
        while(top>0){
            y=stackY[--top]; x=stackX[top];
            if(y<0||y>=ROWS||x<0||x>=COLS||mark[y][x]||occ[y][x]) continue;
            mark[y][x]=true;
            stackX[top]=x-1; stackY[top++]=y;
            stackX[top]=x+1; stackY[top++]=y;
            stackX[top]=x;   stackY[top++]=y-1;
            stackX[top]=x;   stackY[top++]=y+1;
        }
    }


    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        long currentCycleIdx = gameCounter.getCurrentCycleIdx();
        long currentGameIdx = gameCounter.getCurrentGameIdx();
        double explorationProb = Math.pow(0.99, (double) (currentCycleIdx/3 + currentGameIdx/3));
//        System.out.println("Exploration Probability: " + explorationProb);
        return this.getRandom().nextDouble() <= explorationProb;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        Mino best = this.getBestActionAndQValue(game).getFirst();
        List<Mino> allActions = new ArrayList<>(game.getFinalMinoPositions());
        allActions.remove(best);

        if (allActions.isEmpty()) {
            return best;
        }

        int randIdx = this.getRandom().nextInt(allActions.size());
        Mino move = allActions.get(randIdx);
        return move;
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a cycle, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each cycle.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                            lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }


    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game) {
        boolean[][] occupied = new boolean[ROWS][COLS];
        Board board = new Board(game.getBoard());
        for (int y=0; y<ROWS; y++) {
            for (int x = 0; x < COLS; x++) {
                occupied[y][x] = board.isCoordinateOccupied(x, y);
            }
        }

        // The game is over after adding the mino to the board
        // i.e. getNextMino did created zero final positions

        int[] columnHeight = new int[COLS];
        int   highest   = ROWS, numBlocks = 0;
        boolean[][] reachable = new boolean[ROWS][COLS];
        for (int x=0; x<COLS; x++)
            if (!occupied[0][x])
                floodFill(occupied, reachable,0,x);   // reuse iterative fill
        double unreachableBlocks = 0;
        double unreachableRows = 0;
        double roofed = 0;

        for (int y=0; y<ROWS; y++) {
            int counter = 0;
            boolean rowReachable = true;
            for (int x=0; x<COLS; x++) {
                if (occupied[y][x]) {
                    numBlocks++;
                    counter++;
                    if (columnHeight[x] == 0)
                        columnHeight[x]= ROWS - y;
                }
                if (!occupied[y][x]){
                    if (!reachable[y][x]){
                        unreachableBlocks++;
                        rowReachable = false; // row with holes
                    }
                    if (columnHeight[x] != 0){
                        roofed++;
                    }
                }
            }
            if (counter>0 && y<highest)
                highest=y;
            if (!rowReachable)
                unreachableRows++;
        }

        double avgHeight = 0;
        int bumpiness = 0;
        for (int x = 0; x < COLS; x++) {
            avgHeight += columnHeight[x];
            if (x < COLS-1){
                bumpiness += Math.abs(columnHeight[x] - columnHeight[x + 1]);
            }
        }
        avgHeight /= COLS;


        double score = game.getScoreThisTurn();
        double ratioHighest = (double)highest / (double)ROWS;
        double unreachableBlockRatio;
        double roofedRatio;
        if (numBlocks!=0){
            unreachableBlockRatio = unreachableBlocks / (double)numBlocks;
            roofedRatio = roofed / (double)numBlocks;
        }else{
            unreachableBlockRatio = 0.0;
            roofedRatio = 0.0;
            this.numUnreachablePrevious = 0; // reset if no blocks,
        }
        double holesDiff = this.numUnreachablePrevious - unreachableBlocks;
        this.numUnreachablePrevious = (int)unreachableBlocks;

        double flatness = 20 - bumpiness; // flatness is good
        flatness /= 20.0;
        holesDiff *= 0.2;
        double avgHeightRatioReverse = 1 - avgHeight / (double)ROWS;
        double unreachableBlockRatioReverse = 1 - unreachableBlockRatio;
        double unreachableRowsRatioReverse =0.0;
        if (highest != ROWS)
            unreachableRowsRatioReverse = 1 - (unreachableRows / (double)(ROWS-highest));


        double reward =  score*2
                + holesDiff
                + flatness
                + avgHeightRatioReverse
                + ratioHighest
                + unreachableBlockRatioReverse
                + unreachableRowsRatioReverse
                - (game.didAgentLose() ? 20.0 : 0.0)
                - roofedRatio;
        return reward;

    }

    @Override
    public void onGameEnd(GameView game){
        System.out.println("score: " + game.getTotalScore());
    }


}

