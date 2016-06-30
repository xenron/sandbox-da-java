package dg.ann.courseware.ch05;

import geym.nn.adaline.Adaline;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.util.TransferFunctionType;

import java.util.Arrays;

public class AdalineDemo implements LearningEventListener {
    public final static int CHAR_WIDTH = 5;
    public final static int CHAR_HEIGHT = 7;
    public static String[][] DIGITS = {
            {
                    " OOO ",
                    "O   O",
                    "O   O",
                    "O   O",
                    "O   O",
                    "O   O",
                    " OOO "},

            {
                    "  O  ",
                    " OO  ",
                    "O O  ",
                    "  O  ",
                    "  O  ",
                    "  O  ",
                    "  O  "},

            {
                    " OOO ",
                    "O   O",
                    "    O",
                    "   O ",
                    "  O  ",
                    " O   ",
                    "OOOOO"},

            {
                    " OOO ",
                    "O   O",
                    "    O",
                    " OOO ",
                    "    O",
                    "O   O",
                    " OOO "},

            {
                    "   O ",
                    "  OO ",
                    " O O ",
                    "O  O ",
                    "OOOOO",
                    "   O ",
                    "   O "},

            {
                    "OOOOO",
                    "O    ",
                    "O    ",
                    "OOOO ",
                    "    O",
                    "O   O",
                    " OOO "},

            {
                    " OOO ",
                    "O   O",
                    "O    ",
                    "OOOO ",
                    "O   O",
                    "O   O",
                    " OOO "},

            {
                    "OOOOO",
                    "    O",
                    "    O",
                    "   O ",
                    "  O  ",
                    " O   ",
                    "O    "},

            {
                    " OOO ",
                    "O   O",
                    "O   O",
                    " OOO ",
                    "O   O",
                    "O   O",
                    " OOO "},

            {
                    " OOO ",
                    "O   O",
                    "O   O",
                    " OOOO",
                    "    O",
                    "O   O",
                    " OOO "}};

    public static void main(String[] args) {
        Adaline ada = new Adaline(CHAR_WIDTH * CHAR_HEIGHT, DIGITS.length, 0.01d, TransferFunctionType.GAUSSIAN);
        DataSet ds = new DataSet(CHAR_WIDTH * CHAR_HEIGHT, DIGITS.length);
        for (int i = 0; i < DIGITS.length; i++) {
            ds.addRow(createTrainDataRow(DIGITS[i], i));
        }
        ada.getLearningRule().addListener(new AdalineDemo());
        ada.learn(ds);
        for (int i = 0; i < DIGITS.length; i++) {
            ada.setInput(image2data(DIGITS[i]));
            ada.calculate();
            printDigit(DIGITS[i]);
            System.out.println(maxIndex(ada.getOutput()));
            System.out.println(Arrays.toString(ada.getOutput()));
            System.out.println();
        }
    }

    private static int maxIndex(double[] data) {
        int result = -1;
        for (int i = 0; i < data.length; i++) {
            if (result == -1 || data[i] > data[result]) {
                result = i;
            }
        }
        return result;
    }

    private static void printDigit(String[] digit) {
        for (int i = 0; i < digit.length; i++) {
            if (i == digit.length - 1) {
                System.out.print(digit[i] + "===>");
            } else {
                System.out.println(digit[i]);
            }
        }
    }

    private static double[] image2data(String[] image) {
        double[] input = new double[CHAR_WIDTH * CHAR_HEIGHT];

        for (int row = 0; row < CHAR_HEIGHT; row++) {
            for (int col = 0; col < CHAR_WIDTH; col++) {
                int index = (row * CHAR_WIDTH) + col;
                char ch = image[row].charAt(col);
                input[index] = ch == 'O' ? 1 : -1;
            }
        }

        return input;
    }

    private static DataSetRow createTrainDataRow(String[] image, int ideaValue) {
        double[] output = new double[DIGITS.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = -1;
        }
        double[] input = image2data(image);
        output[ideaValue] = 1;
        DataSetRow dsr = new DataSetRow(input, output);
        return dsr;
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        IterativeLearning bp = (IterativeLearning) event.getSource();
        System.out.println("iterate:" + bp.getCurrentIteration());
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
    }
}
