package dg.ann.courseware.ch07;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.mlperceptron.MlPerceptronLineOutput;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

public class StockPrediction1 implements LearningEventListener {

    public static final int Input_Attribute_Len = 6;
    public static final int Output_Attribute_Len = 1;
    public static final double maxError = 0.01d;

    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        new StockPrediction1().run();
    }

    /**
     * @param filepath: like "/dg/ann/courseware/ch07/stock1.txt"
     * @return
     * @throws IOException
     */
    public static List<DataSetRow> getTrainData(String filepath) throws IOException {
        InputStream in = StockPrediction1.class.getResourceAsStream(filepath);
        BufferedReader br = new BufferedReader(new InputStreamReader(new DataInputStream(in)));
        List<Double> prices = new Vector<Double>();
        List<DataSetRow> re = new Vector<DataSetRow>();
        String line = null;
        int i=0;
        while ((line = br.readLine()) != null) {
            String[] item = line.split("\t");
            double[] inputs = new double[Input_Attribute_Len];
//            int i=0;
//            for(i=0;i<Attribute_Len;i++){
//                inputs[i]=Double.parseDouble(item[i]);
//            }
            inputs[0] = Double.parseDouble(item[1]);
            inputs[1] = Double.parseDouble(item[2]);
            inputs[2] = Double.parseDouble(item[5]);
            inputs[3] = Double.parseDouble(item[6]);
            inputs[4] = Double.parseDouble(item[7]);
            inputs[5] = Double.parseDouble(item[8]);

//            inputs[0] = i++;
//            inputs[1] = Double.parseDouble(item[1]);
//            inputs[2] = Double.parseDouble(item[2]);
//            inputs[3] = Double.parseDouble(item[5]);
//            inputs[4] = Double.parseDouble(item[6]);
//            inputs[5] = Double.parseDouble(item[7]);
//            inputs[6] = Double.parseDouble(item[8]);

//            inputs[0] = i++;
//            inputs[1] = Double.parseDouble(item[1]);
//            inputs[2] = Double.parseDouble(item[7]);
//            inputs[3] = Double.parseDouble(item[8]);

            double[] outputs = new double[Output_Attribute_Len];
            if (Double.parseDouble(item[3]) > 0) {
                outputs[0] = 1;
            } else {
                outputs[0] = 0;
            }
//            outputs[0] = Double.parseDouble(item[3]);

//            for(;i<Attribute_Len+7;i++){
//                outputs[i-Attribute_Len]=Double.parseDouble(item[i]);
//            }
            re.add(new DataSetRow(inputs, outputs));
        }
        return re;
    }


    public void run() throws IOException {

        DataSet trainingSet = new DataSet(Input_Attribute_Len, Output_Attribute_Len);
        List<DataSetRow> rows = getTrainData("/dg/ann/courseware/ch07/stock1.txt");
        for (int i = 0; i < rows.size(); i++) {
            trainingSet.addRow(rows.get(i));
        }

//        MlPerceptron myMlPerceptron = new MlPerceptron(TransferFunctionType.SIGMOID, Input_Attribute_Len, 6, Output_Attribute_Len);
        MlPerceptron myMlPerceptron = new MlPerceptronLineOutput(TransferFunctionType.SIGMOID, Input_Attribute_Len, 3, Output_Attribute_Len);
        myMlPerceptron.setLearningRule(new BackPropagation());

        myMlPerceptron.getLearningRule().setMaxError(maxError);
        myMlPerceptron.getLearningRule().addListener(this);

        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron);

    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet) throws IOException {
        List<DataSetRow> rows = getTrainData("/dg/ann/courseware/ch07/stock1.txt");
        for (DataSetRow row : rows) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();
            System.out.print("Input: " + Arrays.toString(row.getInput()));
            System.out.println(" Output: " + Arrays.toString(networkOutput));
            System.out.println("Desired Output: " + Arrays.toString(row.getDesiredOutput()));
        }

    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation) event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : " + bp.getTotalNetworkError());
    }
}
