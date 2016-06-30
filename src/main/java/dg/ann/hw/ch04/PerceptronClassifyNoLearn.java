package dg.ann.hw.ch04;


import java.util.Arrays;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class PerceptronClassifyNoLearn extends NeuralNetwork {

    public PerceptronClassifyNoLearn(int inputNeuronsCount) {
        super();
        this.createNetwork(inputNeuronsCount);
    }

    public static void main(String[] args) {
        DataSet trainingSet = new DataSet(2, 2);
        trainingSet.addRow(new DataSetRow(new double[]{1, 2}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{2, -3}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{2, -1}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{-1, 2}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{-2, 1}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{-1, -1}, new double[]{Double.NaN, Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{-2, -2}, new double[]{Double.NaN, Double.NaN}));

        PerceptronClassifyNoLearn perceptron = new PerceptronClassifyNoLearn(2);

        for (DataSetRow row : trainingSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();
            double[] networkOutput = perceptron.getOutput();
//            System.out.println(Arrays.toString(row.getInput()) + "=" + Arrays.toString(networkOutput));
            System.out.println(Arrays.toString(row.getInput()) + "，位于 " + getQuadrantInfo(networkOutput));
        }

    }

    private void createNetwork(int inputNeuronsCount) {

        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);

        inputLayer.addNeuron(new BiasNeuron());

        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
        Layer outputLayer = LayerFactory.createLayer(2, outputNeuronProperties);
        this.addLayer(outputLayer);

        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);

        Neuron n = outputLayer.getNeuronAt(0);
        n.getInputConnections()[0].getWeight().setValue(-3);
        n.getInputConnections()[1].getWeight().setValue(-1);
        n.getInputConnections()[2].getWeight().setValue(1);

        n = outputLayer.getNeuronAt(1);
        n.getInputConnections()[0].getWeight().setValue(1);
        n.getInputConnections()[1].getWeight().setValue(-2);
        n.getInputConnections()[2].getWeight().setValue(0);
    }

    private static String getQuadrantInfo(double[] networkOutput) {
        String quadrantInfo = "";
        if (Arrays.equals(networkOutput, new double[]{0.0, 0.0})) {
            quadrantInfo = "第一象限";
        } else if (Arrays.equals(networkOutput, new double[]{0.0, 1.0})) {
            quadrantInfo = "第二象限";
        } else if (Arrays.equals(networkOutput, new double[]{1.0, 0.0})) {
            quadrantInfo = "第三象限";
        } else if (Arrays.equals(networkOutput, new double[]{1.0, 1.0})) {
            quadrantInfo = "第四象限";
        }
        return quadrantInfo;
    }
}

