package dg.ann.hw.ch08;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;
import org.neuroph.nnet.comp.neuron.InputOutputNeuron;
import org.neuroph.nnet.learning.UnsupervisedHebbianLearning;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

import java.util.Arrays;

public class HebbSample {

    public static void main(String args[]) {
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("neuronType", InputOutputNeuron.class);
        neuronProperties.setProperty("bias", new Double(0.0D));
        neuronProperties.setProperty("transferFunction", TransferFunctionType.SGN);
        neuronProperties.setProperty("transferFunction.yHigh", new Double(1.0D));
        neuronProperties.setProperty("transferFunction.yLow", new Double(-1.0D));

        DataSet trainingSet = new DataSet(4);
        trainingSet.addRow(new DataSetRow(new double[] { 1, 1, -1, -1 }));

        trainingSet.addRow(new DataSetRow(new double[] { 1, -1, 1, -1 }));

        // create hopfield network
        Hopfield myHopfield = new Hopfield(4, neuronProperties);
        myHopfield.setLearningRule(new HebbianLearning());
        // learn the training set
        myHopfield.learn(trainingSet);

        // test hopfield network
        System.out.println("Testing network");

        DataSetRow h = new DataSetRow(new double[] { 1, 1, -1, 0 });
        trainingSet.addRow(h);

        myHopfield.setInput(h.getInput());

        double[] networkOutput = null;
        double[] preNetworkOutput = null;
        while (true) {
            myHopfield.calculate();
            networkOutput = myHopfield.getOutput();
            if (preNetworkOutput == null) {
                preNetworkOutput = networkOutput;
                continue;
            }
            if (Arrays.equals(networkOutput, preNetworkOutput)) {
                break;
            }
            preNetworkOutput = networkOutput;
        }

        System.out.println("Input : " + Arrays.toString(h.getInput()));
        System.out.println("Output: " + Arrays.toString(networkOutput));

    }
}

class HebbianLearning extends UnsupervisedHebbianLearning {
    private static final long serialVersionUID = 1L;

    protected void updateNeuronWeights(Neuron neuron) {
        double output = neuron.getOutput();
        Connection[] arr$ = neuron.getInputConnections();
        int len$ = arr$.length;

        for(int i$ = 0; i$ < len$; ++i$) {
            Connection connection = arr$[i$];
            double input = connection.getInput();
            if((input <= 0.0D || output <= 0.0D) && (input > 0.0D || output > 0.0D)) {
                connection.getWeight().dec(this.learningRate);
            } else {
                connection.getWeight().inc(this.learningRate);
            }
        }

    }
}