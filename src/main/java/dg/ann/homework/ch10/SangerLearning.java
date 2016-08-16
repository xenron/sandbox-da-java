package dg.ann.hw.ch10;

import java.util.Iterator;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.core.learning.stop.StopCondition;
import org.neuroph.nnet.learning.UnsupervisedHebbianLearning;

public class SangerLearning extends UnsupervisedHebbianLearning {
	private double delta=0;
	private Double[] lastItrWeights;
	
	public class MinWeightChangeStopCondition implements StopCondition {
		private IterativeLearning learningRule;

		public MinWeightChangeStopCondition(IterativeLearning learningRule) {
			this.learningRule = learningRule;
		}

		@Override
		public boolean isReached() {
			return delta<0.000001;
		}

	}
	
	
	
	@Override
    protected void beforeEpoch() {
		lastItrWeights=this.getNeuralNetwork().getWeights();
    }

	@Override
    protected void afterEpoch() {
		Double[] currentWeights=this.getNeuralNetwork().getWeights();
		delta= 0;
		for(int i=0;i<currentWeights.length;i++){
			delta+=Math.pow((currentWeights[i]-lastItrWeights[i]),2);
		}
		delta=Math.sqrt(delta);
    }
    
	@Override
	public void doLearningEpoch(DataSet trainingSet) {
		Iterator<DataSetRow> iterator = trainingSet.iterator();
		while (iterator.hasNext() && !isStopped()) {
			DataSetRow trainingSetRow = iterator.next();
			learnPattern(trainingSetRow);
		}
	}
	
	
    @Override
    protected void onStart() {
    	super.onStart();
    	stopConditions.add(new MinWeightChangeStopCondition(this));
    }
    
    @Override
    protected void updateNetworkWeights() {
        Neuron[] arr$ = this.neuralNetwork.getOutputNeurons();
        int len$ = arr$.length;

        for(int i$ = 0; i$ < len$; ++i$) {
            Neuron neuron = arr$[i$];
            double output = neuron.getOutput();
            for (int j$ = 0; j$ < neuron.getInputConnections().length; ++j$) {
                Connection connection = neuron.getInputConnections()[j$];
                double current_input = connection.getInput();
                double weight = connection.getWeight().getValue();
                double sum_before_input = 0;
                for(int k$ = 0; k$ < i$; ++k$){
                    sum_before_input+=arr$[k$].getInputConnections()[j$].getInput();
                }
                double deltaWeight = ((current_input-sum_before_input) - output * weight) * output * this.learningRate;
                connection.getWeight().inc(deltaWeight);
            }
        }
    }
    
	// @Override
	// protected void updateNeuronWeights(Neuron neuron) {
	// 	double output = neuron.getOutput();
	// 	for(Connection connection : neuron.getInputConnections()) {
	// 		double input = connection.getInput();
	// 		double weight = connection.getWeight().getValue();
	// 		double deltaWeight = (input - output*weight) * output * this.learningRate;
	// 		connection.getWeight().inc(deltaWeight);
	// 	}
	// }
}
