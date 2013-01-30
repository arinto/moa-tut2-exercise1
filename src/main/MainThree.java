package main;

import weka.core.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.RandomTreeGenerator;

public class MainThree {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		RandomTreeGenerator stream = new RandomTreeGenerator();
		stream.prepareForUse();
		
		Classifier learner = new HoeffdingTree();
		learner.setModelContext(stream.getHeader());
		learner.prepareForUse();
		
		int numberSamples = 0;
		int numInstances = 1000000;
		
		while(stream.hasMoreInstances() && (numberSamples < numInstances)){
			Instance trainInst = stream.nextInstance();
			numberSamples++;
			learner.trainOnInstance(trainInst);
		}
		
		int numTestingInstances = 1000000;
		int numberSamplesCorrect = 0;
		double speedChange = 0.001;
			
		RandomRBFGeneratorDrift rbfDriftStream = 
				new RandomRBFGeneratorDrift();
		rbfDriftStream.speedChangeOption.setValue(speedChange);
		rbfDriftStream.prepareForUse();
		
		numberSamples = 0;
		
		while(rbfDriftStream.hasMoreInstances() && (numberSamples < numTestingInstances)){
			Instance testInst = rbfDriftStream.nextInstance();
			numberSamples++;
			if(learner.correctlyClassifies(testInst)){
				numberSamplesCorrect++;
			}
		}
		
        double accuracy = ((double)numberSamplesCorrect/(double)numberSamples)*100.0;
        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy");
	}

}
