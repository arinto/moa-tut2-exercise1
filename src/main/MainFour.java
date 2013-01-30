package main;

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.RandomTreeGenerator;
import weka.core.Instance;

public class MainFour {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		Classifier learner = new HoeffdingTree();
		
		double speedChange = 0.001;
		RandomRBFGeneratorDrift rbfDriftStream = 
				new RandomRBFGeneratorDrift();
		rbfDriftStream.speedChangeOption.setValue(speedChange);
		rbfDriftStream.prepareForUse();
			
		learner.setModelContext(rbfDriftStream.getHeader());
		learner.prepareForUse();
		
		int numberSamples = 0;
		int numInstances = 1000000;
		
		while(rbfDriftStream.hasMoreInstances() && (numberSamples < numInstances)){
			Instance trainInst = rbfDriftStream.nextInstance();
			numberSamples++;
			learner.trainOnInstance(trainInst);
		}
		
		int numTestingInstances = 1000000;
		int numberSamplesCorrect = 0;

		RandomTreeGenerator stream = new RandomTreeGenerator();
		stream.prepareForUse();
		
		numberSamples = 0;
		
		while(stream.hasMoreInstances() && (numberSamples < numTestingInstances)){
			Instance testInst = stream.nextInstance();
			numberSamples++;
			if(learner.correctlyClassifies(testInst)){
				numberSamplesCorrect++;
			}
		}
		
        double accuracy = ((double)numberSamplesCorrect/(double)numberSamples)*100.0;
        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy");

	}

}
