package main;

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluateModel;
import moa.tasks.LearnModel;

public class MainTwo {

	/**
	 * @param args
	 */
	
	public static void main(String[] args) {
		// Learning on Random RBF Generator Drift with speed change of 0.001
		// Evaluation on RandomTreeGenerator stream

		Classifier learner = new HoeffdingTree();
		//1st stream for learning the mode, RBF Generator Drift with speed change of 0.001
		double speedChange = 0.001;
		RandomRBFGeneratorDrift learnRbfDriftStream = new RandomRBFGeneratorDrift();
		learnRbfDriftStream.speedChangeOption.setValue(speedChange);
		learnRbfDriftStream.prepareForUse();

		// instantiate necessary variables for learning the model
		int maxInstances = 1000000;
		int numPasses = 1;
		LearnModel lm = new LearnModel(learner, learnRbfDriftStream, maxInstances,
				numPasses);
		Object resultingModel = lm.doTask();

		//2nd stream for evaluation RandomTreeGenerator stream
		RandomTreeGenerator rtStream = new RandomTreeGenerator();
		rtStream.prepareForUse();
		
		// Prepare and start the evaluation
		ClassificationPerformanceEvaluator evaluator = 
				new BasicClassificationPerformanceEvaluator();
		EvaluateModel em = new EvaluateModel();
		em.modelOption.setCurrentObject(resultingModel);
		em.streamOption.setCurrentObject(rtStream);
		em.maxInstancesOption.setValue(maxInstances);
		em.evaluatorOption.setCurrentObject(evaluator);
		Object resultingEvaluation = em.doTask();

		System.out.println("Learning on Random RBF Generator Drift with speed change of 0.001," +
						" Evaluation on RandomTreeGenerator stream");
		
		System.out.println(resultingEvaluation);

		System.out.println("Finished!");
	}

}
