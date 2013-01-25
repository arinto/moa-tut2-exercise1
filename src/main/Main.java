package main;

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.RandomTreeGenerator;

public class Main {
	public static void main(String[] args) {

		Classifier learner = new HoeffdingTree();
		RandomTreeGenerator rtStream = new RandomTreeGenerator();
		rtStream.prepareForUse();

		RandomRBFGeneratorDrift rbfDriftStream = new RandomRBFGeneratorDrift();
		rbfDriftStream.speedChangeOption.setValue(0.001);
		rbfDriftStream.prepareForUse();
	}
}
