package classification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class NaiveBayes {

	public static void main(String[] args) {
		String usage = "java "
				+ NaiveBayes.class.getName()
				+ " training_file testing_file [printStats]\n\n"
				+ "Trains a Naive Bayes classifier using the training_file then runs "
				+ "it across the training data in testing_file. Output is line "
				+ "separated count: true positives, false negatives, false positives, "
				+ "true negatives. Stats will be printed if printStats is true.";

		if (args.length < 2) {
			System.out.println(usage);
			System.exit(1);
		}

		File trainingFile = new File(args[0]);
		File testingFile = new File(args[1]);
		boolean printStats = false;
		if (args.length > 2 && args[2].equals("true")) {
			printStats = true;
		}

		try {
			List<DataVector> trainingData = ClassifierRunner
					.parseData(trainingFile);

			// build classifier
			Classifier classifier = new NaiveBayesClassifier(
					trainingData);

			ClassifierRunner.runAndPrintResults(testingFile,
					classifier, printStats);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

}
