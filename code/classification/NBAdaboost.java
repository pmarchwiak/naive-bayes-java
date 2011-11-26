package classification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class NBAdaboost {

  public static void main(String[] args) {
    String usage = "java " + NBAdaboost.class.getName()
        + " training_file testing_file [numRounds [printStats]]\n\n" +
        "Trains an ensemble of Naive Bayes classifiers with AdaBoost using " +
        "the training_file then runs it across the training data in " +
        "testing_file. Output is line separated count: true positives, false " +
        "negatives, false positives, true negatives. Stats will be printed " +
        "if printStats is true.";

    if (args.length < 2) {
      System.out.println(usage);
      System.exit(1);
    }

    File trainingFile = new File(args[0]);
    File testingFile = new File(args[1]);
    int numRounds = 5;
    if (args.length > 2) {
      numRounds = Integer.valueOf(args[2]);
    }
    boolean printStats = false;
    if (args.length > 3 && args[3].equals("true")) {
      printStats = true;
    }

    try {
      List<DataVector> trainingData = ClassifierRunner.parseData(trainingFile);
      
      // build classifier
      Classifier classifier = new AdaBoostClassifier(trainingData, numRounds);
      
      ClassifierRunner.runAndPrintResults(testingFile, classifier, printStats);
      
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      System.exit(1);
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }

  }

}
