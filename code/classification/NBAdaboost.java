package classification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class NBAdaboost {

  public static void main(String[] args) {
    String usage = "java " + NBAdaboost.class.getName()
        + " training_file testing_file\n\n" +
        "Trains an ensemble of Naive Bayes classifiers with AdaBoost using " +
        "the training_file then runs it across the training data in " +
        "testing_file. Output is line separated count: true positives, false " +
        "negatives, false positives, true negatives";

    if (args.length != 2) {
      System.out.println(usage);
      System.exit(1);
    }

    File trainingFile = new File(args[0]);
    File testingFile = new File(args[1]);

    try {
      List<DataVector> trainingData = ClassifierRunner.parseData(trainingFile);
      
      // build classifier
      Classifier classifier = new AdaBoostClassifier(trainingData, 3);
      
      ClassifierRunner.runAndPrintResults(testingFile, classifier);
      
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      System.exit(1);
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }

  }

}
