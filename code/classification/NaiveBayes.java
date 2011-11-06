package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class NaiveBayes {

  private static Logger logger = 
      Logger.getLogger(NaiveBayes.class.getPackage().getName());
  static {
    ConsoleHandler ch = new ConsoleHandler();
    ch.setFormatter(new SimpleFormatter());
    logger.addHandler(ch);
  }

  public static void main(String[] args) {
    String usage = "java " + NaiveBayes.class.getName()
        + " training_file testing_file\n\n" +
        "Trains a Naive Bayes classifier using the training_file then runs " +
        "it across the training data in testing_file. Output is line " +
        "separated count: true positives, false negatives, false positives, " +
        "true negatives";

    if (args.length != 2) {
      System.out.println(usage);
      System.exit(1);
    }

    File trainingFile = new File(args[0]);
    File testingFile = new File(args[1]);

    try {
      List<DataVector> trainingData = parseData(trainingFile);
      
      // build classifier
      Classifier classifier = new NaiveBayesClassifier(trainingData);
      
      int numTruePos = 0;
      int numFalseNeg = 0;
      int numFalsePos = 0;
      int numTrueNeg = 0;
      
      List<DataVector> testingData = parseData(testingFile);
      
      // Use the classifier to determine the predicated label of each test 
      // vector and compare against its actual label.
      for (DataVector vector: testingData) {
        int actualLabel = intForLabel(vector.getLabel());
        int predictedLabel = intForLabel(classifier.classify(vector));
        
        if (actualLabel == 1 && predictedLabel == 1) {
          numTruePos++;
        }
        else if (actualLabel == 1 && predictedLabel == -1) {
          numFalseNeg++;
        }
        else if (actualLabel == -1 && predictedLabel == 1) {
          numFalsePos++;
        }
        else if (actualLabel == -1 && predictedLabel == -1) {
          numTrueNeg++;
        }
      }
      
      System.out.println(numTruePos);
      System.out.println(numFalseNeg);
      System.out.println(numFalsePos);
      System.out.println(numTrueNeg);
      
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      System.exit(1);
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }

  }

  private static List<DataVector> parseData(File dataFile) throws IOException {
    FileInputStream fis = new FileInputStream(dataFile);
    BufferedReader br = new BufferedReader(new InputStreamReader(fis));
    String line;
    List<DataVector> dataSet = new ArrayList<DataVector>();
    while ((line = br.readLine()) != null) {
      String[] lineData = line.split("\t");
      String label = lineData[0];

      logger.log(Level.FINE, "Parsing line " + line);

      ArrayList<String> data = new ArrayList<String>();
      for (int i = 1; i < lineData.length; i++) {
        data.add(lineData[i]);
      }

      if (data.size() > 0) {
        dataSet.add(new DataVector(label, data.toArray(new String[] {})));
      }
    }

    return dataSet;
  }
  
  private static Integer intForLabel(String label) {
    if ("-1".equals(label)) {
      return -1;
    }
    else if ("+1".equals(label)) {
      return 1;
    }
    return null;
  }

}
