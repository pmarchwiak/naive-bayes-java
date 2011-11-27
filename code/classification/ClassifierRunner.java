package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * Utility methods for running the classifiers.
 * @author pmarchwiak
 *
 */
public class ClassifierRunner {

  static Logger logger = 
      Logger.getLogger(NaiveBayes.class.getPackage().getName());
  static {
    ConsoleHandler ch = new ConsoleHandler();
    ch.setFormatter(new SimpleFormatter());
    logger.addHandler(ch);
  }
  
  static List<DataVector> parseData(File dataFile) throws IOException {
    FileInputStream fis = new FileInputStream(dataFile);
    BufferedReader br = new BufferedReader(new InputStreamReader(fis));
    String line;
    List<DataVector> dataSet = new ArrayList<DataVector>();
    while ((line = br.readLine()) != null) {
      String[] lineData = line.split("\t");
      String label = lineData[0];
  
      logger.log(Level.FINER, "Parsing line " + line);
  
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

  static void runAndPrintResults(File testingFile, Classifier classifier,
      boolean printStats) throws IOException {
    int tp = 0; // true positive
    int fn = 0; // false neg
    int fp = 0; // false pos
    int tn = 0; // true neg
    
    List<DataVector> testingData = parseData(testingFile);
    
    // Use the classifier to determine the predicted label of each test 
    // vector and compare against its actual label.
    for (DataVector vector: testingData) {
      int actualLabel = ClassifierRunner.intForLabel(vector.getLabel());
      int predictedLabel = 
          ClassifierRunner.intForLabel(classifier.classify(vector));
      
      if (actualLabel == 1 && predictedLabel == 1) {
        tp++;
      }
      else if (actualLabel == 1 && predictedLabel == -1) {
        fn++;
      }
      else if (actualLabel == -1 && predictedLabel == 1) {
        fp++;
      }
      else if (actualLabel == -1 && predictedLabel == -1) {
        tn++;
      }
    }
    
    System.out.println(tp);
    System.out.println(fn);
    System.out.println(fp);
    System.out.println(tn);
    
    if (printStats) {
      StringBuilder sb = new StringBuilder();
      
      int p = tp + fn;
      int n = tn + fp;
      
      DecimalFormat df = new DecimalFormat("#.#####");
      //accuracy
      sb.append(df.format(((double)(tp + tn)) / (p + n))).append(",");
      
      // error rate
      sb.append(df.format(((double)(fp + fn)) / (p + n))).append(",");
      
      double recall = ((double) tp) / p;
      sb.append(df.format(recall)).append(",");
      
      // specificity
      sb.append(df.format(((double) tn) / n)).append(",");
      
      double precision = ((double) tp) / (tp + fp);
      sb.append(df.format(precision)).append(",");
      
      // F-1 score
      sb.append(df.format(
          ((double)(2 * precision * recall))/(precision + recall))).append(",");
      
      // F-Beta scores
      sb.append(df.format(fBeta(0.5, precision, recall))).append(",");
      sb.append(df.format(fBeta(2, precision, recall)));
      
      System.out.println(sb);
    }
  }

  public static double fBeta(double beta, double precision, double recall) {
    double betaSq = Math.pow(beta, 2);
    return ((1+betaSq)*precision*recall)/((betaSq*precision)+recall);
  }
  
  static Integer intForLabel(String label) {
    if ("-1".equals(label)) {
      return -1;
    }
    else if ("+1".equals(label)) {
      return 1;
    }
    logger.fine("unrecognized label: " + label);
    return null;
  }

}
