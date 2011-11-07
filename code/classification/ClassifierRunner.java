package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

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

  static void runAndPrintResults(File testingFile, Classifier classifier)
      throws IOException {
    int numTruePos = 0;
    int numFalseNeg = 0;
    int numFalsePos = 0;
    int numTrueNeg = 0;
    
    List<DataVector> testingData = parseData(testingFile);
    
    // Use the classifier to determine the predicated label of each test 
    // vector and compare against its actual label.
    for (DataVector vector: testingData) {
      int actualLabel = ClassifierRunner.intForLabel(vector.getLabel());
      int predictedLabel = ClassifierRunner.intForLabel(classifier.classify(vector));
      
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
  }

  static Integer intForLabel(String label) {
    if ("-1".equals(label)) {
      return -1;
    }
    else if ("+1".equals(label)) {
      return 1;
    }
    return null;
  }

}
