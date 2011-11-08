package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.ConsoleHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class NaiveBayesClassifier implements Classifier {

  private static Logger logger = 
      Logger.getLogger(NaiveBayesClassifier.class.getPackage().getName());
  static {
    ConsoleHandler ch = new ConsoleHandler();
    ch.setFormatter(new SimpleFormatter());
    logger.addHandler(ch);
  }
  
  private Map<String, Integer> generalLabelCounts = 
      new HashMap<String, Integer>();
  private ArrayList<Map<String, Map<String, Integer>>> attrValLabelCounts = 
      new ArrayList<Map<String, Map<String, Integer>>>();

  /**
   * Builds a classifier using the given training data
   * @param trainingData a list of {@code DataVector}s
   */
  public NaiveBayesClassifier(List<DataVector> trainingData) {
    int numAttributes = trainingData.get(0).getData().length;

    boolean isFirstVector = true;
    for (int v = 0; v < trainingData.size(); v++) {
      DataVector vector = trainingData.get(v);
      logger.fine("Processing vector " + v);

      String label = vector.getLabel();
      // increment number of tuples with current label
      Integer labelCount = generalLabelCounts.get(label);
      if (labelCount == null) {
        labelCount = 0;
      }
      generalLabelCounts.put(label, labelCount + 1);

      for (int attrIdx = 0; attrIdx < numAttributes; attrIdx++) {
        String attrVal = vector.getData()[attrIdx];

        Map<String, Map<String, Integer>> valueLabelCounts;
        if (isFirstVector) {
          valueLabelCounts = new HashMap<String, Map<String, Integer>>();
          attrValLabelCounts.add(valueLabelCounts);
        }
        valueLabelCounts = attrValLabelCounts.get(attrIdx);

        Map<String, Integer> labelCounts = valueLabelCounts.get(attrVal);
        if (labelCounts == null) {
          labelCounts = new HashMap<String, Integer>();
          valueLabelCounts.put(attrVal, labelCounts);
        }

        Integer count = labelCounts.get(label);
        if (count == null) {
          count = 0;
        }
        labelCounts.put(label, count + 1);
      }

      isFirstVector = false;
    }

    logger.fine("Counts per class:");
    for (Entry<String, Integer> entry : generalLabelCounts.entrySet()) {
      logger.fine(entry.getKey() + ": " + entry.getValue());
    }

    for (int i = 0; i < attrValLabelCounts.size(); i++) {
      for (Entry<String, Map<String, Integer>> entry : attrValLabelCounts
          .get(i).entrySet()) {
        String attrVal = entry.getKey();
        for (Entry<String, Integer> count : entry.getValue().entrySet()) {
          logger.fine(String.format(
              "AttrIdx:%d,attrVal:%s,class:%s: %d", i, attrVal, count.getKey(),
              count.getValue()));
        }
      }
    }

  }

  /**
   * Classifies a given vector of attributes.
   * @param vector
   * @return the predicted class label
   */
  @Override
  public String classify(DataVector vector) {
    float maxProduct = 0;
    String predictedLabel = null;
    
    for (Entry<String, Integer> entry: generalLabelCounts.entrySet()) {
      String label = entry.getKey();
      int numMatchingLabel = entry.getValue();
      
      float product = 1;
      String[] data = vector.getData();
      for (int k = 0; k < data.length; k++) {
        // get number of x_k with label C_i
        
        Map<String, Map<String, Integer>> attrKCounts = attrValLabelCounts.get(k);
        Map<String, Integer> valCounts = attrKCounts.get(data[k]);

        int numMatchingValLabel = 0;
        if (valCounts != null && valCounts.containsKey(label)) {
          numMatchingValLabel = valCounts.get(label);
        }
        
        product *= ((float) numMatchingValLabel / numMatchingLabel);
      }
      
      if (product > maxProduct) {
        maxProduct = product;
        predictedLabel = label;
      }
    }
    
    return predictedLabel;
  }
}
