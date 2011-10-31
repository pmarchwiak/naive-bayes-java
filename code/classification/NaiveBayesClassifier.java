package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class NaiveBayesClassifier {

  private Map<String, Integer> generalLabelCounts = 
      new HashMap<String, Integer>();
  private ArrayList<Map<String, Map<String, Integer>>> attrValLabelCounts = 
      new ArrayList<Map<String, Map<String, Integer>>>();

  public NaiveBayesClassifier(List<DataVector> trainingData) {
    int numAttributes = trainingData.get(0).getData().length;

    boolean isFirstVector = true;
    for (int v = 0; v < trainingData.size(); v++) {
      DataVector vector = trainingData.get(v);
      System.out.println("Processing vector " + v);

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

    System.out.println("Counts per class:");
    for (Entry<String, Integer> entry : generalLabelCounts.entrySet()) {
      System.out.println(entry.getKey() + ": " + entry.getValue());
    }

    for (int i = 0; i < attrValLabelCounts.size(); i++) {
      for (Entry<String, Map<String, Integer>> entry : attrValLabelCounts
          .get(i).entrySet()) {
        String attrVal = entry.getKey();
        for (Entry<String, Integer> count : entry.getValue().entrySet()) {
          System.out.println(String.format(
              "AttrIdx:%d,attrVal:%s,class:%s: %d", i, attrVal, count.getKey(),
              count.getValue()));
        }
      }
    }

  }

  public String classify(DataVector vector) {
    return null;
  }
}
