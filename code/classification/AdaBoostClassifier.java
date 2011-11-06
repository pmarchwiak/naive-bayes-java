package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.logging.Logger;

import com.sun.org.apache.xerces.internal.impl.dv.dtd.NMTOKENDatatypeValidator;

public class AdaBoostClassifier implements Classifier {

  private static Logger logger = 
      Logger.getLogger(AdaBoostClassifier.class.getPackage().getName());
  
  public AdaBoostClassifier(List<DataVector> trainingData, int numRounds) {     
    Random random = new Random();
    int numSamples = trainingData.size();
    
    Map<Classifier, Float> modelErrorRates = new HashMap<Classifier, Float>();
    
    //init tuple weights to 1 / numSamples
    Map<DataVector, Float> weights = new HashMap<DataVector, Float>();
    float initialWeight = ((float) 1) / numSamples;
    for (DataVector v: trainingData) {
      weights.put(v, initialWeight);
    }
    
    // build the composite model
    for (int k = 0; k < numRounds; k++) {
      // sample D with replacement
      // TODO base probability of selection on weight
      List<DataVector> dataForRound = new ArrayList<DataVector>();
      for (int i = 0; i < numSamples; i++) {
        DataVector randomSelection = trainingData.get(random.nextInt(numSamples));
        dataForRound.add(randomSelection);
      }
      
      float errorRate = 0;
      boolean[] isCorrect = new boolean[numSamples];
      do {
        // create model for this round
        NaiveBayesClassifier nb = new NaiveBayesClassifier(dataForRound);
        
        // calculate error rate 
        
        for (int n = 0; n < numSamples; n++) {
          DataVector v = dataForRound.get(n);
          String predicted = nb.classify(v);
          
          if (predicted.equals(v.getLabel())) {
            isCorrect[n] = true;
          }
          else {
            errorRate += weights.get(v);
          }
        }
        
        modelErrorRates.put(nb, errorRate);
      } 
      while (errorRate > 0.5);
      
      float oldWeightsSum = 0;
      for (Float weight: weights.values()) {
        oldWeightsSum += weight;
      }
      
      // TODO multiply the weight of each correctly classified tuple 
      // by errR/(1-errR)
      float weightMultiplier = errorRate / (1 - errorRate);
      
      for (int n = 0; n < numSamples; n++) {
        if (isCorrect[n]) {
          // FIXME if vector appears more than once it shouldn't be multiplied 
          // twice
          DataVector vec = dataForRound.get(n);
          float oldWeight = weights.get(vec);
          float newWeight = oldWeight * weightMultiplier;
          weights.put(vec, newWeight);
        }
      }
      
      float newWeightsSum = 0;
      for (Float weight: weights.values()) {
        newWeightsSum += weight;
      }
      
      // normalize the weight of each tuple
      float multiplier = oldWeightsSum / newWeightsSum;
      for (Entry<DataVector, Float> entry: weights.entrySet()) {
        float weight = entry.getValue();
        weights.put(entry.getKey(), weight * multiplier);
      }
    }
  }

  @Override
  public String classify(DataVector vector) {
    // TODO Auto-generated method stub
    return null;
  }
}
