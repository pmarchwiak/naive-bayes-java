package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class AdaBoostClassifier implements Classifier {

  public AdaBoostClassifier(List<DataVector> trainingData, int numRounds) {     
    Random random = new Random();
    int numSamples = trainingData.size();
    
    // models and their error rates
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
      List<DataVector> dataForRound = new ArrayList<DataVector>();
      for (int i = 0; i < numSamples; i++) {
        DataVector randomSelection = trainingData.get(random.nextInt(numSamples));
        dataForRound.add(randomSelection);
      }
      
      float errorRate = 0;
      do {
        // create model for this round
        NaiveBayesClassifier nb = new NaiveBayesClassifier(dataForRound);
        
        // calculate error rate 
        
        for (DataVector v: dataForRound) {
          String predicted = nb.classify(v);
          if (!predicted.equals(v.getLabel())) {
            errorRate += weights.get(v); 
          }
        }
        
        modelErrorRates.put(nb, errorRate);
      } 
      while (errorRate > 0.5);
      
      // TODO multiply the weight of each correctly classified tuple 
      // by errR/(1-errR)
      
      // TODO normalize the weight of each tuple
    }
  }

  @Override
  public String classify(DataVector vector) {
    // TODO Auto-generated method stub
    return null;
  }
}
