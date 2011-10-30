package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class NaiveBayes {

	public static void main(String[] args) {
		String usage = "java " + NaiveBayes.class.getName() + " training_file testing_file";
		
		if (args.length != 2) {
		  System.out.println(usage);
		  System.exit(1);
		}
		
		File trainingFile = new File(args[0]);
		File testingFile = new File(args[1]);
		
		List<DataVector> trainingData;
    try {
      trainingData = parseData(trainingFile);
      NaiveBayesClassifier classifier = new NaiveBayesClassifier(trainingData);
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

	}

  private static List<DataVector> parseData(File dataFile) throws IOException {
    FileInputStream fis = new FileInputStream(dataFile);
    BufferedReader br = new BufferedReader(new InputStreamReader(fis));
    String line;
    List<DataVector> dataSet = new ArrayList<DataVector>();
    while((line = br.readLine()) != null) {
      String[] lineData = line.split("\t");
      String label = lineData[0];
      ArrayList<String> data = new ArrayList<String>();
      for (int i = 1; i < lineData.length; i++) {
        data.add(lineData[i]);
      }
      
      dataSet.add(new DataVector(label, data.toArray(new String[]{})));
    }
    
    return dataSet;
  }

}
