package classification;

public interface Classifier {

	/**
	 * 
	 * @param vector
	 * @return the predicted class label
	 */
	public abstract String classify(DataVector vector);

}
