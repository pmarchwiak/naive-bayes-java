package classification;

/**
 * Represents a data instance or tuple along with a label, if known.
 * 
 * @author pmarchwiak
 * 
 */
public class DataVector {
	private String label;
	private String[] data;

	public DataVector(String label, String[] data) {
		this.label = label;
		this.data = data;
	}

	public String getLabel() {
		return label;
	}

	public String[] getData() {
		return data;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("{label: ");
		sb.append(label);
		sb.append(", data: [");
		for (int i = 0; i < data.length; i++) {
			sb.append(data[i]);
			sb.append(", ");
		}
		sb.append("]}");
		return sb.toString();
	}
}
