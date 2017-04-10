import java.util.ArrayList;
public class MatlabNeuralNet {
	ArrayList<Matrix> weights = new ArrayList<Matrix>();
	ArrayList<Matrix> biases = new ArrayList<Matrix>();
	Matrix weight_layer;
	Matrix bias_layer;
	public MatlabNeuralNet(){
		weight_layer = new Matrix(new Double[][]{{0.12546,-0.15884,0.19174,-0.014808,0.13872},{0.32396,-0.37396,-0.14865,-0.065095,0.040594}});
		weights.add(weight_layer);
		weight_layer = new Matrix(new Double[][]{{0.36595,-0.67934,0.33258},{0.83987,0.75453,-0.83017},{1.0061,-0.96057,0.66322},{1.0611,0.79183,-1.1105},{0.32963,-0.4784,1.0789}});
		weights.add(weight_layer);
		weight_layer = new Matrix(new Double[][]{{-0.398},{0.22034},{-0.11844}});
		weights.add(weight_layer);
		bias_layer = new Matrix(new Double[]{-1.6573,1.9481,-1.2606,-0.0029191,-2.0171});
		biases.add(bias_layer);
		bias_layer = new Matrix(new Double[]{-1.727,0.069135,1.6317});
		biases.add(bias_layer);
		bias_layer = new Matrix(new Double[]{-0.18187});
		biases.add(bias_layer);
	}
	
	public Double apply(Double[] input){
		Matrix y = new Matrix(input);
		for (int i = 0; i < weights.size(); i++){
			if (i == weights.size()-1){
				//y = weights{j} * y + bias{j};
				y = weights.get(i).transpose().times(y);
				y = y.addCoefficient(biases.get(i).valueAt(0,0));
			}else{
				//y = 2 ./ (1 + exp(-2 * (weights{j} * y + bias{j}))) - 1;
				y = weights.get(i).transpose().times(y);//.transpose();
				y = biases.get(i).plus(y);
				y = y.multiplyCoefficient(-2.0);
				y = y.exp();
				y = y.addCoefficient(1.0);
				y = y.coefficientOver(2.0);
				y = y.addCoefficient(-1.0);
			}
		}
		return y.valueAt(0,0);
	}
}