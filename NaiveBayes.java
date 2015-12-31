import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class NaiveBayes {
	final int d = 57;		
	final int n = 4601;
	final int k = 10;
	double[][] data = new double[n][d+1];
	double[][] means_NB1 = new double[k][d];				//also used for NB2 and NB3

	double[][][] priors_NB1 = new double[k][d][2];			//priors_NB1[fold_i][feature_j being <= mean][spam=1 or nonspam=0]
	double[][] p_NB = new double[k][2];						//p_NB[fold_i][spam=1 or nonspam=0] 
	double[][] size = new double[k][2];						

	double[][][] means_NB2 = new double[k][d][2];			//also used for NB3
	double[][][] vars_NB2 = new double[k][d][2];
	double[][] overalVars_NB2 = new double[k][d];
	
	double[][][][] priors_NB3 = new double[k][d][4][2];		//priors_NB3[fold_i][feature_j][bucket_4][spam=1 or nonspam=0]
	
	NumberFormat formatter = new DecimalFormat("0.#####");
	
	public void readData() {
		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader("spambase.data"));
			String line;
			int t = 0;
			while ((line = in.readLine()) != null) {
				String[] tokens = line.trim().split(",");
				for (int i = 0; i < tokens.length; i++)
					data[t][i] = Double.parseDouble(tokens[i]);
				t++;				
			}
			in.close();
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	public void computePriors_NB1() {
		//compute mean and prior class prob for each fold
		for (int i = 0; i < n; i++) {
			int fold = i%10;
			for (int j = 0; j < k; j++)
				if (fold != j) {
					for (int l = 0; l < d; l++) 
						means_NB1[j][l] += data[i][l];
					size[j][(int)data[i][d]]++;
				}
		}
		
		for (int i = 0; i < k; i++) 
			for (int j = 0; j < d; j++) 
				means_NB1[i][j] /= (size[i][0]+size[i][1]);
		
		//compute conditional prior attribute probs for each fold
		for (int i = 0; i < n; i++) {
			int fold = i%10;
			for (int j = 0; j < d; j++) 
				if (data[i][j] <= means_NB1[fold][j]) 
					for (int l = 0; l < k; l++) 
						if (fold != l) 
							priors_NB1[l][j][(int)data[i][d]]++;			
		}
		
		for (int i = 0; i < k; i++) {			
			for (int j = 0; j < 2; j++)
				for (int l = 0; l < d; l++)
					priors_NB1[i][l][j] = (priors_NB1[i][l][j]+1)/(size[i][j]+2);
			p_NB[i][0] = size[i][0]/(size[i][0]+size[i][1]);
			p_NB[i][1] = size[i][1]/(size[i][0]+size[i][1]);
		}
	}
	
	public void computePriors_NB2() {
		//compute conditional mean, conditional variance for each fold
		for (int i = 0; i < n; i++) {
			int fold = i%10;
			for (int j = 0; j < k; j++)
				if (fold != j)
					for (int l = 0; l < d; l++)
						means_NB2[j][l][(int)data[i][d]] += data[i][l];
		}
				
		for (int i = 0; i < k; i++) 		
			for (int j = 0; j < 2; j++) 
				for (int l = 0; l < d; l++) 
					means_NB2[i][l][j] /= size[i][j];
		
		for (int i = 0; i < n; i++) {
			int fold = i%10;
			for (int j = 0; j < k; j++)
				if (fold != j)
					for (int l = 0; l < d; l++) {
						vars_NB2[j][l][(int)data[i][d]] += Math.pow(data[i][l]-means_NB2[j][l][(int)data[i][d]], 2);
						overalVars_NB2[j][l] += Math.pow(data[i][l]-means_NB1[j][l], 2);
					}
		}
		
		for (int i = 0; i < k; i++) 		
			for (int j = 0; j < 2; j++) 
				for (int l = 0; l < d; l++)
					vars_NB2[i][l][j] /= (size[i][j]-1);
		
		for (int i = 0; i < k; i++) 		
			for (int l = 0; l < d; l++)
				overalVars_NB2[i][l] /= (size[i][0]+size[i][1]-1);

	}
	
	public void computePriors_NB3() {
		//compute conditional prior attribute probs for each fold
		for (int i = 0; i < n; i++) {
			int fold = i%10;
			for (int j = 0; j < d; j++) {
				double low_mean = Math.min(means_NB2[fold][j][0], means_NB2[fold][j][1]);
				double high_mean = Math.max(means_NB2[fold][j][0], means_NB2[fold][j][1]);
				int bucket;
				if (data[i][j] <= low_mean)
					bucket = 0;
				else if (data[i][j] <= means_NB1[fold][j]) 
					bucket = 1;
				else if (data[i][j] <= high_mean) 
					bucket = 2;
				else 
					bucket = 3;
				
				for (int l = 0; l < k; l++)
					if (fold != l)
						priors_NB3[l][j][bucket][(int)data[i][d]]++;			
			}						
		}
		
		for (int i = 0; i < k; i++) 			
			for (int j = 0; j < 2; j++) 
				for (int l = 0; l < d; l++) 
					for (int q = 0; q < 4; q++) 
						priors_NB3[i][l][q][j] = (priors_NB3[i][l][q][j]+1)/(size[i][j]+4);
	}

	public double NB1(double[] p_c, double[][] p_a, double[] means, double[] instance) {		//p_c[2]: class prob, p_a[d][2]: att prob
		double result = Math.log(p_c[1]/p_c[0]);
		for (int i = 0; i < d; i++) 
			if (instance[i] <= means[i])
				result += Math.log(p_a[i][1]/p_a[i][0]);
			else
				result += Math.log((1-p_a[i][1])/(1-p_a[i][0]));
		return result;
	}
	
	public double NB2(double[] p_c, double[][] means, double[][] vars, double[] overalVars, double size, double[] instance) {
		double result = Math.log(p_c[1]/p_c[0]);
		for (int i = 0; i < d; i++) {
			double lambda = size/(size+1);
			double var1 = lambda*vars[i][1]+(1-lambda)*overalVars[i];
			double var0 = lambda*vars[i][0]+(1-lambda)*overalVars[i];
			double p1 = 1/Math.sqrt(2*Math.PI*var1)
				* Math.pow(Math.E, -Math.pow(instance[i]-means[i][1],2)/(2*var1)); 
			double p0 = 1/Math.sqrt(2*Math.PI*var0)
				* Math.pow(Math.E, -Math.pow(instance[i]-means[i][0],2)/(2*var0));
			result += Math.log(p1/p0);			
		}
		return result;
	}
	
	public double NB3(double[] p_c, double[][][] p_a, double[] means, double[][] cond_means, double[] instance) {		//p_a[d][4][2]: att prob
		double result = Math.log(p_c[1]/p_c[0]);
		for (int i = 0; i < d; i++) {
			double low_mean = Math.min(cond_means[i][0], cond_means[i][1]);
			double high_mean = Math.max(cond_means[i][0], cond_means[i][1]);
			int bucket;
			if (instance[i] <= low_mean) 
				bucket = 0;
			else if (instance[i] <= means[i]) 
				bucket = 1;
			else if (instance[i] <= high_mean) 
				bucket = 2;
			else 
				bucket = 3;
			result += Math.log(p_a[i][bucket][1]/p_a[i][bucket][0]);
		}
		return result;
	}
	
	public void evaluate(int model) {
		double[] fp = new double[k];
		double[] fn = new double[k];
		double[] tp = new double[k];
		double[] tn = new double[k];
		
		double[] fpr = new double[k];
		double[] fnr = new double[k];
		double[] err = new double[k];
		
		double avg_fpr = 0;
		double avg_fnr = 0;
		double avg_error = 0;
		
		ArrayList<Instance> rocResults = new ArrayList<Instance>();
		for (int i = 0; i < n; i++) {
			int fold = i%k;
			double result = 0;
			if (model == 1)
				result = NB1(p_NB[fold], priors_NB1[fold], means_NB1[fold], data[i]);
			else if (model == 2)
				result = NB2(p_NB[fold], means_NB2[fold], vars_NB2[fold], overalVars_NB2[fold], size[fold][0]+size[fold][1], data[i]);
			else if (model == 3)
				result = NB3(p_NB[fold], priors_NB3[fold], means_NB1[fold], means_NB2[fold], data[i]);
			if (fold == 1)
				rocResults.add(new Instance((int)data[i][d], result));
			if (result >= 0) 							//predicted spam
				if (data[i][d] == 1)					//actual spam
					tp[fold]++;
				else									//actual non-spam
					fp[fold]++;
			else										//predicted non-spam
				if (data[i][d] == 1)					//actual spam
					fn[fold]++;
				else									//actual non-spam
					tn[fold]++;
		}
		
		System.out.println("MODEL " + model + "\n-------\nFPR,FNR,Error for each fold");
		for (int j = 1; j <= k; j++) {
			int i = j%k;
			fpr[i] = fp[i]/(fp[i]+tn[i]);
			fnr[i] = fn[i]/(fn[i]+tp[i]);
			err[i] = (fp[i]+fn[i])/(fp[i]+fn[i]+tp[i]+tn[i]);
			System.out.println(formatter.format(fpr[i]) + "," + formatter.format(fnr[i]) + "," + formatter.format(err[i]));
		}
		
		for (int i = 0; i < k; i++) {
			avg_fpr += fpr[i];
			avg_fnr += fnr[i];
			avg_error += err[i];
		}
		avg_fpr /= k;
		avg_fnr /= k;
		avg_error /= k;
		System.out.println("Average FPR,FNR,Error across all folds");
		System.out.println(formatter.format(avg_fpr) + "," + formatter.format(avg_fnr) + "," + formatter.format(avg_error));
		
		Collections.sort(rocResults);
		double[] tp_roc = new double[rocResults.size()+1];
		double[] fp_roc = new double[rocResults.size()+1];
		double[] tn_roc = new double[rocResults.size()+1];
		double[] fn_roc = new double[rocResults.size()+1];
		double[] tpr_roc = new double[rocResults.size()+1];
		double[] fpr_roc = new double[rocResults.size()+1];
			
		for (Iterator<Instance> iterator = rocResults.iterator(); iterator.hasNext();) {
			int r = iterator.next().actual;
			if (r == 1)
				tp_roc[0]++;
			else
				fp_roc[0]++;			
		}
		
		fpr_roc[0] = fp_roc[0]/(fp_roc[0]+tn_roc[0]);
		tpr_roc[0] = tp_roc[0]/(tp_roc[0]+fn_roc[0]);
		
		int i = 1;
		for (Iterator<Instance> iterator = rocResults.iterator(); iterator.hasNext();) {
			int r = iterator.next().actual;
			if (r == 1) {
				tp_roc[i] = tp_roc[i-1]-1;
				fp_roc[i] = fp_roc[i-1];
				tn_roc[i] = tn_roc[i-1];
				fn_roc[i] = fn_roc[i-1]+1;
			}
			else {
				tp_roc[i] = tp_roc[i-1];
				fp_roc[i] = fp_roc[i-1]-1;
				tn_roc[i] = tn_roc[i-1]+1;
				fn_roc[i] = fn_roc[i-1];
			}
			fpr_roc[i] = fp_roc[i]/(fp_roc[i]+tn_roc[i]);
			tpr_roc[i] = tp_roc[i]/(tp_roc[i]+fn_roc[i]);
			
			i++;
		}
		
		System.out.println("Points for ROC");		
		for (int j = 0; j < fpr_roc.length; j++)
			System.out.println(fpr_roc[j]+","+tpr_roc[j]);
	}
	
	public static void main(String[] args) {
		NaiveBayes nb = new NaiveBayes();
		
		nb.readData();
		
		nb.computePriors_NB1();
		nb.computePriors_NB2();
		nb.computePriors_NB3();
		
		nb.evaluate(1);			
		nb.evaluate(2);
		nb.evaluate(3);
	}
	
	private class Instance implements Comparable<Instance> {
		int actual;
		double predicted;
		public Instance(int actual, double predicted) {
			this.actual = actual;
			this.predicted = predicted;
		}
		
		@Override
		public int compareTo(Instance inst) {
			if (this.predicted < inst.predicted)
				return -1;
			else if (this.predicted > inst.predicted)
				return 1;
			return 0;
		}		
	}
}
