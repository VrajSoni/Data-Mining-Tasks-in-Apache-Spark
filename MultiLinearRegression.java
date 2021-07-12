import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.python.DoubleArrayToWritableConverter;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.DoubleAccumulator;
public class MultiLinearRegression {
	public static void main(String[] args) throws FileNotFoundException {
		SparkConf conf = new SparkConf().setAppName("MLRegression").setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SparkSession spark= SparkSession.builder().appName("MultiLinearRegression").master("local").getOrCreate();
		String path = args[0];
		Dataset<Row> df = spark.read().format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", " ").load(path);
		
		//df.printSchema();
		//Note - Normalization Remaining
		//df.show();
		int iterations = 10;
		int features = 3;
		long m = df.count();
				
		List<Double> x = new ArrayList<Double>();
		for(int i=0;i<features+1;i++)
		{
			x.add(1.0);
		}
		// wo + w1x1 + w2x2 + w3x3;
		//For given dataset x1 = RM, x2 = AGE, x3 = B.
		double learningRate = 1e-5;
		//System.out.println(learningRate);
		DoubleAccumulator deriv = jsc.sc().doubleAccumulator();
		for(int i=0;i<iterations;i++)
		{
			Broadcast<List<Double>> omega = jsc.broadcast(x);
			List<DoubleAccumulator> predicted = new ArrayList<DoubleAccumulator>();;
			Broadcast<List<DoubleAccumulator>> pred;
			for(int j=0;j<m;j++) predicted.add(jsc.sc().doubleAccumulator());
			AtomicInteger cnt = new AtomicInteger(); 
			df.foreach((ForeachFunction<Row>)row -> {
				for(int w = 0;w < features+1;w++)
				{
					if(w!=0) predicted.get(cnt.intValue()).add(omega.getValue().get(w) * row.getDouble(w-1));//wo + w1x1 + w2x2 + w3x3
					else predicted.get(cnt.intValue()).add(omega.getValue().get(w));
				}
				cnt.addAndGet(1);
			});
//			for(int j=0;j<m;j++)
//			{
//				System.out.println(predicted.get(j));
//			}
			pred = jsc.broadcast(predicted);
			for(int w = 0;w <= features;w++)
			{
				cnt.set(0);
				deriv.setValue(0.0);
				final int W = w;
				df.foreach((ForeachFunction<Row>)row -> {
					double xij = 1;
					if(W != 0) xij = row.getDouble(W-1);
					//System.out.println(pred.value().get(cnt.get()));
					double err_i = pred.value().get(cnt.get()).value() - row.getDouble(features);// predicted - y
					//System.out.println(err_i);
					deriv.add(err_i * xij);
					cnt.addAndGet(1);
				});
				x.set(w, omega.getValue().get(w) - learningRate * deriv.value() / m); 
			}
			System.out.println(x);
		}
		
	}
}
