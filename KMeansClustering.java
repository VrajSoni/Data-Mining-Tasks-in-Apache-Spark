import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.DoubleAccumulator;
import org.apache.spark.util.LongAccumulator;
import org.sparkproject.guava.collect.ArrayListMultimap;
import org.sparkproject.guava.collect.ListMultimap;

import java.util.ArrayList;
import java.util.List;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.*;
import java.lang.Math;

public class KMeansClustering {
	
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("KMeans").setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SparkSession spark= SparkSession.builder().appName("KMeansClustering").master("local").getOrCreate();
		String path = args[0];
		Dataset<Row> df = spark.read().format("csv").option("header", "false").option("inferSchema", "true").load(path);
		//df.printSchema();
		
		int K = 4;
		int features = 3;
		
		Dataset<Row> random = df.sample(false,0.5).limit(K);
		List<Row> centroidRows = new ArrayList<Row>();
		centroidRows = random.collectAsList();
		
		System.out.println(centroidRows);
		int iterations = 10;
		List<List<DoubleAccumulator>> sum = new ArrayList<List<DoubleAccumulator>>();
		for(int i=0;i<K;i++) 
		{
			sum.add(new ArrayList<DoubleAccumulator>());
			for(int j=0;j<features;j++)
			{
				sum.get(i).add(jsc.sc().doubleAccumulator());
			}

		}
		for(int i=0;i<1;i++)
		{
			Broadcast<List<Row>> centroids = jsc.broadcast(centroidRows);
			AtomicInteger cnt = new AtomicInteger(); 
			for(int j=0;j<K;j++) 
			{
				for(int k=0;k<features;k++)
				{
					sum.get(j).get(k).reset();
				}
			}
			df.foreach((ForeachFunction<Row>)row -> {//iterate over each row to assign it to a centroid
				List<Double> temp = new ArrayList<Double>();//To store the distance from each centroid
				for(int j=0;j<K;j++){//Iterate over each centroid and calculate distance
					double curDist = 0.0;
					for(int k=0;k<features;k++)
					{
						double cVal = centroids.getValue().get(j).getDouble(k);
						double pVal = row.getDouble(k);
						
						curDist += (cVal-pVal)*(cVal-pVal);
					}
					curDist = Math.sqrt(curDist);
					temp.add(curDist);
				}
				int minIndex = temp.indexOf(Collections.min(temp));
				for(int j=0;j<features;j++)
				{
					sum.get(minIndex).get(j).add(row.getDouble(j));
				}
			});

			List<Row> newCentroids = new ArrayList<Row>();
			for(int j=0;j<K;j++)
			{
				List<Double> l = new ArrayList<Double>();
				for(int k=0;k<features;k++) l.add(sum.get(j).get(k).avg());
				
				newCentroids.add(RowFactory.create(l.toArray()));
			}
			if(newCentroids == centroids.getValue()) break;
			else centroidRows = newCentroids;
 		}
		
		System.out.println(centroidRows);
	}
}
