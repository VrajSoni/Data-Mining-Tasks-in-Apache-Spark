import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.DoubleAccumulator;

public class KNN {
	public static int getIndexOfLargest( int[] array )
	{
	  if ( array == null || array.length == 0 ) return -1; // null or empty

	  int largest = 0;
	  for ( int i = 1; i < array.length; i++ )
	  {
	      if ( array[i] > array[largest] ) largest = i;
	  }
	  return largest; // position of the first largest found
	}
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("KNN").setMaster("local");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SparkSession spark= SparkSession.builder().appName("KNNAlgo").master("local").getOrCreate();
		String trainDatapath = args[0];
		String testDataPath = args[1];
		Dataset<Row> train = spark.read().format("csv").option("header", "false").option("inferSchema", "true").load(trainDatapath);
		
		long rows = train.count();
		int cols = 2;
		int k = 3;
		int totalCategories = 2;
		
		//Dataset<Row> test = spark.read().format("csv").option("header", "false").option("inferSchema", "true").load(testDataPath);
		List<List<Double> > testData = new ArrayList<List<Double> >();
		testData.add(new ArrayList<Double>(Arrays.asList(4.5, 4.5)));
		testData.add(new ArrayList<Double>(Arrays.asList(9.5, 10.5)));
		
		AtomicInteger cnt = new AtomicInteger(); 
		cnt.set(1);
		for(int i=0;i<testData.size();i++){
			Map<Double,Integer> mp = new TreeMap<Double,Integer>();// Dist -> Category
			Broadcast<Map<Double,Integer>> distanceMap = jsc.broadcast(mp);
			final int ind = i;
			train.foreach((ForeachFunction<Row>) trainRow ->{
				double dist = 0.0;
				for(int j=0;j<cols;j++)
				{
					double testVal = testData.get(ind).get(j);
					double trainVal = trainRow.getDouble(j);
					dist += (testVal - trainVal) * (testVal - trainVal);
				}
				dist = Math.sqrt(dist);
				distanceMap.getValue().put(dist,trainRow.getInt(cols));
			});
			int[] counts = new int[totalCategories + 1];
			mp = distanceMap.getValue();
			Iterator<Map.Entry<Double,Integer>> iterator = mp.entrySet().iterator();
	        int temp = k;
	        while (iterator.hasNext()) {
				Map.Entry entry = iterator.next();
				//System.out.println("Distance : " + entry.getKey() + " Category : " + entry.getValue());
				if(temp > 0)counts[(int) entry.getValue()]++;
				temp--;
			}
	        int category = getIndexOfLargest(counts);
	        System.out.println("The Category of Point Number " + cnt.getAndAdd(1) + " is " + "Category : " + category);
		}
	}
}
