//Ronan Murphy: 15397831
//Assignment 3 : Part 2 
package assignment3b;

import java.util.Arrays;
import java.util.Vector;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

public class IMDB_SVM {
	
	public static void main(String[] args) {
		//create hasing function to enable transform from string to vector with value
		final HashingTF hf= new HashingTF(10000);
		
		//set property to hadoop home 
		System.setProperty("hadoop.home.dir", "C:/winutils");
		//initalize spark configuration and context 
		SparkConf sparkConf = new SparkConf().setAppName("IMDB_SVM")
				.setMaster("local[2]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		//create the path to read in the text file
		String path = "C:/Spark/sentiment labelled sentences/imdb_labelled.txt";
		
		//create a java pair RDD of integer
		//reads in the text file and maps the values split by a tab space
		//the 2nd element is added to a tuple of value intger either 0/1 which is the classification
		//the string title is added as the 2nd element in the pair
		JavaPairRDD<Integer, String> inputs = sc.textFile(path)
				.map(b -> b.split("\\t"))
				.mapToPair(s -> new Tuple2<Integer,String>(Integer.parseInt(s[1]), s[0]));
		//for the mlib functions to train and test the data must convert it to a labled point
		//and transform the string title value to vector values for each word
		//the SVM cant read in string values and they need to be transformed to strings in this way
		JavaRDD<LabeledPoint> data = inputs
				.map(a-> new LabeledPoint(a._1(), hf.transform(Arrays.asList(a._2().split(" ")))));

		// randomly Split initial RDD into two 60% for the training and 40% for the test
		JavaRDD<LabeledPoint>[] splits =
				data.randomSplit(new double[]{0.6, 0.4}, 11L);
		//create a labelled point for the training and test data so they can be broken into their split ratios
		JavaRDD<LabeledPoint> training = splits[0].cache();
		JavaRDD<LabeledPoint> test = splits[1];

		// Run training algorithm to build the SVM model with 10000 iterations
		SVMModel model = SVMWithSGD.train(training.rdd(), 10000);

		// Clear the prediction threshold so the model will return probabilities
		model.clearThreshold();
		
		
		
		// Based on the test set it will predict the results of the tests and compare the results to determine an accuracy 
		JavaPairRDD<Object, Object> predictionAndLabels = 
				test.mapToPair(p ->new Tuple2<>(model.predict(p.features()), p.label()));
		
		
		//print test values of title and classification for each movie
		//movie title is in vector form as transformed with hashing function
		//this makes it impossible to tell which title it is as the test set is randomly chosen
		//correctly predicted each title in test with a 70.37% accuracy
		predictionAndLabels.foreach(p -> {
	        System.out.println("Movie Title in Vector form: " +p._1() + " ; Predicted Classification : " + p._2());
	    }); 
		
		// Get evaluation metrics for binary classification so we can calculate the accuracy of the the algorithm
		BinaryClassificationMetrics metrics =
				new BinaryClassificationMetrics(predictionAndLabels.rdd());
		
		
		// Calculate the area under the curve and return the result - this displays the accuracy of the algorithm
		//area under the ROC curve depicts the ratio of TP/FP of results the closer the area under 
		//ROC curve is to 100 the more accurate the algorithm 
		System.out.println("Area under ROC = " + metrics.areaUnderROC());

	}


}
