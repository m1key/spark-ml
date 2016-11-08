package me.m1key.sparkml;

import lombok.SneakyThrows;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.ClassificationModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.impurity.Entropy;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.model.TreeEnsembleModel;
import scala.Tuple2;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static com.google.common.io.Resources.getResource;
import static java.lang.String.format;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;

public class Sample {

    public static void main(String[] args) throws Exception {

        SparkConf conf = new SparkConf().setAppName("SparkTest").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<LabeledPoint> data = sc.textFile(getResourceUrl("impression_counts_by_date_and_dow_rated.csv"))
                .map(line -> line.split(","))
                .map(array -> {
                    List<Double> doubles = stream(array)
                            .map(Double::valueOf)
                            .collect(toList());

                    Double label = doubles.remove(doubles.size() - 1);
                    double[] features = doubles.stream().mapToDouble(d -> d).toArray();
                    return new LabeledPoint(label, Vectors.dense(features));
                });

        //scale
        StandardScalerModel scaler = new StandardScaler(true, true).fit(data.map(LabeledPoint::features).rdd());
        JavaRDD<LabeledPoint> scaled = data.map(lp -> new LabeledPoint(lp.label(), scaler.transform(lp.features())));

        // split
        JavaRDD<LabeledPoint>[] splits = scaled.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1].cache();

        TreeEnsembleModel[] treeEnsembleModels = {
                randomForestRegressor(training),
                randomForestClassifier(training)};

        ClassificationModel[] classificationModels = {
                logisticRegression(training),
                svm(training)};

        DecisionTreeModel[] decisionTreeModels = {
                decisionTree(training)};

        verify("16826,3,23189491", scaler, test, treeEnsembleModels, classificationModels, decisionTreeModels);

        Files.lines(Paths.get(ClassLoader.getSystemResource("int_impression_counts_by_date_and_dow.csv")
                .toURI())).forEach(s -> verify(s, scaler, test, treeEnsembleModels, classificationModels, decisionTreeModels));

        sc.stop();
    }

    private static void verify(String s, StandardScalerModel scaler, JavaRDD<LabeledPoint> test,
                               TreeEnsembleModel[] treeEnsembleModels,
                               ClassificationModel[] classificationModels,
                               DecisionTreeModel[] decisionTreeModels) {
        try {
            System.out.println("Running for: " + s);
            String[] values = s.split(",");
            double[] doubleValues = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                doubleValues[i] = Double.parseDouble(values[i]);
            }
            Vector vector = scaler.transform(Vectors.dense(doubleValues));

            Stream<TreeEnsembleModel> treeEnsembleModelStream = Arrays.stream(treeEnsembleModels);
            Stream<ClassificationModel> classificationModelStream = Arrays.stream(classificationModels);
            Stream<DecisionTreeModel> decisionTreeModelStream = Arrays.stream(decisionTreeModels);

            Collection<Tuple2<Double, String>> list = treeEnsembleModelStream.map(v -> {
                try {
                    return new Tuple2<>(v.predict(vector), accuracy(test, v::predict));
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            }).collect(toList());
            list.forEach(print());

            list = classificationModelStream.map(v -> {
                try {
                    return new Tuple2<>(v.predict(vector), accuracy(test, v::predict));
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            }).collect(toList());
            list.forEach(print());

            list = decisionTreeModelStream.map(v -> {
                try {
                    return new Tuple2<>(v.predict(vector), accuracy(test, v::predict));
                } catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            }).collect(toList());
            list.forEach(print());

//            System.out.println(decisionTree(training).toDebugString());
        } catch (NumberFormatException e) {
            // woops
        }
    }

    private static <T> Consumer<T> print() {
        return System.out::println;
    }

    private static DecisionTreeModel decisionTree(JavaRDD<LabeledPoint> data) {
        int maxTreeDepth = 30;
        DecisionTreeModel tree = DecisionTree.train(data.rdd(), Algo.Classification(), Entropy.instance(), maxTreeDepth);

//        System.out.println(tree.toDebugString());

        return tree;
    }

    private static ClassificationModel svm(JavaRDD<LabeledPoint> data) {
        int numIterations = 100;
        return SVMWithSGD.train(data.rdd(), numIterations);
    }

    private static ClassificationModel logisticRegression(JavaRDD<LabeledPoint> data) {
        int numIterations = 100;
        return LogisticRegressionWithSGD.train(data.rdd(), numIterations);
    }

    private static RandomForestModel randomForestClassifier(JavaRDD<LabeledPoint> data) {
        Integer numClasses = 20;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer numTrees = 30;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 4;
        Integer maxBins = 32;
        Integer seed = 12345;

        return RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
    }

    private static RandomForestModel randomForestRegressor(JavaRDD<LabeledPoint> data) {
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer numTrees = 20;
        String featureSubsetStrategy = "auto";
        String impurity = "variance";
        Integer maxDepth = 10;
        Integer maxBins = 32;
        Integer seed = 12345;

        return RandomForest.trainRegressor(data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
    }

    private static String accuracy(JavaRDD<LabeledPoint> data, Function<Vector, Double> predict) {
        return format("%.2f%%", (double)(data
                .map(point -> predict.call(point.features()) == point.label() ? 1 : 0)
                .reduce((a, b) -> a + b)) / data.count() * 100);
    }

    @SneakyThrows
    public static String getResourceUrl(String path) {
        return getResource(path).toURI().getPath();
    }
}
