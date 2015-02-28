package com.mach;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.*;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;

public class Evaluator
{
    static final double EVALUATION_PERCENTAGE = 1;
    static final double TRAINING_PERCENTAGE = 0.9;


    public static void main(String[] args) throws IOException, TasteException {
        File userPreferencesFile = new File("data/orders-boolean.csv");
        DataModel dataModel = new GenericBooleanPrefDataModel(new FileDataModel(userPreferencesFile));
        DataModelBuilder modelBuilder = new DataModelBuilder() {
            @Override
            public DataModel buildDataModel(
                    FastByIDMap<PreferenceArray> trainingData) {
                return new GenericBooleanPrefDataModel(
                        GenericBooleanPrefDataModel.toDataMap(trainingData)); }
        };
        RecommenderBuilder rb;

        System.out.println("------Item Based Recommender---------------------------------------------------------------");
        rb = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) throws TasteException {
                return RecommenderFactory.createLogLikeItemSimilarityRecommender(model);
            }
        };
        eval(rb, modelBuilder, dataModel, "LogLike");
        rb = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) throws TasteException {
                return RecommenderFactory.createTanimotoItemSimilarityRecommender(model);
            }
        };
        eval(rb, modelBuilder, dataModel, "Tamimoto");


        int[] sizes = {2, 10, 100};
        for (final int size: sizes){
            System.out.println(String.format("------User Based Recommender (neghbourhoodsize: %d)------------------------------------------------", size));
            rb = new RecommenderBuilder() {
                public Recommender buildRecommender(DataModel model) throws TasteException {
                    return RecommenderFactory.createLogLikeUserSimilarityRecommenderWithSize(model, size);
                }
            };
            eval(rb, modelBuilder, dataModel, "LogLike");
            rb = new RecommenderBuilder() {
                public Recommender buildRecommender(DataModel model) throws TasteException {
                    return RecommenderFactory.createTanimotoUserSimilarityRecommenderWithSize(model, size);
                }
            };
            eval(rb, modelBuilder, dataModel, "Tamimoto");
        }

        double[] thresholds = {0.1, 0.5, 0.7};
        for (final double threshold: thresholds){
            System.out.println(String.format("------User Based Recommender (neghbourhoodsize: %f)------------------------------------------------", threshold));
            rb = new RecommenderBuilder() {
                public Recommender buildRecommender(DataModel model) throws TasteException {
                    return RecommenderFactory.createLogLikeUserSimilarityRecommenderWithThreshold(model, threshold);
                }
            };
            eval(rb, modelBuilder, dataModel, "LogLike");
            rb = new RecommenderBuilder() {
                public Recommender buildRecommender(DataModel model) throws TasteException {
                    return RecommenderFactory.createTanimotoUserSimilarityRecommenderWithThreshold(model, threshold);
                }
            };
            eval(rb, modelBuilder, dataModel, "Tamimoto");
        }

        rb = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) throws TasteException {
                return RecommenderFactory.createLogLikeUserSimilarityRecommenderWithSize(model, 2);
            }
        };
        irEval(rb, modelBuilder, dataModel);
    }

    private static void irEval(RecommenderBuilder rb, DataModelBuilder modelBuilder, DataModel dataModel) throws TasteException {
        RecommenderIRStatsEvaluator irStatsEvaluator =
                new GenericRecommenderIRStatsEvaluator ();
        IRStatistics stats = irStatsEvaluator.evaluate(rb, modelBuilder, dataModel, null, 5,
                GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 0.05);
        System.out.println(stats.getPrecision());
        System.out.println(stats.getRecall());
    }

    private static void eval(RecommenderBuilder recommenderBuilder, DataModelBuilder modelBuilder, DataModel dataModel, String algorithmName) throws TasteException {
        RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        double score = evaluator.evaluate(recommenderBuilder, modelBuilder, dataModel, TRAINING_PERCENTAGE, EVALUATION_PERCENTAGE);
        System.out.println(String.format("%s[%s]: %f", recommenderBuilder.buildRecommender(dataModel).getClass().getSimpleName(), algorithmName, score));
    }
}
