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


        RecommenderBuilder tanimotoItemRecommenderBuilder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) throws TasteException {
                return RecommenderFactory.createTanimotoItemSimilarityRecommender(model);
            }
        };

        RecommenderBuilder logItemLikeRecommenderBuilder = new RecommenderBuilder() {
            public Recommender buildRecommender(DataModel model) throws TasteException {
                return RecommenderFactory.createLogLikeItemSimilarityRecommender(model);
            }
        };

        RecommenderEvaluator evaluator =
                new AverageAbsoluteDifferenceRecommenderEvaluator();

        System.out.println("------Item Based Recommender--------");
        double tanimotoItemScore = evaluator.evaluate(
                tanimotoItemRecommenderBuilder, modelBuilder, dataModel, 0.5, 1.0);
        System.out.println("TanimotoCoefficient: " + tanimotoItemScore);

        double logLikeItemScore = evaluator.evaluate(
                logItemLikeRecommenderBuilder, modelBuilder, dataModel, 0.5, 1.0);
        System.out.println("LogLike: : " + logLikeItemScore);

        System.out.println("------User Based Recommender--------");
        double tanimotoUserScore = evaluator.evaluate(
                tanimotoItemRecommenderBuilder, modelBuilder, dataModel, 0.5, 1.0);
        System.out.println("TanimotoCoefficient: " + tanimotoUserScore);

        double logLikeUserScore = evaluator.evaluate(
                logItemLikeRecommenderBuilder, modelBuilder, dataModel, 0.5, 1.0);
        System.out.println("LogLike: : " + logLikeUserScore);
    }
}
