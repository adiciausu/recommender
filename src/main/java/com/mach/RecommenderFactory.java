package com.mach;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * Created by adi on 11.02.15.
 */
public class RecommenderFactory
{
    // user recommenders
    public static Recommender createLogLikeUserSimilarityRecommenderWithSize(DataModel dataModel, int neigbourhoodSize) throws TasteException {
        UserSimilarity userSimilarity = new LogLikelihoodSimilarity(dataModel);
        UserNeighborhood userNeighborhood = new NearestNUserNeighborhood(neigbourhoodSize, userSimilarity, dataModel);

        return new GenericUserBasedRecommender(dataModel, userNeighborhood, userSimilarity);
    }

    public static Recommender createTanimotoUserSimilarityRecommenderWithSize(DataModel dataModel, int neigbourhoodSize) throws TasteException {
        UserSimilarity userSimilarity = new TanimotoCoefficientSimilarity(dataModel);
        UserNeighborhood userNeighborhood = new NearestNUserNeighborhood(neigbourhoodSize, userSimilarity, dataModel);

        return new GenericUserBasedRecommender(dataModel, userNeighborhood, userSimilarity);
    }

    public static Recommender createLogLikeUserSimilarityRecommenderWithThreshold(DataModel dataModel, double neighbourhoodThreshold) throws TasteException {
        UserSimilarity userSimilarity = new LogLikelihoodSimilarity(dataModel);
        UserNeighborhood userNeighborhood = new ThresholdUserNeighborhood(neighbourhoodThreshold, userSimilarity, dataModel);

        return new GenericUserBasedRecommender(dataModel, userNeighborhood, userSimilarity);
    }

    public static Recommender createTanimotoUserSimilarityRecommenderWithThreshold(DataModel dataModel, double neighbourhoodThreshold) throws TasteException {
        UserSimilarity userSimilarity = new TanimotoCoefficientSimilarity(dataModel);
        UserNeighborhood userNeighborhood = new ThresholdUserNeighborhood(neighbourhoodThreshold, userSimilarity, dataModel);

        return new GenericUserBasedRecommender(dataModel, userNeighborhood, userSimilarity);
    }


    // item recommenders
    public static Recommender createLogLikeItemSimilarityRecommender(DataModel dataModel) throws TasteException {
        ItemSimilarity itemSimilarity = new LogLikelihoodSimilarity(dataModel);

        return new GenericBooleanPrefItemBasedRecommender(dataModel, itemSimilarity);
    }

    public static Recommender createTanimotoItemSimilarityRecommender(DataModel dataModel) throws TasteException {
        ItemSimilarity itemSimilarity = new TanimotoCoefficientSimilarity(dataModel);

        return new GenericBooleanPrefItemBasedRecommender(dataModel, itemSimilarity);
    }
}
