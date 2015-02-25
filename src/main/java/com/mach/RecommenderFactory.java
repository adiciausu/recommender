package com.mach;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
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
    public static Recommender createTanimotoSimilarityRecommender(DataModel dataModel) throws TasteException {
        UserSimilarity userSimilarity = new LogLikelihoodSimilarity(dataModel);
        UserNeighborhood userNeighborhood = new NearestNUserNeighborhood(1000, userSimilarity, dataModel);

        // Create a generic user based recommender with the dataModel, the userNeighborhood and the userSimilarity
        return new GenericUserBasedRecommender(dataModel, userNeighborhood, userSimilarity);
    }

    public static Recommender createLogLikeUserSimilarityRecommender(DataModel dataModel) throws TasteException {
        UserSimilarity userSimilarity = new TanimotoCoefficientSimilarity(dataModel);
        UserNeighborhood userNeighborhood = new NearestNUserNeighborhood(1000, userSimilarity, dataModel);

        // Create a generic user based recommender with the dataModel, the userNeighborhood and the userSimilarity
        return new GenericUserBasedRecommender(dataModel, userNeighborhood, userSimilarity);
    }

    public static Recommender createLogLikeItemSimilarityRecommender(DataModel dataModel) throws TasteException {
        ItemSimilarity itemSimilarity = new LogLikelihoodSimilarity(dataModel);

        return new GenericBooleanPrefItemBasedRecommender(dataModel, itemSimilarity);
    }

    public static Recommender createTanimotoItemSimilarityRecommender(DataModel dataModel) throws TasteException {
        ItemSimilarity itemSimilarity = new TanimotoCoefficientSimilarity(dataModel);

        return new GenericBooleanPrefItemBasedRecommender(dataModel, itemSimilarity);
    }
}
