package com.mach;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.List;

public class App 
{
    public static void main(String[] args) throws IOException, TasteException, SQLException {
        RecommRepo recommRepo = new RecommRepo();
        // Create a data source from the CSV file
        File userPreferencesFile = new File("data/orders-boolean.csv");
        DataModel dataModel = new GenericBooleanPrefDataModel(new FileDataModel(userPreferencesFile));
        Recommender recom = RecommenderFactory.createLogLikeUserSimilarityRecommenderWithSize(dataModel, 2);

        LongPrimitiveIterator users = dataModel.getUserIDs();

        while (users.hasNext()) {
            Long currentUser = users.peek();
            System.out.println(String.format("Processing recommendations for userid: %s", currentUser));
            List<RecommendedItem> itemRecommendations = recom.recommend(users.next(), 20);
            for (RecommendedItem recommendedItem : itemRecommendations) {
                System.out.println(recommendedItem);
                recommRepo.save(currentUser, recommendedItem.getItemID(), recommendedItem.getValue());
            }
        }

        System.out.println("Finished");
    }
}
