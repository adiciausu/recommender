package com.mach;


import java.sql.Statement;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class RecommRepo
{
    private String dbUser = "adi";
    private String dbPass = "machteam%^&";
    private String endpoint = "jdbc:mysql://86.105.192.27:3306/teamdeals_network";

    public void save(Long userId, Long dealId, Float value) throws SQLException {

        Connection con = DriverManager.getConnection(endpoint, dbUser, dbPass);
        Statement st = con.createStatement();

        st.executeUpdate(String.format("INSERT IGNORE INTO recommendations (user_id, deal_id, value) VALUES (%d, %d, %f)", userId, dealId, value));

        con.close();
    }
}
