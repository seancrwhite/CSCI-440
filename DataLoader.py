import mysql.connector as dbc

class DataLoader:
    def __init__(self):
        self.db = dbc.connect(port = 3306,
                         user = "root",
                         passwd = "password",
                         db = "IMDB")
        self.cursor = self.db.cursor()

    # query database and load data into dictionary
    def fetch_data(self, question):
        query = "" # default empty query

        if question == 1:
            query = ("SELECT Gross_Revanue, Budget, Duration, Aspect_Ratio, Release_Year, Votes, IMDB_Score "
                     "FROM IMDB.EARNINGS NATURAL JOIN IMDB.SHOW NATURAL JOIN IMDB.SCORE "
                     "WHERE Gross_Revanue > 999999 AND Duration IS NOT NULL")
        elif question == 2:
            query = ("SELECT Show_Id, Title, Gross_Revanue, Name, Role "
                     "FROM IMDB.EARNINGS NATURAL JOIN IMDB.SHOW NATURAL JOIN "
                     "IMDB.WORKED_ON NATURAL JOIN IMDB.PERSON")
        elif question == 3:
            query = ("SELECT Duration, Aspect_Ratio, Release_Year, Budget, Gross_Revanue, Avg_Rating, Votes, IMDB_Score "
                     "FROM IMDB.SHOW NATURAL JOIN IMDB.EARNINGS NATURAL JOIN IMDB.SCORE WHERE Duration IS NOT NULL")
        elif question == 4:
            query = ("SELECT Title, Description, Genre "
                     "FROM IMDB.SHOW NATURAL JOIN IMDB.GENRE "
                     "WHERE Description IS NOT NULL")
        elif question == 5:
            query = ("SELECT Name, Birth_Year, Death_Year, Gross_Revanue "
                     "FROM IMDB.SHOW NATURAL JOIN IMDB.EARNINGS NATURAL JOIN "
                     "IMDB.WORKED_ON NATURAL JOIN IMDB.PERSON "
                     "WHERE Gross_Revanue > 0 AND Role='Actor' AND Birth_Year > 0")

        self.cursor.execute(query)
        data = self.cursor.fetchall()
        return data
