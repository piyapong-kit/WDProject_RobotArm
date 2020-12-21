#   https://tutorialslides.com/how-to-connect-and-create-database-on-mysql-phpmyadmin-using-python/.
# Prepare all function for implement to Machine learning.
import pymysql

def create_table():
    '''
    Create table and drop table if Exist,
    then insert ID 1 waiting for status.
    '''

    #database connection
    connection = pymysql.connect(host="localhost", user="root", passwd="", database="tictactoe")
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS TictactoeTest")
    # Query for creating table
    TictactoeTestTableSql = """CREATE TABLE TictactoeTest(
    ID INT(10) PRIMARY KEY AUTO_INCREMENT,
    EPISODE  INT(10),
    NSTATE VARCHAR(300),
    NVALUE VARCHAR(50),
    CHOOSE INT(1),
    PICK INT(1),
    STATE_NOW VARCHAR(30),
    ACTION INT(1),
    NOTE VARCHAR(200))
    """
    # To execute the SQL query 
    cursor.execute(TictactoeTestTableSql)

    # r
    db_table_tictactoe_state = "INSERT INTO TictactoeTest(ID) VALUES(1);" 
    cursor.execute(db_table_tictactoe_state)
    # To commit the changes
    connection.commit()
    connection.close()

def select_data():
    #database connection
    connection = pymysql.connect(host="localhost", user="root", passwd="", database="tictactoe")
    cursor = connection.cursor()

    # queries for retrievint all rows
    retrive = "Select ID='1' from TictactoeTest;"

    #executing the quires
    cursor.execute(retrive)
    rows = cursor.fetchall()
    for row in rows:
        print(row)


    #commiting the connection then closing it.
    connection.commit()
    connection.close()

def update_state():
    #database connection
    connection = pymysql.connect(host="localhost", user="root", passwd="", database="tictactoe")
    cursor = connection.cursor()

    updateState = "UPDATE  TictactoeTest SET EPISODE= '1',CHOOSE= '1'  WHERE ID = '1' ;"
    cursor.execute(updateState)


    #commiting the connection then closing it.
    connection.commit()
    connection.close()

def clear_state():
    #database connection
    connection = pymysql.connect(host="localhost", user="root", passwd="", database="tictactoe")
    cursor = connection.cursor()
    #delete data 
    deleteState = "DELETE FROM TictactoeTest WHERE ID = '1'; "
    cursor.execute(deleteState)
    #input new data to table
    db_table_tictactoe_state = "INSERT INTO TictactoeTest(ID) VALUES(1);" 
    cursor.execute(db_table_tictactoe_state)


    #commiting the connection then closing it.
    connection.commit()
    connection.close()

