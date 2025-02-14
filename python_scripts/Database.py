from mysql.connector import connect as sqlConnect
from mysql.connector import pooling
from mysql.connector import Error as sqlError
import logging


class DataBase():
    def __init__(self,host="127.0.0.1",database="phantom",user="root",password="",port=3306):
        self.host=host
        self.database=database
        self.user=user
        self.password=password
        self.port=port

        self.connect(self.host,self.database,self.user,self.password,self.port)

    def connect(self,host,database,user,password,port):
        try:
            while(True):
                try:
                    connection_pool=pooling.MySQLConnectionPool( pool_name="mysql_pool",
                                                                # pool_size=5,
                                                                port=port,
                                                                pool_reset_session=True,
                                                                host=host,
                                                                database=database,
                                                                user=user,
                                                                password=password
                                                                )
                    logging.info(f"Connection Pool Name - { connection_pool.pool_name}")
                    self.connection_object=connection_pool.get_connection() #get object from connection
                    self.cursordb=self.connection_object.cursor() #get cursor for survey of database
                    break
                except sqlError as err:
                        logging.error(f"error:{ err.msg}")  
                        logging.warn("Retrying to connect to the database")
        except KeyboardInterrupt:
            logging.critical("Exiting App")
        return

    def close(self):
        logging.info("close Database")
        self.cursordb.close()
        self.connection_object.close()         

    def read_query(self,query,verbose=False):
        while(True):
            try:
                if(verbose):
                    logging.info(f">>>Read query:{query}")
                self.cursordb.execute(query) #execute query
                queryresults = self.cursordb.fetchall() #and fetch results
                if(verbose):
                    print(queryresults)
                desc = self.cursordb.description
                column_names = [col[0] for col in desc]
                data = [dict(zip(column_names, row)) for row in queryresults]
                return data
            except sqlError as err:
                logging.error("failed run query! close database and try to reconnect database")
                logging.error(f"error:{ err.msg}")  
                self.close()
                self.connect(self.host,self.database,self.user,self.password,self.port)
    
    def write_query(self,query,verbose=False):
        while(True):
            try:
                if(verbose):
                    logging.info(f'>>>Write query:{query}')
                self.cursordb.execute(query) 
                self.connection_object.commit()  #commit after writing             
                break
            except sqlError as err:
                logging.error("failed run query! close database and try to reconnect database")
                logging.error(f"error:{ err.msg}")
                self.close()
                self.connect(self.host,self.database,self.user,self.password,self.port)              
         

