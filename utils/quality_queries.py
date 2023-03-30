# Import required packages
import mysql.connector
import mysql
import requests
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')

server_code = {'live': 1,
               'uat': 2,
               'both': 3}
mydb = ''


def get_db_creds(server):
    """
    Function to get the server database credentials to connect with the database.
    :param server: Server name either Live or UAT.
    :return: Return the database Credentials.
    """

    try:
        url = 'https://igrade-authentication-testing.waycool.in/api/request/credentials'
        headers = {'X-App-Name': 'qqLQbroLvYOhfwwViDXTCg==',
                   'X-Api-Key': '8cwcsk0kkww488swgk8o84gk0ow4ooo8k0g888k8'}
        credential = requests.post(url, headers=headers, data={'server_type': server_code[server]})
        credential = credential.json()

        if credential['status']:
            credential = credential['data'][server]
            host = credential['host']
            database = credential['database']
            user = credential['user']
            password = credential['password']

            return host, database, user, password
        else:
            logging.error(f"Give proper db name.. either live or uat...!!!")
            return False

    except Exception as e:
        logging.error(f'{e}')
        return False


# Definition for connecting DataBase
def db_connect_quality(server):
    """
    Function to connect with the environment database.
    :param server: Server name either Live or UAT.
    :return: Return the status database connection and its status.
    """

    global mydb
    try:
        # Get the database credentials
        host_cred, database_cred, user_cred, password_cred = get_db_creds(server)

        # Connect to the database
        mydb = mysql.connector.connect(
            host=host_cred,
            database=database_cred,
            user=user_cred,
            password=password_cred)

        logging.info(f'Database Connection Successful...!!!')
        return mydb, True

    except Exception as e:
        logging.error(f'Database Connection Failed... {e}')
        return False, False


# Definition for select query execution
def execute_select_query(query):
    my_cursor = mydb.cursor(buffered=True)
    my_cursor.execute(query)
    print('my_cursor', my_cursor)
    row = [x for x in my_cursor]
    return row


# definition for update query execution
def execute_update_query(query):
    my_cursor = mydb.cursor(buffered=True)
    my_cursor.execute(query)
    mydb.commit()
    return True


# Get the ticket details
def get_ticketid_details_quality(ticket_id):
    query = "SELECT rd_id,cad_id,cd_id,vd_id,svd_id,rd_image_wise_details,rd_combined_overall_details," \
            "rd_coin_diameter from igrade_request_details where rd_id = {}".format(str(ticket_id))
    x = execute_select_query(query)
    return x


# Get the image details
def get_image_details_quality(ticket_id):
    query = "SELECT id,image_path,image_weight from igrade_request_ml_response_details where ticket_id = {}".format(
        str(ticket_id))
    x = execute_select_query(query)
    return x


# Update the ticket with output results
def update_quality_data(ticket_id, image_wise_data, combined_overall_data):
    query = "update igrade_request_details set rd_image_wise_details='{}', rd_combined_overall_details='{}'" \
            ",rd_quality_status=3 where rd_id = {}".format(str(image_wise_data),
                                                           str(combined_overall_data),
                                                           str(ticket_id))
    x = execute_update_query(query)
    return x
