import mysql.connector
import mysql
import requests
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')
server_code = {'live': 1,
               'uat': 2,
               'both': 3}
mydb = ''
url = 'https://igrade-authentication-testing.waycool.in/api/request/credentials'
headers = {'X-App-Name': 'qqLQbroLvYOhfwwViDXTCg==',
           'X-Api-Key': '8cwcsk0kkww488swgk8o84gk0ow4ooo8k0g888k8'}


def get_db_creds(server):
    """
    Function to get the server database credentials to connect with the database.
    :param server: Server name either Live or UAT.
    :return: Return the database Credentials.
    """

    try:
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
def db_connect(server):
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


# definition for query execution
def execute_insert_query(query):
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute(query)
    mydb.commit()
    return True


# definition for query execution
def execute_update_query(query):
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute(query)
    mydb.commit()
    return True


# definition for query execution
def execute_select_query(query):
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute(query)
    row = [x for x in mycursor]
    return row


def execute_delete_query(query):
    # print('[info] : QUERY EXECUTION : ',query)
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute(query)
    mydb.commit()
    return True


def get_pending_tickets_ids_list():
    query = "SELECT rd_id,rd_type from igrade_request_details where rd_status = '1' and rd_type = '1'"
    x = execute_select_query(query)
    return x


def get_image_details_quick_imgpro(ticket_id):
    query = "SELECT id,image_path,image_weight from igrade_request_ml_response_details where ticket_id = {}".format(
        str(ticket_id))
    x = execute_select_query(query)
    return x


def get_coin_diameter_quick_imgpro(ticket_id):
    query = "SELECT rd_coin_diameter from igrade_request_details where rd_id = {}".format(str(ticket_id))
    x = execute_select_query(query)
    return x


def update_quick_imgpro_data(ticket_id, image_wise_data, combined_overall_data):
    query = "update igrade_request_details set rd_image_wise_details='{}', rd_combined_overall_details='{}'," \
            "rd_status=3 where rd_id = {}".format(
        str(image_wise_data), str(combined_overall_data), str(ticket_id))
    x = execute_update_query(query)
    return x


def get_commodity_range_grade_details(commodity_id):
    query = "select cd_small_from,cd_small_to,cd_regular_from,cd_regular_to,cd_bold_from,cd_bold_to" \
            " from igrade_commodity_details where cd_id={}".format(
        str(commodity_id))
    x = execute_select_query(query)
    return x


def get_commodity_id(ticket_id):
    query = "select cd_id from igrade_request_details where rd_id={}".format(str(ticket_id))
    x = execute_select_query(query)
    return x


def get_ticketid_imageid_from_rdno(request_no):
    query = "select mlrd.id,mlrd.ticket_id from igrade_request_ml_response_details as mlrd" \
            " inner join igrade_request_details as rd on rd.rd_id = mlrd.ticket_id where rd.rd_no='{}'".format(
        str(request_no))
    x = execute_select_query(query)
    return x


def get_combined_overall_details(ticket_id):
    query = "select rd_id, rd_combined_overall_details from igrade_request_details where rd_id={}".format(
        str(ticket_id))
    x = execute_select_query(query)
    return x


def update_edit_grade_range_data(ticket_id, combined_overall_data):
    query = "update igrade_request_details set rd_combined_overall_details='{}',rd_status=3 where rd_id = {}".format(
        str(combined_overall_data), str(ticket_id))
    x = execute_update_query(query)
    return x
