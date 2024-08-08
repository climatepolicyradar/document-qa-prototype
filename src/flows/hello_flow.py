from prefect import flow 
from src.flows.utils import log_essentials, get_secret, send_slack_message

@flow 
def hello_flow():
    deets = log_essentials()
    print("Hello, world!")
    print(get_secret("test_secret"))
    send_slack_message(deets)
    send_slack_message("Hello, world!")
    
    
    
if __name__ == "__main__":
    hello_flow()