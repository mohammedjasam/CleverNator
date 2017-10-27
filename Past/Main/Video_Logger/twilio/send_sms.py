#twilio integration file

# Download the twilio-python library from http://twilio.com/docs/libraries
# pip install twilio should work, otherwise use easy_install twilio
import sys
from twilio.rest import TwilioRestClient

# Find these values at https://twilio.com/user/account
account_sid = "-----"
auth_token = "-----"
client = TwilioRestClient(account_sid, auth_token)
#"+17654097348" shivam's # if needed
message = client.messages.create(to="-----", from_="-----",
                                     body="Testing send_sms.py at WildHacks 2016")