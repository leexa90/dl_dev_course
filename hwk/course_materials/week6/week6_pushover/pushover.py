# Pushover Helper
import http.client
import urllib

conn = http.client.HTTPSConnection("api.pushover.net:443")


class PushOver():
    ''' Push Over Class'''

    ''' Inputs is app and User'''

    def __init__(self, app_token, user_key):  #
        #self.app = app
        #self.user = user
        self.msg = ''
        self.sound = 'bugle'
        self.title = ''
        self.conn = http.client.HTTPSConnection("api.pushover.net:443")
        self.user_key = user_key  # ''
        self.app_token = app_token   # ''

    def message(self, msg):
        self.msg = msg

    def alert_sound(self, sound):
        self.sound = sound

    def send(self):
        if self.msg != '':
            print('sending')
            self.conn = http.client.HTTPSConnection("api.pushover.net:443")
            self.conn.request("POST", "/1/messages.json",
                              urllib.parse.urlencode({
                                  "token": self.app_token,
                                  "user": self.user_key,
                                  "message": self.msg,
                                  "title": self.title,
                                  "sound": self.sound,
                                  "priority": 1
                              }), {"Content-type": "application/x-www-form-urlencoded"})

    def show_message(self):
        print('Message:', self.msg)
        print('Title:', self.title)
        print('Sound:', self.sound)


#'''
class PushMessage():
    def __init__(self, message):
        """
        Creates a PushoverMessage object.
        """
        self.vars = {}
        self.vars['message'] = message

    def set(self, key, value):
        """
        Sets the value of a field "key" to the value of "value".
        """
        if value is not None:
            self.vars[key] = value

    def get(self):
        """
        Returns a dictionary with the values for the specified message.
        """
        return self.vars

    def user(self, user_token, user_device=None):
        """
        Sets a single user to be the recipient of this message with token "user_token" and device "user_device".
        """
        self.set('user', user_token)
        self.set('device', user_device)

    def __str__(self):
        return "PushoverMessage: " + str(self.vars)
#'''
