"""### https://stackoverflow.com/questions/40460846/using-flask-inside-class

from sanic import Sanic, response


class EndpointAction(object):

    def __init__(self):
        #self.action = action
        #self.response = response(status=200, headers={})
        self.id = '123456'
        self.empty_response = {  "version": "4.5.6",
                               "flags": {},
                               "shapes": [],
                               "imagePath": "",
                               "imageData": None,
                               "imageHeight": 0,
                               "imageWidth": 0,
                               "shapes": [],
                               "message": "",
                              }

    def __call__(self
                ):
        #self.action()
        print(self.id)
        return response.json(self.empty_response)



class SanicAppWrapper(object):
    app = None

    def __init__(self, name):
        self.app = Sanic(name)

    def run(self):
        self.app.run(host = '192.168.211.14', port = 5000)

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.app.add_route(EndpointAction(), endpoint, methods=['POST'])


def action():
    # Execute anything
    pass

a = SanicAppWrapper('wrap')
epa = EndpointAction
a.add_endpoint(endpoint='/sarcopenia', endpoint_name='sarcopenia')#, handler=action)
a.run()
"""


###https://stackoverflow.com/questions/49528824/flask-access-request-object-from-method-in-class
#from flask import Flask, request
from sanic import Sanic, request

class MyAPIClass(object):

    def __init__(self, address='192.168.211.14', port=5000):
        self.address = address
        self.port = port
        self.app = Sanic(__name__)
        self.app.add_route(self.create
                           , "/sarcopenia", methods=['POST'])

    def create(self, request):
        root = request.form.get('root')
        home = request.form.get('home')
        print(request.json)
        size = int(request.form.get('size', 0))
        print('CREATE {} -> {} -> {} -> {}'.format(id, root, home, size))

    def start(self):
        self.app.run(host=self.address, port=self.port)
        
        
if __name__ == "__main__":
    api = MyAPIClass()
    api.start()