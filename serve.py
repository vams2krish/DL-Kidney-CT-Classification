"""Run the demo with a production WSGI server, bound locally by default."""
import os
from waitress import serve
from app import app

if __name__ == '__main__':
    serve(app, host=os.environ.get('HOST', '127.0.0.1'),
          port=int(os.environ.get('PORT', '8080')), threads=4,
          max_request_body_size=8 * 1024 * 1024)
