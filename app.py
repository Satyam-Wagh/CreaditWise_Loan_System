import joblib as jb
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# Load models
scaler = jb.load("scaler.pkl")
model = jb.load("logistic_regression_model.pkl")

class Myserver(BaseHTTPRequestHandler):

    def do_OPTIONS(self):

        self.send_response(200)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()


    def do_POST(self):

        try:

            print("\nPOST request received")
            print("Path:", self.path)

            if self.path == "/prediction":

                # Get request length
                content_length = int(
                    self.headers["Content-Length"]
                )

                # Read request body
                body = self.rfile.read(content_length)

                print("Body:", body)

                # JSON -> Python dictionary
                data = json.loads(body)

                # Get input
                inputs = data["input"]

                print("Input:", inputs)
                print("Number of inputs:", len(inputs))

                # Scale
                scale_value = scaler.transform([inputs])

                print("Scaling successful")

                # Predict
                result = model.predict(scale_value)

                print("Prediction:", result)

                # Response
                response = {
                    "result": result.tolist()
                }

                response = json.dumps(response).encode()

                # HTTP response
                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "application/json"
                )

                self.send_header(
                    "Access-Control-Allow-Origin",
                    "*"
                )

                self.send_header(
                    "Content-Length",
                    str(len(response))
                )

                self.end_headers()

                self.wfile.write(response)

                print("Response sent successfully")

        except Exception as e:

            print("\n========== ERROR ==========")
            print(type(e).__name__)
            print(e)
            print("===========================\n")

            response = {
                "error": str(e)
            }

            response = json.dumps(response).encode()

            self.send_response(500)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )

            self.send_header(
                "Content-Length",
                str(len(response))
            )

            self.end_headers()

            self.wfile.write(response)


server = HTTPServer(
    ("localhost", 8000),
    Myserver
)

print("Server running at http://localhost:8000")

server.serve_forever()