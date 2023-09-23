from flask import Flask, request

app = Flask(__name__)

#global central_model
#global n_clients =  0

@app.route("/get-model", methods=["GET"])
def get_model():
    n_clients += 1
    pass

@app.route("/push_weights", method=["POST"])
def push_weights():
    n_clients -= 1
    pass

if __name__ = "__main__":
    app.run(debug=False)