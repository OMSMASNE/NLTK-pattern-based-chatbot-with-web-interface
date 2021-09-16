# Copyright (c) 2021 OM SANTOSHKUMAR MASNE.
# All Rights Reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for license information.

from flask import Flask, render_template
from flask_socketio import SocketIO, emit, socketio

from chatbot import get_answer

app = Flask(__name__)

socketio = SocketIO(app)

@app.route("/")
def homepage():
    return render_template("homepage.html")

@socketio.on("new_message")
def send_reply(data):
    user_message = str(data["user_message"])

    print("\n\n")
    print("USER MESSAGE: " + user_message)
    print("\n\n")

    answer = get_answer(user_message)
    
    print("\n\n")
    print("BOT ANSWER: " + answer)
    print("\n\n")

    emit("answer", {"answer": answer}, broadcast=True)
