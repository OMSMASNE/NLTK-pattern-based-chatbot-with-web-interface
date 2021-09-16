/*
Copyright (c) 2021 OM SANTOSHKUMAR MASNE.
All Rights Reserved.
Licensed under the MIT license.
See LICENSE file in the project root for license information.
*/

var socket = io.connect(location.protocol + "//" + document.domain + ":" + location.port);

document.addEventListener("DOMContentLoaded", startup);

function startup()
{
    document.getElementById("send-button").addEventListener('click', send_message);
    socket.on("answer", data => {
        console.log("Answer recieved:");
        console.log(data.answer);

        let message_box = document.createElement("li");
        message_box.setAttribute("class", "msg bot-msg");
        message_box.innerText = data.answer;
    
        document.getElementById("message-list").append(message_box);

        message_box.scrollIntoView();
    });
}

function send_message()
{
    let text = document.getElementById("user-input").value;
    console.log("User message: " + text);

    if(text === "")
        return;

    socket.emit("new_message", {"user_message": text});

    document.getElementById("user-input").value = "";

    let message_box = document.createElement("li");
    message_box.setAttribute("class", "msg user-msg");
    message_box.innerText = text;

    document.getElementById("message-list").append(message_box);
}
