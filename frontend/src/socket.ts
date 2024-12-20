import { io } from "socket.io-client";
import { BACKEND_URL } from "./constants";

// export const socket = io("http://localhost:5000", {
//     transports: ['websocket']
// })

// export const socket = io("http://localhost:5000", {
//     autoConnect: false
// })

export const socket = io(BACKEND_URL, {
    autoConnect: false,
    transports: ['websocket']
})
console.log("socket", socket)
