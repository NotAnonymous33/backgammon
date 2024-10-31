import { io } from "socket.io-client";

// export const socket = io("http://localhost:5000", {
//     transports: ['websocket']
// })

// export const socket = io("http://localhost:5000", {
//     autoConnect: false
// })

export const socket = io("http://localhost:5000", {
    autoConnect: false,
    transports: ['websocket']
})
console.log("socket", socket)
