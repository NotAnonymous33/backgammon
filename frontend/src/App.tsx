import './App.css'
import Board from './components/Board'
import Home from './components/Home'
import { BrowserRouter, Routes, Route } from 'react-router-dom'


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/game/:room_code" element={<Board />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
