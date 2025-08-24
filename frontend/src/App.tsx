import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Report from './pages/Report'
import Layout from './components/Layout'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/report/:id" element={<Report />} />
      </Routes>
    </Layout>
  )
}

export default App