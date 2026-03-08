import { useState } from 'react'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-950 text-white">
      <h1 className="text-4xl font-bold mb-8">Vite + React + Tailwind</h1>
      <div className="flex flex-col items-center gap-4">
        <button
          className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-medium transition-colors"
          onClick={() => setCount((c) => c + 1)}
        >
          count is {count}
        </button>
        <p className="text-gray-400">
          Edit <code className="text-gray-300">src/App.tsx</code> and save to test HMR
        </p>
      </div>
    </div>
  )
}

export default App
