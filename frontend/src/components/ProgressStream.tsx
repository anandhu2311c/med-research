import { useNavigate } from 'react-router-dom'

interface ProgressStreamProps {
  progress: string[]
  reportId: string | null
}

export default function ProgressStream({ progress, reportId }: ProgressStreamProps) {
  const navigate = useNavigate()

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Processing...</h3>
      
      <div className="space-y-2 mb-4">
        {progress.map((step, index) => (
          <div key={index} className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="text-sm text-gray-600">{step}</span>
          </div>
        ))}
      </div>

      {reportId && (
        <div className="border-t pt-4">
          <p className="text-green-600 mb-3">✓ Analysis complete!</p>
          <button
            onClick={() => navigate(`/report/${reportId}`)}
            className="btn btn-primary"
          >
            View Report
          </button>
        </div>
      )}
    </div>
  )
}