import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import { getReport } from '../lib/api'

export default function Report() {
  const { id } = useParams<{ id: string }>()
  
  const { data: report, isLoading, error } = useQuery({
    queryKey: ['report', id],
    queryFn: () => getReport(id!),
    enabled: !!id
  })

  if (isLoading) {
    return <div className="text-center py-8">Loading report...</div>
  }

  if (error) {
    return <div className="text-red-600 text-center py-8">Error loading report</div>
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow p-8">
        <div className="mb-6">
          <button
            onClick={() => window.history.back()}
            className="text-blue-600 hover:text-blue-800 mb-4"
          >
            ← Back to Search
          </button>
          
          <div className="flex justify-between items-start">
            <h1 className="text-2xl font-bold">Literature Review Report</h1>
            <div className="flex gap-3">
              <button 
                onClick={() => window.print()}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
              >
                Print
              </button>
              <a
                href={`/api/report/${id}/download`}
                download
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 inline-block"
              >
                Download PDF
              </a>
            </div>
          </div>
        </div>

        <div className="prose prose-lg max-w-none">
          <ReactMarkdown 
            className="text-gray-800 leading-relaxed"
            components={{
              h1: ({children}) => <h1 className="text-3xl font-bold text-blue-900 mb-6">{children}</h1>,
              h2: ({children}) => <h2 className="text-2xl font-semibold text-blue-800 mt-8 mb-4">{children}</h2>,
              h3: ({children}) => <h3 className="text-xl font-medium text-blue-700 mt-6 mb-3">{children}</h3>,
              p: ({children}) => <p className="mb-4 text-gray-700 leading-7">{children}</p>,
              strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
              em: ({children}) => <em className="italic text-blue-600">{children}</em>
            }}
          >
            {report?.content || ''}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  )
}