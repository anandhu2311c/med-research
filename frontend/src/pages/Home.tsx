import { useState } from 'react'
import QueryForm from '../components/QueryForm'
import ProgressStream from '../components/ProgressStream'
import { streamQuery } from '../lib/api'

export default function Home() {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<string[]>([])
  const [reportId, setReportId] = useState<string | null>(null)

  const handleQuery = async (query: string, filters: any) => {
    setLoading(true)
    setProgress([])
    setReportId(null)

    try {
      const cleanup = await streamQuery(
        { query, ...filters },
        (event) => {
          const data = JSON.parse(event.data)
          if (data.type === 'progress') {
            setProgress(prev => [...prev, data.message])
          } else if (data.type === 'complete') {
            setReportId(data.reportId)
            setLoading(false)
          } else if (data.type === 'error') {
            console.error('Query error:', data.message)
            setLoading(false)
          }
        }
      )

      // Cleanup on unmount or new query
      return cleanup
    } catch (error) {
      console.error('Query failed:', error)
      setProgress(prev => [...prev, `Error: ${error.message}`])
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold mb-4">
          AI-Powered Literature Search
        </h2>
        <p className="text-gray-600">
          Search across PubMed, arXiv, and ClinicalTrials.gov with 
          AI-powered summarization and evidence extraction.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-8">
        <QueryForm onSubmit={handleQuery} loading={loading} />
      </div>

      {(loading || progress.length > 0) && (
        <ProgressStream progress={progress} reportId={reportId} />
      )}
    </div>
  )
}