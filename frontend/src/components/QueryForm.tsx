import { useState } from 'react'

interface QueryFormProps {
  onSubmit: (query: string, filters: any) => void
  loading: boolean
}

export default function QueryForm({ onSubmit, loading }: QueryFormProps) {
  const [query, setQuery] = useState('')
  const [sources, setSources] = useState(['arxiv', 'pubmed'])
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(query, { sources })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">
          Research Query
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your research question..."
          className="w-full p-3 border rounded-lg"
          rows={3}
          required
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Sources</label>
        <div className="space-x-4">
          {['arxiv', 'pubmed', 'ctgov'].map(source => (
            <label key={source} className="inline-flex items-center">
              <input
                type="checkbox"
                checked={sources.includes(source)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSources([...sources, source])
                  } else {
                    setSources(sources.filter(s => s !== source))
                  }
                }}
                className="mr-2"
              />
              {source.toUpperCase()}
            </label>
          ))}
        </div>
      </div>
      
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="btn btn-primary disabled:opacity-50"
      >
        {loading ? 'Searching...' : 'Search Literature'}
      </button>
    </form>
  )
}