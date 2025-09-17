import React, { Suspense, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { useEODataStore } from './stores/eoDataStore'
import { mockGeoJsonData, mockEOTiles } from './utils/mockData'
import ErrorBoundary from './components/ErrorBoundary'
import LoadingSpinner from './components/LoadingSpinner'

// Lazy load components for better performance
const Dashboard = React.lazy(() => import('./components/Dashboard'))
const Globe3D = React.lazy(() => import('./components/Globe3D'))
const DataExplorer = React.lazy(() => import('./components/DataExplorer'))
const AnalysisPanel = React.lazy(() => import('./components/AnalysisPanel'))
const VoiceInterface = React.lazy(() => import('./components/VoiceInterface'))

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

const App: React.FC = () => {
  const { setGeoJsonData, setEOTiles, setAvailablePersonas } = useEODataStore()

  useEffect(() => {
    // Initialize with mock data (replace with real API calls)
    setGeoJsonData(mockGeoJsonData)
    setEOTiles(mockEOTiles)
    
    // Initialize voice personas
    setAvailablePersonas([
      {
        id: 'geo-expert',
        name: 'Dr. Geo',
        description: 'Geospatial analysis expert specializing in satellite imagery interpretation',
        voiceModel: 'openai-tts',
        personality: {
          expertise: ['satellite imagery', 'land cover analysis', 'change detection'],
          tone: 'professional',
          language: 'en-US'
        },
        contextMemory: []
      },
      {
        id: 'eco-analyst',
        name: 'Eco',
        description: 'Environmental analyst focused on climate and ecosystem monitoring',
        voiceModel: 'elevenlabs',
        personality: {
          expertise: ['environmental monitoring', 'climate analysis', 'ecosystem health'],
          tone: 'educational',
          language: 'en-US'
        },
        contextMemory: []
      }
    ])
  }, [setGeoJsonData, setEOTiles, setAvailablePersonas])

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
            <Suspense fallback={<LoadingSpinner />}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/globe" element={<Globe3D />} />
                <Route path="/data" element={<DataExplorer />} />
                <Route path="/analysis" element={<AnalysisPanel />} />
                <Route path="/voice" element={<VoiceInterface />} />
              </Routes>
            </Suspense>
            
            {/* Global components */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1e293b',
                  color: '#f8fafc',
                  border: '1px solid #334155'
                },
              }}
            />
          </div>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App
