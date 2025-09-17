import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, 
  Image, 
  MessageSquare, 
  BarChart3, 
  Map,
  Play,
  Pause,
  Download,
  Share
} from 'lucide-react'
import { useEODataStore } from '../stores/eoDataStore'

const AnalysisPanel: React.FC = () => {
  const { 
    selectedTile,
    analysisResults,
    getAnalysisForTile,
    isProcessing,
    setIsProcessing
  } = useEODataStore()

  const [analysisType, setAnalysisType] = useState<'classification' | 'change_detection' | 'vqa' | 'caption'>('classification')
  const [query, setQuery] = useState('')

  const currentAnalysis = selectedTile ? getAnalysisForTile(selectedTile.id) : []

  const handleRunAnalysis = async () => {
    if (!selectedTile) return
    
    setIsProcessing(true)
    // Simulate analysis processing
    setTimeout(() => {
      setIsProcessing(false)
    }, 3000)
  }

  const analysisTypes = [
    {
      id: 'classification' as const,
      name: 'Land Cover Classification',
      description: 'Identify different land cover types in the satellite image',
      icon: Map
    },
    {
      id: 'change_detection' as const,
      name: 'Change Detection',
      description: 'Detect changes over time in the selected area',
      icon: BarChart3
    },
    {
      id: 'vqa' as const,
      name: 'Visual Q&A',
      description: 'Ask questions about what you see in the satellite image',
      icon: MessageSquare
    },
    {
      id: 'caption' as const,
      name: 'Image Captioning',
      description: 'Generate natural language description of the image',
      icon: Image
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">AI Analysis</h1>
          <p className="text-slate-400">Multimodal analysis with GPT-OSS integration</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Analysis Configuration */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6 mb-6">
              <h2 className="text-xl font-semibold text-white mb-4">Analysis Type</h2>
              
              <div className="space-y-3">
                {analysisTypes.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setAnalysisType(type.id)}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      analysisType === type.id
                        ? 'bg-blue-500/20 border-blue-500/50 text-white'
                        : 'bg-slate-700/30 border-slate-600/50 text-slate-300 hover:bg-slate-700/50'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <type.icon className="w-5 h-5 mt-0.5 flex-shrink-0" />
                      <div>
                        <div className="font-medium">{type.name}</div>
                        <div className="text-sm text-slate-400 mt-1">{type.description}</div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Query Input for VQA */}
            {analysisType === 'vqa' && (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Your Question</h3>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="What do you want to know about this satellite image?"
                  className="w-full h-24 px-3 py-2 bg-slate-700/50 border border-slate-600/50 rounded text-white placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
            )}

            {/* Selected Tile Info */}
            {selectedTile ? (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Selected Tile</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">ID:</span>
                    <span className="text-white font-mono text-xs">{selectedTile.id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Satellite:</span>
                    <span className="text-white">{selectedTile.satellite}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sensor:</span>
                    <span className="text-white">{selectedTile.sensor}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Date:</span>
                    <span className="text-white">
                      {new Date(selectedTile.captureDate).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Cloud Cover:</span>
                    <span className="text-white">{selectedTile.cloudCover.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6 mb-6 text-center">
                <Brain className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                <p className="text-slate-400">Select a tile from the globe to start analysis</p>
              </div>
            )}

            {/* Run Analysis Button */}
            <button
              onClick={handleRunAnalysis}
              disabled={!selectedTile || isProcessing}
              className="w-full px-4 py-3 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:from-slate-600 disabled:to-slate-600 text-white font-medium rounded-lg transition-all flex items-center justify-center gap-2"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Analysis
                </>
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white">Analysis Results</h2>
                {currentAnalysis.length > 0 && (
                  <div className="flex gap-2">
                    <button className="p-2 text-slate-400 hover:text-white transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                    <button className="p-2 text-slate-400 hover:text-white transition-colors">
                      <Share className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>

              {currentAnalysis.length > 0 ? (
                <div className="space-y-6">
                  {currentAnalysis.map((result) => (
                    <motion.div
                      key={result.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="border border-slate-600/50 rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-lg font-semibold text-white">
                          {result.classification}
                        </h3>
                        <span className="text-sm text-slate-400">
                          {new Date(result.timestamp).toLocaleString()}
                        </span>
                      </div>

                      <p className="text-slate-300 mb-4">{result.description}</p>

                      {/* Confidence */}
                      <div className="mb-4">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-slate-400">Confidence</span>
                          <span className="text-white">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full"
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Land Cover Analysis */}
                      {result.landCoverAnalysis && (
                        <div className="mb-4">
                          <h4 className="text-sm font-medium text-white mb-2">Land Cover Distribution</h4>
                          <div className="space-y-2">
                            {result.landCoverAnalysis.classes.map((cls) => (
                              <div key={cls.className} className="flex items-center gap-3">
                                <div
                                  className="w-4 h-4 rounded"
                                  style={{ backgroundColor: cls.color }}
                                />
                                <span className="text-sm text-slate-300 flex-1">
                                  {cls.className}
                                </span>
                                <span className="text-sm text-white">
                                  {cls.percentage.toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Environmental Indicators */}
                      {result.environmentalIndicators && (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-slate-700/30 rounded p-3">
                            <div className="text-xs text-slate-400 mb-1">NDVI</div>
                            <div className="text-lg font-semibold text-green-400">
                              {result.environmentalIndicators.vegetationIndex.toFixed(2)}
                            </div>
                          </div>
                          <div className="bg-slate-700/30 rounded p-3">
                            <div className="text-xs text-slate-400 mb-1">Water Index</div>
                            <div className="text-lg font-semibold text-blue-400">
                              {result.environmentalIndicators.waterIndex.toFixed(2)}
                            </div>
                          </div>
                          <div className="bg-slate-700/30 rounded p-3">
                            <div className="text-xs text-slate-400 mb-1">Urban Index</div>
                            <div className="text-lg font-semibold text-orange-400">
                              {result.environmentalIndicators.urbanIndex.toFixed(2)}
                            </div>
                          </div>
                          <div className="bg-slate-700/30 rounded p-3">
                            <div className="text-xs text-slate-400 mb-1">Surface Temp</div>
                            <div className="text-lg font-semibold text-red-400">
                              {result.environmentalIndicators.surfaceTemperature.toFixed(1)}°C
                            </div>
                          </div>
                        </div>
                      )}

                      {/* GPT Response */}
                      {result.multimodalAnalysis.gptResponse && (
                        <div className="mt-4 p-3 bg-slate-900/50 rounded border border-slate-600/30">
                          <h4 className="text-sm font-medium text-white mb-2">AI Analysis</h4>
                          <p className="text-sm text-slate-300">
                            {result.multimodalAnalysis.gptResponse}
                          </p>
                        </div>
                      )}

                      {/* Tags */}
                      {result.tags.length > 0 && (
                        <div className="mt-4">
                          <div className="flex flex-wrap gap-2">
                            {result.tags.map((tag) => (
                              <span
                                key={tag}
                                className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Brain className="w-16 h-16 text-slate-500 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">No analysis results yet</h3>
                  <p className="text-slate-400">
                    Select a satellite tile and run an analysis to see results here.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AnalysisPanel
