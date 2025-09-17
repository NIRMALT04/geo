import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { 
  Globe, 
  Database, 
  Brain, 
  Mic, 
  Satellite,
  Map,
  BarChart3,
  Settings,
  Play,
  Pause
} from 'lucide-react'
import Globe3D from './Globe3D'
import { useEODataStore } from '../stores/eoDataStore'

const Dashboard: React.FC = () => {
  const { 
    eoTiles, 
    analysisResults, 
    selectedTile, 
    setSelectedTile,
    viewMode,
    setViewMode,
    isProcessing,
    activePersona
  } = useEODataStore()

  const [isGlobeActive, setIsGlobeActive] = useState(true)

  const stats = {
    totalTiles: eoTiles.length,
    processedTiles: eoTiles.filter(tile => tile.analysisStatus === 'completed').length,
    analysisResults: analysisResults.length,
    activeAnalyses: eoTiles.filter(tile => tile.analysisStatus === 'processing').length
  }

  const navigationItems = [
    {
      title: '3D Globe',
      description: 'Interactive Earth visualization with satellite data',
      icon: Globe,
      path: '/globe',
      color: 'from-blue-500 to-cyan-500',
      stats: `${stats.totalTiles} tiles loaded`
    },
    {
      title: 'Data Explorer',
      description: 'Browse and manage EO satellite datasets',
      icon: Database,
      path: '/data',
      color: 'from-green-500 to-emerald-500',
      stats: `${stats.processedTiles} processed`
    },
    {
      title: 'AI Analysis',
      description: 'Multimodal analysis with GPT-OSS integration',
      icon: Brain,
      path: '/analysis',
      color: 'from-purple-500 to-pink-500',
      stats: `${stats.analysisResults} results`
    },
    {
      title: 'Voice Interface',
      description: 'Conversational AI for spatial queries',
      icon: Mic,
      path: '/voice',
      color: 'from-orange-500 to-red-500',
      stats: activePersona ? `${activePersona.name} active` : 'No persona'
    }
  ]

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <motion.div
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className="w-80 wireframe-panel border-r border-tech p-6 relative"
      >
        {/* Subtle scan lines overlay */}
        <div className="absolute inset-0 scan-lines opacity-30 pointer-events-none"></div>
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
              <Satellite className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">EO Analysis</h1>
              <p className="text-sm text-slate-400">Multimodal Earth Observation</p>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 gap-3 mb-6 relative z-10">
          <div className="wireframe-border bg-dark-surface rounded-lg p-3 tech-glow">
            <div className="text-2xl font-bold text-tech">{stats.totalTiles}</div>
            <div className="text-xs text-tech-dim">Total Tiles</div>
          </div>
          <div className="wireframe-border bg-dark-surface rounded-lg p-3 tech-glow">
            <div className="text-2xl font-bold text-green-400">{stats.processedTiles}</div>
            <div className="text-xs text-tech-dim">Processed</div>
          </div>
          <div className="wireframe-border bg-dark-surface rounded-lg p-3 tech-glow">
            <div className="text-2xl font-bold text-purple-400">{stats.analysisResults}</div>
            <div className="text-xs text-tech-dim">Analyses</div>
          </div>
          <div className="wireframe-border bg-dark-surface rounded-lg p-3 tech-glow">
            <div className="text-2xl font-bold text-orange-400">{stats.activeAnalyses}</div>
            <div className="text-xs text-tech-dim">Processing</div>
          </div>
        </div>

        {/* Navigation */}
        <div className="space-y-3 relative z-10">
          {navigationItems.map((item) => (
            <Link key={item.path} to={item.path}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="group relative overflow-hidden rounded-lg wireframe-border bg-dark-surface hover:bg-dark-panel transition-all duration-200 p-4 tech-glow"
              >
                <div className={`absolute inset-0 bg-gradient-to-r ${item.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
                
                <div className="relative flex items-start gap-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${item.color}`}>
                    <item.icon className="w-5 h-5 text-white" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-tech group-hover:text-cyan-300 transition-colors">
                      {item.title}
                    </h3>
                    <p className="text-sm text-tech-dim mb-1">
                      {item.description}
                    </p>
                    <div className="text-xs text-tech-dim">
                      {item.stats}
                    </div>
                  </div>
                </div>
              </motion.div>
            </Link>
          ))}
        </div>

        {/* View Mode Toggle */}
        <div className="mt-6 p-4 wireframe-border bg-dark-surface rounded-lg tech-glow relative z-10">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-tech">View Mode</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsGlobeActive(!isGlobeActive)}
                className="p-1 rounded text-tech-dim hover:text-tech transition-colors"
              >
                {isGlobeActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-1 bg-dark-panel rounded-lg p-1 border border-tech">
            {['3d', '2d', 'split'].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as '3d' | '2d' | 'split')}
                className={`px-3 py-1 text-xs font-medium rounded transition-all ${
                  viewMode === mode
                    ? 'bg-blue-500 text-white shadow-lg tech-glow'
                    : 'text-tech-dim hover:text-tech hover:bg-dark-surface'
                }`}
              >
                {mode.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Processing Status */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-3 hologram-border bg-dark-surface rounded-lg tech-glow relative z-10"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              <span className="text-sm text-blue-300">Processing analysis...</span>
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Main Content */}
      <div className="flex-1 relative">
        {/* Subtle grid background */}
        <div className="absolute inset-0 grid-overlay opacity-20"></div>
        
        {isGlobeActive && (
          <div className="absolute inset-0">
            <Globe3D
              onTileClick={setSelectedTile}
              selectedTile={selectedTile}
              analysisResults={analysisResults}
            />
          </div>
        )}

        {/* Overlay Controls */}
        <div className="absolute top-4 right-4 z-10">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex gap-2"
          >
            <button className="p-3 wireframe-border bg-dark-panel backdrop-blur-sm rounded-lg text-tech hover:text-white tech-glow transition-all">
              <Map className="w-5 h-5" />
            </button>
            <button className="p-3 wireframe-border bg-dark-panel backdrop-blur-sm rounded-lg text-tech hover:text-white tech-glow transition-all">
              <BarChart3 className="w-5 h-5" />
            </button>
            <button className="p-3 wireframe-border bg-dark-panel backdrop-blur-sm rounded-lg text-tech hover:text-white tech-glow transition-all">
              <Settings className="w-5 h-5" />
            </button>
          </motion.div>
        </div>

        {/* Selected Tile Info */}
        {selectedTile && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            className="absolute top-4 left-4 z-10 wireframe-panel backdrop-blur-sm rounded-lg p-4 max-w-sm tech-glow"
          >
            <h3 className="font-semibold text-tech mb-2">Selected Tile</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-tech-dim">Satellite:</span>
                <span className="text-tech">{selectedTile.satellite}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-tech-dim">Sensor:</span>
                <span className="text-tech">{selectedTile.sensor}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-tech-dim">Date:</span>
                <span className="text-tech">
                  {new Date(selectedTile.captureDate).toLocaleDateString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-tech-dim">Cloud Cover:</span>
                <span className="text-tech">{selectedTile.cloudCover}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-tech-dim">Status:</span>
                <span className={`font-medium ${
                  selectedTile.analysisStatus === 'completed' ? 'text-green-400' :
                  selectedTile.analysisStatus === 'processing' ? 'text-yellow-400' :
                  selectedTile.analysisStatus === 'failed' ? 'text-red-400' :
                  'text-tech-dim'
                }`}>
                  {selectedTile.analysisStatus}
                </span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Welcome Message */}
        {!isGlobeActive && (
          <div className="flex items-center justify-center h-full">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center max-w-md mx-auto p-8"
            >
              <div className="mb-6">
                <div className="w-24 h-24 mx-auto mb-4 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                  <Satellite className="w-12 h-12 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-white mb-2">
                  Welcome to EO Analysis
                </h2>
                <p className="text-slate-400">
                  Advanced multimodal Earth observation analysis with AI-powered insights
                </p>
              </div>
              
              <button
                onClick={() => setIsGlobeActive(true)}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-medium rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all transform hover:scale-105"
              >
                Launch 3D Globe
              </button>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
