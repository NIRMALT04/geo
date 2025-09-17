import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Search, 
  Filter, 
  Download, 
  Eye, 
  Calendar,
  MapPin,
  Satellite,
  Cloud
} from 'lucide-react'
import { useEODataStore } from '../stores/eoDataStore'

const DataExplorer: React.FC = () => {
  const { 
    eoTiles, 
    getFilteredTiles, 
    setSelectedTile,
    filterByCloudCover,
    setCloudCoverFilter,
    filterByDate,
    setDateFilter
  } = useEODataStore()

  const [searchTerm, setSearchTerm] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const filteredTiles = getFilteredTiles().filter(tile =>
    tile.satellite.toLowerCase().includes(searchTerm.toLowerCase()) ||
    tile.sensor.toLowerCase().includes(searchTerm.toLowerCase()) ||
    tile.id.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 bg-green-400/20'
      case 'processing': return 'text-yellow-400 bg-yellow-400/20'
      case 'failed': return 'text-red-400 bg-red-400/20'
      default: return 'text-slate-400 bg-slate-400/20'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Data Explorer</h1>
          <p className="text-slate-400">Browse and manage Earth Observation datasets</p>
        </div>

        {/* Search and Filters */}
        <div className="mb-6 space-y-4">
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                placeholder="Search by satellite, sensor, or ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-4 py-3 bg-slate-800/50 border border-slate-600/50 rounded-lg text-white hover:bg-slate-700/50 transition-colors flex items-center gap-2"
            >
              <Filter className="w-5 h-5" />
              Filters
            </button>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-slate-800/30 border border-slate-600/50 rounded-lg p-4"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Cloud Cover (max {filterByCloudCover}%)
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={filterByCloudCover}
                    onChange={(e) => setCloudCoverFilter(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-white mb-2">Start Date</label>
                  <input
                    type="date"
                    value={filterByDate?.start || ''}
                    onChange={(e) => setDateFilter(
                      e.target.value ? { 
                        start: e.target.value, 
                        end: filterByDate?.end || e.target.value 
                      } : null
                    )}
                    className="w-full px-3 py-2 bg-slate-700/50 border border-slate-600/50 rounded text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-white mb-2">End Date</label>
                  <input
                    type="date"
                    value={filterByDate?.end || ''}
                    onChange={(e) => setDateFilter(
                      e.target.value && filterByDate ? { 
                        start: filterByDate.start, 
                        end: e.target.value 
                      } : null
                    )}
                    className="w-full px-3 py-2 bg-slate-700/50 border border-slate-600/50 rounded text-white"
                  />
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Results */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTiles.map((tile) => (
            <motion.div
              key={tile.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.02 }}
              className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg overflow-hidden hover:border-slate-500/50 transition-all"
            >
              {/* Thumbnail */}
              <div className="aspect-video bg-slate-700/50 relative overflow-hidden">
                {tile.thumbnailUrl ? (
                  <img
                    src={tile.thumbnailUrl}
                    alt={`Satellite tile ${tile.id}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Satellite className="w-12 h-12 text-slate-500" />
                  </div>
                )}
                
                {/* Status Badge */}
                <div className={`absolute top-2 right-2 px-2 py-1 rounded text-xs font-medium ${getStatusColor(tile.analysisStatus)}`}>
                  {tile.analysisStatus}
                </div>
              </div>

              {/* Content */}
              <div className="p-4">
                <h3 className="font-semibold text-white mb-2 truncate">{tile.id}</h3>
                
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2 text-slate-300">
                    <Satellite className="w-4 h-4" />
                    <span>{tile.satellite} - {tile.sensor}</span>
                  </div>
                  
                  <div className="flex items-center gap-2 text-slate-300">
                    <Calendar className="w-4 h-4" />
                    <span>{new Date(tile.captureDate).toLocaleDateString()}</span>
                  </div>
                  
                  <div className="flex items-center gap-2 text-slate-300">
                    <MapPin className="w-4 h-4" />
                    <span>{tile.latitude.toFixed(2)}, {tile.longitude.toFixed(2)}</span>
                  </div>
                  
                  <div className="flex items-center gap-2 text-slate-300">
                    <Cloud className="w-4 h-4" />
                    <span>{tile.cloudCover.toFixed(1)}% cloud cover</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 mt-4">
                  <button
                    onClick={() => setSelectedTile(tile)}
                    className="flex-1 px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <Eye className="w-4 h-4" />
                    View
                  </button>
                  <button
                    disabled={!tile.imageUrl}
                    className="px-3 py-2 bg-slate-600 hover:bg-slate-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded text-sm transition-colors"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Empty State */}
        {filteredTiles.length === 0 && (
          <div className="text-center py-12">
            <Satellite className="w-16 h-16 text-slate-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No tiles found</h3>
            <p className="text-slate-400">Try adjusting your search criteria or filters.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default DataExplorer
