import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { EOTile, AnalysisResult, MultimodalQuery, MultimodalResponse, VoicePersona } from '../types/eo'

interface EODataState {
  // Data state
  eoTiles: EOTile[]
  analysisResults: AnalysisResult[]
  geoJsonData: any
  selectedTile: EOTile | null
  selectedTiles: string[]
  
  // Query state
  currentQuery: MultimodalQuery | null
  queryHistory: MultimodalQuery[]
  responses: MultimodalResponse[]
  
  // Voice state
  activePersona: VoicePersona | null
  availablePersonas: VoicePersona[]
  isListening: boolean
  isProcessing: boolean
  
  // UI state
  viewMode: '3d' | '2d' | 'split'
  showAnalysisOverlay: boolean
  showBoundaries: boolean
  filterByDate: { start: string; end: string } | null
  filterByCloudCover: number
  
  // Actions
  setEOTiles: (tiles: EOTile[]) => void
  addEOTile: (tile: EOTile) => void
  updateEOTile: (id: string, updates: Partial<EOTile>) => void
  removeEOTile: (id: string) => void
  
  setAnalysisResults: (results: AnalysisResult[]) => void
  addAnalysisResult: (result: AnalysisResult) => void
  updateAnalysisResult: (id: string, updates: Partial<AnalysisResult>) => void
  
  setGeoJsonData: (data: any) => void
  setSelectedTile: (tile: EOTile | null) => void
  toggleTileSelection: (tileId: string) => void
  clearSelectedTiles: () => void
  
  setCurrentQuery: (query: MultimodalQuery | null) => void
  addQueryToHistory: (query: MultimodalQuery) => void
  addResponse: (response: MultimodalResponse) => void
  
  setActivePersona: (persona: VoicePersona | null) => void
  setAvailablePersonas: (personas: VoicePersona[]) => void
  setIsListening: (listening: boolean) => void
  setIsProcessing: (processing: boolean) => void
  
  setViewMode: (mode: '3d' | '2d' | 'split') => void
  toggleAnalysisOverlay: () => void
  toggleBoundaries: () => void
  setDateFilter: (filter: { start: string; end: string } | null) => void
  setCloudCoverFilter: (threshold: number) => void
  
  // Computed getters
  getFilteredTiles: () => EOTile[]
  getTileById: (id: string) => EOTile | undefined
  getAnalysisForTile: (tileId: string) => AnalysisResult[]
}

export const useEODataStore = create<EODataState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        eoTiles: [],
        analysisResults: [],
        geoJsonData: null,
        selectedTile: null,
        selectedTiles: [],
        
        currentQuery: null,
        queryHistory: [],
        responses: [],
        
        activePersona: null,
        availablePersonas: [],
        isListening: false,
        isProcessing: false,
        
        viewMode: '3d',
        showAnalysisOverlay: true,
        showBoundaries: true,
        filterByDate: null,
        filterByCloudCover: 100,
        
        // Actions
        setEOTiles: (tiles) => set({ eoTiles: tiles }),
        addEOTile: (tile) => set((state) => ({ 
          eoTiles: [...state.eoTiles, tile] 
        })),
        updateEOTile: (id, updates) => set((state) => ({
          eoTiles: state.eoTiles.map(tile => 
            tile.id === id ? { ...tile, ...updates } : tile
          )
        })),
        removeEOTile: (id) => set((state) => ({
          eoTiles: state.eoTiles.filter(tile => tile.id !== id),
          selectedTiles: state.selectedTiles.filter(tileId => tileId !== id),
          selectedTile: state.selectedTile?.id === id ? null : state.selectedTile
        })),
        
        setAnalysisResults: (results) => set({ analysisResults: results }),
        addAnalysisResult: (result) => set((state) => ({
          analysisResults: [...state.analysisResults, result]
        })),
        updateAnalysisResult: (id, updates) => set((state) => ({
          analysisResults: state.analysisResults.map(result =>
            result.id === id ? { ...result, ...updates } : result
          )
        })),
        
        setGeoJsonData: (data) => set({ geoJsonData: data }),
        setSelectedTile: (tile) => set({ selectedTile: tile }),
        toggleTileSelection: (tileId) => set((state) => ({
          selectedTiles: state.selectedTiles.includes(tileId)
            ? state.selectedTiles.filter(id => id !== tileId)
            : [...state.selectedTiles, tileId]
        })),
        clearSelectedTiles: () => set({ selectedTiles: [] }),
        
        setCurrentQuery: (query) => set({ currentQuery: query }),
        addQueryToHistory: (query) => set((state) => ({
          queryHistory: [query, ...state.queryHistory.slice(0, 49)] // Keep last 50
        })),
        addResponse: (response) => set((state) => ({
          responses: [response, ...state.responses]
        })),
        
        setActivePersona: (persona) => set({ activePersona: persona }),
        setAvailablePersonas: (personas) => set({ availablePersonas: personas }),
        setIsListening: (listening) => set({ isListening: listening }),
        setIsProcessing: (processing) => set({ isProcessing: processing }),
        
        setViewMode: (mode) => set({ viewMode: mode }),
        toggleAnalysisOverlay: () => set((state) => ({
          showAnalysisOverlay: !state.showAnalysisOverlay
        })),
        toggleBoundaries: () => set((state) => ({
          showBoundaries: !state.showBoundaries
        })),
        setDateFilter: (filter) => set({ filterByDate: filter }),
        setCloudCoverFilter: (threshold) => set({ filterByCloudCover: threshold }),
        
        // Computed getters
        getFilteredTiles: () => {
          const state = get()
          let filtered = state.eoTiles
          
          // Filter by cloud cover
          filtered = filtered.filter(tile => tile.cloudCover <= state.filterByCloudCover)
          
          // Filter by date
          if (state.filterByDate) {
            const startDate = new Date(state.filterByDate.start)
            const endDate = new Date(state.filterByDate.end)
            filtered = filtered.filter(tile => {
              const tileDate = new Date(tile.captureDate)
              return tileDate >= startDate && tileDate <= endDate
            })
          }
          
          return filtered
        },
        
        getTileById: (id) => get().eoTiles.find(tile => tile.id === id),
        
        getAnalysisForTile: (tileId) => 
          get().analysisResults.filter(result => result.tileId === tileId),
      }),
      {
        name: 'eo-data-store',
        partialize: (state) => ({
          eoTiles: state.eoTiles,
          analysisResults: state.analysisResults,
          queryHistory: state.queryHistory,
          viewMode: state.viewMode,
          showAnalysisOverlay: state.showAnalysisOverlay,
          showBoundaries: state.showBoundaries,
          filterByCloudCover: state.filterByCloudCover,
        }),
      }
    ),
    {
      name: 'eo-data-store',
    }
  )
)
