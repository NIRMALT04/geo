import { EOTile, AnalysisResult, VoicePersona } from '../types/eo'

// Mock GeoJSON data for boundaries (simplified version of your existing data)
export const mockGeoJsonData = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {
        name: "India",
        iso_code: "IN"
      },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [68.0, 6.0], [97.0, 6.0], [97.0, 37.0], [68.0, 37.0], [68.0, 6.0]
        ]]
      }
    }
  ]
}

// Mock EO tiles data
export const mockEOTiles: EOTile[] = [
  {
    id: "ISRO_L1C_20240115_001",
    latitude: 28.6139,
    longitude: 77.2090,
    captureDate: "2024-01-15T10:30:00Z",
    satellite: "ResourceSat-2A",
    sensor: "LISS-IV",
    resolution: 5.8,
    spectralBands: ["B2", "B3", "B4", "B5"],
    cloudCover: 15.2,
    analysisStatus: "completed",
    imageUrl: "/mock-images/delhi-satellite.jpg",
    thumbnailUrl: "/mock-images/delhi-thumb.jpg",
    metadata: {
      path: "145",
      row: "040",
      scene: "ISRO_L1C_20240115_001_145040",
      processingLevel: "L1C",
      dataFormat: "GeoTIFF",
      fileSize: 52428800
    }
  },
  {
    id: "ISRO_L1C_20240116_002",
    latitude: 19.0760,
    longitude: 72.8777,
    captureDate: "2024-01-16T10:45:00Z",
    satellite: "ResourceSat-2A",
    sensor: "LISS-IV",
    resolution: 5.8,
    spectralBands: ["B2", "B3", "B4", "B5"],
    cloudCover: 8.7,
    analysisStatus: "completed",
    imageUrl: "/mock-images/mumbai-satellite.jpg",
    thumbnailUrl: "/mock-images/mumbai-thumb.jpg",
    metadata: {
      path: "148",
      row: "047",
      scene: "ISRO_L1C_20240116_002_148047",
      processingLevel: "L1C",
      dataFormat: "GeoTIFF",
      fileSize: 48234496
    }
  },
  {
    id: "ISRO_L1C_20240117_003",
    latitude: 13.0827,
    longitude: 80.2707,
    captureDate: "2024-01-17T11:00:00Z",
    satellite: "ResourceSat-2A",
    sensor: "LISS-IV",
    resolution: 5.8,
    spectralBands: ["B2", "B3", "B4", "B5"],
    cloudCover: 22.1,
    analysisStatus: "processing",
    imageUrl: "/mock-images/chennai-satellite.jpg",
    thumbnailUrl: "/mock-images/chennai-thumb.jpg",
    metadata: {
      path: "142",
      row: "051",
      scene: "ISRO_L1C_20240117_003_142051",
      processingLevel: "L1C",
      dataFormat: "GeoTIFF",
      fileSize: 51380224
    }
  },
  {
    id: "ISRO_L1C_20240118_004",
    latitude: 22.5726,
    longitude: 88.3639,
    captureDate: "2024-01-18T10:15:00Z",
    satellite: "ResourceSat-2A",
    sensor: "LISS-IV",
    resolution: 5.8,
    spectralBands: ["B2", "B3", "B4", "B5"],
    cloudCover: 5.3,
    analysisStatus: "completed",
    imageUrl: "/mock-images/kolkata-satellite.jpg",
    thumbnailUrl: "/mock-images/kolkata-thumb.jpg",
    metadata: {
      path: "138",
      row: "044",
      scene: "ISRO_L1C_20240118_004_138044",
      processingLevel: "L1C",
      dataFormat: "GeoTIFF",
      fileSize: 49876992
    }
  },
  {
    id: "ISRO_L1C_20240119_005",
    latitude: 17.3850,
    longitude: 78.4867,
    captureDate: "2024-01-19T10:30:00Z",
    satellite: "ResourceSat-2A",
    sensor: "LISS-IV",
    resolution: 5.8,
    spectralBands: ["B2", "B3", "B4", "B5"],
    cloudCover: 12.8,
    analysisStatus: "pending",
    imageUrl: "/mock-images/hyderabad-satellite.jpg",
    thumbnailUrl: "/mock-images/hyderabad-thumb.jpg",
    metadata: {
      path: "144",
      row: "049",
      scene: "ISRO_L1C_20240119_005_144049",
      processingLevel: "L1C",
      dataFormat: "GeoTIFF",
      fileSize: 47562752
    }
  }
]

// Mock analysis results
export const mockAnalysisResults: AnalysisResult[] = [
  {
    id: "analysis_001",
    tileId: "ISRO_L1C_20240115_001",
    latitude: 28.6139,
    longitude: 77.2090,
    timestamp: "2024-01-15T12:00:00Z",
    classification: "Urban Metropolitan Area",
    description: "Dense urban environment with mixed residential, commercial, and industrial areas. High built-up density with significant infrastructure development.",
    confidence: 0.89,
    tags: ["urban", "metropolitan", "high-density", "infrastructure"],
    multimodalAnalysis: {
      visionEmbedding: new Array(512).fill(0).map(() => Math.random()),
      textEmbedding: new Array(768).fill(0).map(() => Math.random()),
      alignedEmbedding: new Array(768).fill(0).map(() => Math.random()),
      gptResponse: "This satellite image shows the Delhi metropolitan area, characterized by dense urban development with a mix of residential and commercial structures. The image reveals extensive infrastructure including major roadways, built-up areas, and urban planning patterns typical of a major Indian city.",
      reasoning: "Analysis based on spectral signatures indicating high built-up index, road network density, and urban morphology patterns consistent with metropolitan areas."
    },
    landCoverAnalysis: {
      classes: [
        { className: "Built-up", percentage: 65.4, confidence: 0.92, color: "#ff4500" },
        { className: "Vegetation", percentage: 18.2, confidence: 0.85, color: "#228b22" },
        { className: "Water", percentage: 8.1, confidence: 0.78, color: "#0077be" },
        { className: "Bare Soil", percentage: 8.3, confidence: 0.71, color: "#daa520" }
      ]
    },
    environmentalIndicators: {
      vegetationIndex: 0.32,
      waterIndex: 0.15,
      urbanIndex: 0.78,
      soilMoisture: 0.25,
      surfaceTemperature: 28.5
    }
  },
  {
    id: "analysis_002",
    tileId: "ISRO_L1C_20240116_002",
    latitude: 19.0760,
    longitude: 72.8777,
    timestamp: "2024-01-16T12:30:00Z",
    classification: "Coastal Urban Area",
    description: "Coastal metropolitan region with urban development adjacent to water bodies. Mix of high-rise buildings, port infrastructure, and residential areas.",
    confidence: 0.91,
    tags: ["coastal", "urban", "port", "mixed-development"],
    multimodalAnalysis: {
      visionEmbedding: new Array(512).fill(0).map(() => Math.random()),
      textEmbedding: new Array(768).fill(0).map(() => Math.random()),
      alignedEmbedding: new Array(768).fill(0).map(() => Math.random()),
      gptResponse: "The satellite imagery captures Mumbai's coastal urban landscape, showing the characteristic blend of dense urban development and maritime infrastructure. The image reveals the city's unique geography with urban areas extending along the coastline.",
      reasoning: "Coastal urban classification based on proximity to water bodies, urban density patterns, and characteristic coastal development signatures."
    },
    landCoverAnalysis: {
      classes: [
        { className: "Built-up", percentage: 58.7, confidence: 0.89, color: "#ff4500" },
        { className: "Water", percentage: 25.3, confidence: 0.94, color: "#0077be" },
        { className: "Vegetation", percentage: 12.1, confidence: 0.82, color: "#228b22" },
        { className: "Bare Soil", percentage: 3.9, confidence: 0.68, color: "#daa520" }
      ]
    },
    environmentalIndicators: {
      vegetationIndex: 0.28,
      waterIndex: 0.45,
      urbanIndex: 0.72,
      soilMoisture: 0.35,
      surfaceTemperature: 26.8
    }
  }
]

// Mock voice personas
export const mockVoicePersonas: VoicePersona[] = [
  {
    id: 'geo-expert',
    name: 'Dr. Geo',
    description: 'Geospatial analysis expert specializing in satellite imagery interpretation',
    voiceModel: 'openai-tts',
    personality: {
      expertise: ['satellite imagery', 'land cover analysis', 'change detection', 'geospatial analysis'],
      tone: 'professional',
      language: 'en-US'
    },
    contextMemory: [
      {
        timestamp: "2024-01-15T14:30:00Z",
        query: "What can you tell me about the urban development in this area?",
        response: "Based on the satellite imagery analysis, this area shows significant urban development with high built-up density. The spectral signatures indicate mixed residential and commercial zones with well-developed infrastructure.",
        relevantTiles: ["ISRO_L1C_20240115_001"]
      }
    ]
  },
  {
    id: 'eco-analyst',
    name: 'Eco',
    description: 'Environmental analyst focused on climate and ecosystem monitoring',
    voiceModel: 'elevenlabs',
    personality: {
      expertise: ['environmental monitoring', 'climate analysis', 'ecosystem health', 'vegetation indices'],
      tone: 'educational',
      language: 'en-US'
    },
    contextMemory: [
      {
        timestamp: "2024-01-16T15:45:00Z",
        query: "How is the vegetation health in this coastal region?",
        response: "The vegetation health in this coastal area shows moderate levels with an NDVI of 0.28. The proximity to water bodies helps maintain soil moisture, but urban pressure is limiting green space expansion.",
        relevantTiles: ["ISRO_L1C_20240116_002"]
      }
    ]
  },
  {
    id: 'change-detector',
    name: 'Delta',
    description: 'Change detection specialist for temporal analysis of satellite data',
    voiceModel: 'coqui-tts',
    personality: {
      expertise: ['change detection', 'temporal analysis', 'land use change', 'deforestation monitoring'],
      tone: 'analytical',
      language: 'en-US'
    },
    contextMemory: []
  }
]

// Utility function to generate random coordinates within India bounds
export const generateRandomIndianCoordinate = (): [number, number] => {
  const minLat = 6.0
  const maxLat = 37.0
  const minLon = 68.0
  const maxLon = 97.0
  
  const lat = minLat + Math.random() * (maxLat - minLat)
  const lon = minLon + Math.random() * (maxLon - minLon)
  
  return [lat, lon]
}

// Utility function to generate mock satellite data
export const generateMockSatelliteData = (count: number = 10): EOTile[] => {
  const satellites = ["ResourceSat-2A", "ResourceSat-2", "Cartosat-2S", "Cartosat-2"]
  const sensors = ["LISS-IV", "LISS-III", "AWiFS", "PAN"]
  const statuses: Array<"pending" | "processing" | "completed" | "failed"> = ["pending", "processing", "completed", "completed", "completed"] // Weight towards completed
  
  return Array.from({ length: count }, (_, i) => {
    const [lat, lon] = generateRandomIndianCoordinate()
    const satellite = satellites[Math.floor(Math.random() * satellites.length)]
    const sensor = sensors[Math.floor(Math.random() * sensors.length)]
    const status = statuses[Math.floor(Math.random() * statuses.length)]
    
    return {
      id: `MOCK_${Date.now()}_${i.toString().padStart(3, '0')}`,
      latitude: lat,
      longitude: lon,
      captureDate: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(), // Random date within last 30 days
      satellite,
      sensor,
      resolution: sensor === "LISS-IV" ? 5.8 : sensor === "LISS-III" ? 23.5 : sensor === "AWiFS" ? 56.0 : 2.5,
      spectralBands: sensor === "PAN" ? ["PAN"] : ["B2", "B3", "B4", "B5"],
      cloudCover: Math.random() * 50,
      analysisStatus: status,
      metadata: {
        path: Math.floor(Math.random() * 200 + 100).toString(),
        row: Math.floor(Math.random() * 100 + 20).toString().padStart(3, '0'),
        scene: `MOCK_L1C_${new Date().toISOString().split('T')[0].replace(/-/g, '')}_${i.toString().padStart(3, '0')}`,
        processingLevel: "L1C",
        dataFormat: "GeoTIFF",
        fileSize: Math.floor(Math.random() * 100 + 20) * 1024 * 1024 // 20-120 MB
      }
    }
  })
}
