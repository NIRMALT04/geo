// Earth Observation data types

export interface EOTile {
  id: string
  latitude: number
  longitude: number
  captureDate: string
  satellite: string
  sensor: string
  resolution: number // meters per pixel
  spectralBands: string[]
  cloudCover: number // percentage
  analysisStatus: 'pending' | 'processing' | 'completed' | 'failed'
  imageUrl?: string
  thumbnailUrl?: string
  metadata: {
    path: string
    row: string
    scene: string
    processingLevel: string
    dataFormat: string
    fileSize: number
  }
}

export interface AnalysisResult {
  id: string
  tileId: string
  latitude: number
  longitude: number
  timestamp: string
  classification: string
  description: string
  confidence: number
  tags: string[]
  multimodalAnalysis: {
    visionEmbedding: number[]
    textEmbedding: number[]
    alignedEmbedding: number[]
    gptResponse: string
    reasoning: string
  }
  landCoverAnalysis?: {
    classes: LandCoverClass[]
    changeDetection?: ChangeDetection
  }
  environmentalIndicators?: {
    vegetationIndex: number
    waterIndex: number
    urbanIndex: number
    soilMoisture: number
    surfaceTemperature: number
  }
}

export interface LandCoverClass {
  className: string
  percentage: number
  confidence: number
  color: string
}

export interface ChangeDetection {
  timeRange: {
    start: string
    end: string
  }
  changes: Array<{
    type: 'deforestation' | 'urbanization' | 'flooding' | 'drought' | 'vegetation_growth'
    severity: 'low' | 'medium' | 'high'
    area: number // square kilometers
    confidence: number
  }>
}

export interface MultimodalQuery {
  id: string
  query: string
  type: 'text' | 'voice' | 'image'
  context?: {
    selectedTiles: string[]
    timeRange?: {
      start: string
      end: string
    }
    spatialBounds?: {
      north: number
      south: number
      east: number
      west: number
    }
  }
}

export interface MultimodalResponse {
  id: string
  queryId: string
  response: string
  confidence: number
  sources: Array<{
    type: 'eo_tile' | 'knowledge_base' | 'external_api'
    id: string
    relevance: number
  }>
  visualizations?: Array<{
    type: 'map_overlay' | 'chart' | 'timeline' | 'comparison'
    data: any
    config: any
  }>
  voiceResponse?: {
    audioUrl: string
    duration: number
    transcript: string
  }
}

export interface VoicePersona {
  id: string
  name: string
  description: string
  voiceModel: string
  personality: {
    expertise: string[]
    tone: 'professional' | 'casual' | 'educational' | 'enthusiastic'
    language: string
  }
  contextMemory: Array<{
    timestamp: string
    query: string
    response: string
    relevantTiles: string[]
  }>
}

export interface DataSource {
  id: string
  name: string
  type: 'isro' | 'nasa' | 'esa' | 'commercial' | 'open'
  apiEndpoint?: string
  authRequired: boolean
  supportedSensors: string[]
  coverageArea: {
    global: boolean
    regions?: string[]
  }
  updateFrequency: string
  dataFormats: string[]
  processingLevels: string[]
}

export interface TrainingDataset {
  id: string
  name: string
  description: string
  type: 'image_caption' | 'vqa' | 'land_cover' | 'change_detection' | 'multimodal'
  size: number // number of samples
  source: string
  version: string
  splits: {
    train: number
    validation: number
    test: number
  }
  annotations: {
    format: string
    categories: string[]
    quality: 'low' | 'medium' | 'high'
  }
  metadata: {
    createdAt: string
    updatedAt: string
    license: string
    citation: string
  }
}

export interface ModelConfig {
  id: string
  name: string
  type: 'vision_encoder' | 'language_model' | 'multimodal' | 'classifier'
  architecture: string
  parameters: {
    total: number
    trainable: number
  }
  performance: {
    accuracy?: number
    f1Score?: number
    bleuScore?: number
    perplexity?: number
  }
  training: {
    dataset: string
    epochs: number
    batchSize: number
    learningRate: number
    optimizer: string
  }
  deployment: {
    quantization?: '8bit' | '16bit' | 'fp32'
    device: 'cpu' | 'gpu' | 'tpu'
    memoryUsage: number // MB
    inferenceTime: number // ms
  }
}
