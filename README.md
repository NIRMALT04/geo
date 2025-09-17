# Multimodal Earth Observation Analysis System

A comprehensive AI-powered platform for analyzing satellite imagery with GPT-OSS integration, featuring multimodal understanding, voice-enabled personas, and interactive 3D visualization.

## 🌍 Overview

This system integrates cutting-edge vision encoders with GPT-OSS embeddings to provide advanced Earth Observation (EO) analysis capabilities. It processes ISRO satellite data and provides multimodal AI insights through an intuitive 3D globe interface.

## ✨ Key Features

### 🤖 AI-Powered Analysis
- **Vision Encoder Integration**: CLIP and BLIP models for vision-language understanding
- **GPT-OSS Alignment**: Trainable projection layers for seamless multimodal fusion
- **Multimodal Training**: Support for image captioning, VQA, and land cover classification
- **Parameter-Efficient Fine-tuning**: LoRA and adapter methods for resource optimization

### 🛰️ Earth Observation Pipeline
- **ISRO Data Integration**: Automated downloading and processing of Bhuvan/NRSC datasets
- **Multi-sensor Support**: ResourceSat, Cartosat, IRS-P6 satellite data
- **Geospatial Processing**: GDAL-based preprocessing with coordinate transformation
- **Quality Control**: Cloud cover filtering, geometric accuracy validation

### 🌐 Interactive Visualization
- **3D Globe Interface**: React Three Fiber powered Earth visualization
- **Real-time Updates**: WebSocket-based live data streaming
- **Drill-down Analytics**: Clickable geographic points with detailed analysis
- **Multi-view Support**: 3D, 2D, and split-screen modes

### 🎤 Voice-Enabled Personas
- **Conversational AI**: Natural language queries about satellite data
- **Multiple Personas**: Specialized experts (Geo-analyst, Eco-monitor, Change-detector)
- **Contextual Memory**: Session-aware conversations with spatial context
- **Real-time Streaming**: Low-latency voice interaction

### 🏗️ Scalable Architecture
- **Microservices**: Containerized services with Docker/Kubernetes
- **Geospatial Database**: MongoDB with spatial indexing
- **Caching Layer**: Redis for performance optimization
- **Load Balancing**: Multi-GPU model parallelism support

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Python 3.9+
- Docker & Docker Compose
- NVIDIA GPU (optional, for AI acceleration)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd multimodal-eo-analysis
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Start with Docker Compose**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

4. **Manual setup (development)**
```bash
# Install dependencies
npm install

# Start backend
cd backend && npm install && npm run dev

# Start frontend (new terminal)
cd frontend && npm install && npm run dev

# Start AI services (new terminal)
cd ai-models && pip install -r requirements.txt && python src/multimodal_trainer.py

# Start data processing (new terminal)
cd data-processing && pip install -r requirements.txt && python src/isro_data_pipeline.py
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Grafana Monitoring**: http://localhost:3001
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
├── frontend/                 # React Three Fiber UI
│   ├── src/
│   │   ├── components/      # 3D Globe, Dashboard, Analysis panels
│   │   ├── stores/          # Zustand state management
│   │   ├── types/           # TypeScript definitions
│   │   └── utils/           # Utilities and mock data
│   └── Dockerfile
├── backend/                  # Node.js API server
│   ├── src/
│   │   ├── ai/             # Vision encoder integration
│   │   ├── controllers/    # API route handlers
│   │   ├── services/       # Business logic
│   │   └── types/          # TypeScript types
│   └── Dockerfile
├── ai-models/               # ML training and inference
│   ├── src/
│   │   ├── multimodal_trainer.py  # Training pipeline
│   │   └── inference.py           # Model serving
│   ├── configs/            # Model configurations
│   └── requirements.txt
├── data-processing/         # EO data pipeline
│   ├── src/
│   │   └── isro_data_pipeline.py  # ISRO data processing
│   ├── configs/            # Pipeline configurations
│   └── requirements.txt
├── docker/                  # Docker configurations
├── config/                  # Application configs
└── docker-compose.yml      # Service orchestration
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
MONGO_PASSWORD=your_mongo_password
REDIS_PASSWORD=your_redis_password

# API Keys
OPENAI_API_KEY=your_openai_key
VAPI_API_KEY=your_vapi_key
ISRO_API_KEY=your_isro_key

# AWS (for data storage)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=your_bucket_name

# Security
JWT_SECRET=your_jwt_secret
```

### AI Model Configuration
Edit `ai-models/configs/model_config.yaml`:
```yaml
vision_encoder: "openai/clip-vit-base-patch32"
language_model: "gpt2-medium"
projection_dim: 768
use_lora: true
batch_size: 16
learning_rate: 1e-4
```

### Data Pipeline Configuration
Edit `data-processing/configs/isro_pipeline_config.yaml`:
```yaml
spatial_bounds:
  min_lon: 68.0
  min_lat: 6.0
  max_lon: 98.0
  max_lat: 38.0
max_cloud_cover: 30.0
parallel_downloads: 4
```

## 🎯 Usage Examples

### 1. Query Satellite Data
```typescript
// Frontend - Query EO tiles
const { data: tiles } = useQuery({
  queryKey: ['eo-tiles', bounds],
  queryFn: () => api.searchTiles({
    startDate: '2024-01-01',
    endDate: '2024-01-31',
    bounds: [68, 6, 98, 38],
    maxCloudCover: 30
  })
})
```

### 2. Multimodal Analysis
```python
# AI Models - Process satellite imagery
from src.multimodal_trainer import MultimodalEOModel

model = MultimodalEOModel(config)
result = await model.analyze_image(
    image_buffer=satellite_image,
    query="What land cover types are visible in this area?"
)
```

### 3. Voice Interaction
```typescript
// Frontend - Voice query
const response = await voiceService.query({
  text: "Show me agricultural areas with high vegetation index",
  persona: "eco-analyst",
  context: {
    selectedTiles: ["tile_id_1", "tile_id_2"],
    spatialBounds: currentViewBounds
  }
})
```

## 🧪 Training Custom Models

### 1. Prepare Training Data
```bash
# Download and preprocess ISRO data
cd data-processing
python src/isro_data_pipeline.py --config configs/training_config.yaml
```

### 2. Train Vision-Language Alignment
```bash
# Train projection layer
cd ai-models
python src/multimodal_trainer.py --config configs/training_config.yaml
```

### 3. Fine-tune with LoRA
```python
# Use parameter-efficient fine-tuning
config = MultimodalConfig(
    use_lora=True,
    lora_r=16,
    freeze_vision_encoder=True,
    freeze_language_model=True
)
trainer = MultimodalTrainer(config)
trainer.train()
```

## 📊 Monitoring & Performance

### Metrics Dashboard
- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090

### Key Metrics
- Model inference latency
- Data processing throughput  
- API response times
- Memory and GPU utilization
- Active user sessions

### Performance Optimizations
- **Model Quantization**: 8-bit and 16-bit precision
- **Batch Inference**: Parallel processing
- **Caching**: Redis for frequent queries
- **CDN**: Static asset delivery
- **Lazy Loading**: On-demand component loading

## 🌐 API Reference

### EO Data Endpoints
```
GET  /api/tiles              # Search satellite tiles
POST /api/tiles/analyze      # Trigger analysis
GET  /api/analysis/:id       # Get analysis results
GET  /api/tiles/:id/download # Download tile data
```

### AI Analysis Endpoints
```
POST /api/ai/multimodal     # Multimodal analysis
POST /api/ai/caption        # Image captioning
POST /api/ai/vqa           # Visual question answering
POST /api/ai/classify      # Land cover classification
```

### Voice Interface Endpoints
```
POST /api/voice/query       # Voice query processing
GET  /api/voice/personas    # Available personas
POST /api/voice/tts         # Text-to-speech
POST /api/voice/stt         # Speech-to-text
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript/Python type hints
- Write comprehensive tests
- Update documentation
- Use conventional commits
- Ensure Docker builds pass

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ISRO/NRSC** for satellite data access
- **OpenAI** for CLIP model architecture
- **Hugging Face** for transformer models
- **Three.js** community for 3D visualization
- **React Three Fiber** for React integration

## 📞 Support

- **Documentation**: [Wiki](wiki-url)
- **Issues**: [GitHub Issues](issues-url)
- **Discussions**: [GitHub Discussions](discussions-url)
- **Email**: support@eo-analysis.com

---

**Built with ❤️ for Earth Observation and AI Research**
